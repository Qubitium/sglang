import asyncio
import concurrent.futures
import dataclasses
import logging
import multiprocessing as mp
import os
from typing import List
import numpy as np
import transformers
import uvloop
from sglang.srt.hf_transformers_utils import (
    get_config,
    get_context_length,
    get_processor,
    get_tokenizer,
)
from sglang.srt.managers.io_struct import (
    BatchStrOut,
    DetokenizeReqInput,
    FlushCacheReq,
    GenerateReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.mm_utils import expand2square, process_anyres_image
from sglang.srt.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_exception_traceback, is_multimodal_model, load_image

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ReqState:
    out_list: List[dict]
    finished: bool
    event: asyncio.Event


global global_processor

THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=8)


def init_global_processor(server_args: ServerArgs):
    global global_processor
    transformers.logging.set_verbosity_error()
    global_processor = get_processor(
        server_args.tokenizer_path,
        tokenizer_mode=server_args.tokenizer_mode,
        trust_remote_code=server_args.trust_remote_code,
    )


def get_pixel_values(
        image_data, image_aspect_ratio=None, image_grid_pinpoints=None, processor=None
):
    try:
        processor = processor or global_processor
        image = load_image(image_data)
        image_hash = hash(image_data)
        if image_aspect_ratio == "pad":
            image = expand2square(
                image, tuple(int(x * 255) for x in processor.image_processor.image_mean)
            )
            pixel_values = processor.image_processor(image)["pixel_values"][0]
        elif image_aspect_ratio == "anyres":
            pixel_values = process_anyres_image(
                image, processor.image_processor, image_grid_pinpoints
            )
        else:
            pixel_values = processor.image_processor(image)["pixel_values"][0]
        pixel_values = pixel_values.astype(np.float16)
        return pixel_values, image_hash, image.size
    except Exception:
        print("Exception in TokenizerManager:\n" + get_exception_traceback())


class TokenizerManager:
    def __init__(
            self,
            server_args: ServerArgs,
            router_chan: mp.Queue,
            detokenizer_chan: mp.Queue,
            idle_chan: mp.Queue,
    ):
        # num of pending requests
        self.pending = 0

        self.server_args = server_args
        self.router_chan = router_chan
        self.detokenizer_chan = detokenizer_chan
        self.idle_chan = idle_chan

        self.model_path = server_args.model_path
        self.hf_config = get_config(
            self.model_path, trust_remote_code=server_args.trust_remote_code
        )

        self.context_len = get_context_length(self.hf_config)

        if is_multimodal_model(self.model_path):
            self.processor = get_processor(
                server_args.tokenizer_path,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=server_args.trust_remote_code,
            )
            self.tokenizer = self.processor.tokenizer
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            self.executor = concurrent.futures.ProcessPoolExecutor(
                initializer=init_global_processor,
                mp_context=mp.get_context("fork"),
                initargs=(server_args,),
            )
        else:
            self.tokenizer = get_tokenizer(
                server_args.tokenizer_path,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=server_args.trust_remote_code,
            )

        self.to_create_loop = True
        self.rid_to_state = {}  # Dict[str -> ReqState]
        self.decoder_task = None

    async def start(self):
        if self.decoder_task is None:
            # TODO FIXME make sure sglang loads only the FAST tokenizers which rust based
            print("tokenizer generate_request decoder loop")
            self.decoder_task = asyncio.create_task(self.decoder_loop())

    async def get_pixel_values(self, image_data):
        aspect_ratio = getattr(self.hf_config, "image_aspect_ratio", None)
        grid_pinpoints = (
            self.hf_config.image_grid_pinpoints if aspect_ratio == "anyres" else None
        )
        if self.executor is not None:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor,
                get_pixel_values,
                image_data,
                aspect_ratio,
                grid_pinpoints,
            )
        else:
            return get_pixel_values(
                image_data, aspect_ratio, grid_pinpoints, self.processor
            )

    async def generate_request(self, obj: GenerateReqInput):
        await self.start()

        is_single = obj.is_single
        if is_single:
            self.pending += 1
            # print(f"PENDING {self.pending}")
            # print(f"tokenizer generate_request single request")
            rid = obj.rid

            if obj.input_ids is None:
                input_ids = await asyncio.get_event_loop().run_in_executor(THREAD_POOL, self.tokenizer.encode, obj.text)
            else:
                input_ids = obj.input_ids

            sampling_params = SamplingParams(**obj.sampling_params)
            if sampling_params.max_new_tokens != 0:
                sampling_params.normalize(self.tokenizer)
                sampling_params.verify()

            if isinstance(obj.image_data, list) and len(obj.image_data) > 0:
                pixel_values, image_hash, image_size = await self.get_pixel_values(
                    obj.image_data[0]
                )
            elif isinstance(obj.image_data, str):
                pixel_values, image_hash, image_size = await self.get_pixel_values(
                    obj.image_data
                )
            else:
                pixel_values, image_hash, image_size = None, None, None
            tokenized_obj = TokenizedGenerateReqInput(
                rid=rid,
                input_text=obj.text,
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_hash=image_hash,
                image_size=image_size,
                sampling_params=sampling_params,
                return_logprob=obj.return_logprob,
                logprob_start_len=obj.logprob_start_len,
                top_logprobs_num=obj.top_logprobs_num,
                stream=obj.stream,
            )

            event = asyncio.Event()
            state = ReqState([], False, event)
            self.rid_to_state[rid] = state

            # no need to wait
            asyncio.get_event_loop().run_in_executor(THREAD_POOL, self.router_chan.put_nowait, tokenized_obj)

            # print(f"tokenizer generate request single wait for event rid: {rid}")
            await event.wait()

            self.pending -= 1
            assert self.pending >= 0
            if self.pending == 0:
                # print("PENDING state.finished => empty rid_stats! signal!")
                asyncio.get_event_loop().run_in_executor(THREAD_POOL, self.idle_chan.put_nowait, [True])
            else:
                pass
                # print(f"PENDING size: {self.pending}")

            assert state.finished
            del self.rid_to_state[rid]

            # print(f"tokenizer generate request single wait for event done rid: {rid}")
            yield self.convert_logprob_style(state.out_list[-1],
                                                 obj.return_logprob,
                                                 obj.top_logprobs_num,
                                                 obj.return_text_in_logprobs)

        else:
            # print(f"tokenizer generate_request multiple request")
            assert obj.stream is False

            bs = len(obj.text)
            self.pending += bs
            # print(f"PENDING {self.pending}")

            for i in range(bs):
                rid = obj.rid[i]

                if obj.input_ids is None:
                    input_text = obj.text[i]
                    input_ids = self.tokenizer.encode(obj.text[i])
                else:
                    input_text = None
                    input_ids = obj.input_ids[i]

                sampling_params = SamplingParams(**obj.sampling_params[i])
                if sampling_params.max_new_tokens != 0:
                    sampling_params.normalize(self.tokenizer)
                    sampling_params.verify()
                if obj.image_data[i] is None:
                    pixel_values, image_hash, image_size = None, None, None
                else:
                    pixel_values, image_hash, image_size = await self.get_pixel_values(
                        obj.image_data[i]
                    )
                tokenized_obj = TokenizedGenerateReqInput(
                    rid=rid,
                    input_text=input_text,
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    image_hash=image_hash,
                    image_size=image_size,
                    sampling_params=sampling_params,
                    return_logprob=obj.return_logprob[i],
                    logprob_start_len=obj.logprob_start_len[i],
                    top_logprobs_num=obj.top_logprobs_num[i],
                    stream=obj.stream,
                )

                # print(f"tokenizer generate_request router_chan put")
                # no need to wait
                asyncio.get_event_loop().run_in_executor(THREAD_POOL, self.router_chan.put_nowait, tokenized_obj)
                # print(f"tokenizer generate_request router_chan put done")

                event = asyncio.Event()
                state = ReqState([], False, event)
                self.rid_to_state[rid] = state

            output_list = []
            for i in range(bs):
                rid = obj.rid[i]
                state = self.rid_to_state[rid]
                # print("tokenizer generate request multiple wait for event")
                await state.event.wait()

                # print("tokenizer generate request multiple wait for event complete")
                output_list.append(
                    self.convert_logprob_style(state.out_list[-1],
                                               obj.return_logprob[i],
                                               obj.top_logprobs_num[i],
                                               obj.return_text_in_logprobs))
                assert state.finished

                del self.rid_to_state[rid]

                self.pending -= 1
                assert self.pending >= 0
                if self.pending == 0:
                    # print("PENDING state.finished => empty rid_stats! signal!")
                    asyncio.get_event_loop().run_in_executor(THREAD_POOL, self.idle_chan.put_nowait, [True])
                else:
                    pass
                    # print(f"PENDING size: {self.pending}")

            yield output_list

    def detokenize(self, obj: DetokenizeReqInput):
        token_texts = self.tokenizer.convert_ids_to_tokens(obj.input_ids)
        return [t.decode() if isinstance(t, bytes) else t for t in token_texts]

    async def flush_cache(self):
        flush_cache_req = FlushCacheReq()
        # self.send_to_router.send_pyobj(flush_cache_req)
        self.router_chan.put_nowait(flush_cache_req)

    async def decoder_loop(self):
        print("in decoder_loop ")
        while True:
            # print(f"detokenizer detokenizer_chan get wait...")
            recv_obj = await asyncio.get_event_loop().run_in_executor(THREAD_POOL, self.detokenizer_chan.get)
            # print(f"detokenizer detokenizer_chan get done: {recv_obj}")

            output_tokens = recv_obj.output_tokens

            # TODO(lmzheng): handle skip_special_tokens per request
            output_strs = await asyncio.get_event_loop().run_in_executor(THREAD_POOL, self.tokenizer.batch_decode,
                                                                         output_tokens, recv_obj.skip_special_tokens[0]
                                                                         )

            # Trim stop str
            # TODO(lmzheng): handle the case where multiple stop strs are hit
            for i in range(len(output_strs)):
                if recv_obj.hit_stop_str[i] is not None:
                    pos = output_strs[i].find(recv_obj.hit_stop_str[i])
                    if pos != -1:
                        output_strs[i] = output_strs[i][:pos]

                if len(output_tokens[i]) > 0:
                    first_token = await asyncio.get_event_loop().run_in_executor(THREAD_POOL,
                                                                                 self.tokenizer.convert_ids_to_tokens,
                                                                                 int(output_tokens[i][0])
                                                                                 )

                    if not isinstance(first_token, str):
                        first_token = first_token.decode("utf-8", errors="ignore")
                    if first_token.startswith("▁"):
                        output_strs[i] = " " + output_strs[i]

                output_strs[i] = (
                        recv_obj.output_and_jump_forward_strs[i] + output_strs[i]
                )

            # print(f"detokenizer tokenizer_chan put")
            output = BatchStrOut(
                recv_obj.rids,
                output_strs,
                recv_obj.meta_info,
                recv_obj.finished,
            )

            # previously in another thread/loop
            for i, rid in enumerate(output.rids):
                output.meta_info[i]["id"] = rid
                out_dict = {
                    "text": output.output_str[i],
                    "meta_info": output.meta_info[i],
                }
                state = self.rid_to_state[rid]

                state.out_list.append(out_dict)
                state.finished = output.finished[i]
                # print(f"tokenizer state.event.set ready rid: {rid}")
                state.event.set()
                # print(f"tokenizer state.event.set ready done rid: {rid}")

    def convert_logprob_style(self, ret, return_logprob, top_logprobs_num, return_text_in_logprobs):
        if return_logprob:
            ret["meta_info"]["prefill_token_logprobs"] = self.detokenize_logprob_tokens(
                ret["meta_info"]["prefill_token_logprobs"], return_text_in_logprobs
            )
            ret["meta_info"]["decode_token_logprobs"] = self.detokenize_logprob_tokens(
                ret["meta_info"]["decode_token_logprobs"], return_text_in_logprobs
            )
        if top_logprobs_num > 0:
            ret["meta_info"]["prefill_top_logprobs"] = self.detokenize_top_logprobs_tokens(
                ret["meta_info"]["prefill_top_logprobs"], return_text_in_logprobs
            )
            ret["meta_info"]["decode_top_logprobs"] = self.detokenize_top_logprobs_tokens(
                ret["meta_info"]["decode_top_logprobs"], return_text_in_logprobs
            )
        return ret

    def detokenize_logprob_tokens(self, token_logprobs, decode_to_text):
        if not decode_to_text:
            return [(logprob, token_id, None) for logprob, token_id in token_logprobs]

        token_ids = [tid for _, tid in token_logprobs]
        token_texts = self.tokenizer.batch_decode(token_ids)
        return [
            (logprob, token_id, token_text)
            for (logprob, token_id), token_text, in zip(token_logprobs, token_texts)
        ]

    def detokenize_top_logprobs_tokens(self, top_logprobs, decode_to_text):
        for i, t in enumerate(top_logprobs):
            if t:
                top_logprobs[i] = self.detokenize_logprob_tokens(t, decode_to_text)
        return top_logprobs
