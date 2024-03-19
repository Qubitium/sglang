import asyncio
import concurrent.futures
import dataclasses
import multiprocessing
import multiprocessing as mp
import os
import threading
from typing import List
from sglang.srt.utils import make_async_thread
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
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import get_exception_traceback, is_multimodal_model, load_image

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

@dataclasses.dataclass
class ReqState:
    out: List[dict]
    finished: bool
    event: asyncio.Event


global global_processor


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
            tokenizer_chan: multiprocessing.Queue,
            router_chan: multiprocessing.Queue,
    ):
        self.server_args = server_args
        self.tokenizer_chan = tokenizer_chan
        self.router_chan = router_chan

        self.lock = threading.Lock()

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

        self.model_output_thread = None
        self.rid_to_state = {}  # Dict[str -> ReqState]

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
        if self.model_output_thread is None:

            print(f"tokenizer generate_request create handel loop")
            # await self.create_handle_loop()
            self.model_output_thread = threading.Thread(target=self.model_output_loop, daemon=True)
            self.model_output_thread.start()

        is_single = isinstance(obj.text, str)

        if is_single:
            # print(f"tokenizer generate_request single request")
            rid = obj.rid
            input_ids = self.tokenizer.encode(obj.text)
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
                stream=obj.stream,
            )

            event = asyncio.Event()
            state = ReqState([], False, event)
            with self.lock:
                self.rid_to_state[rid] = state

            self.router_chan.put_nowait(tokenized_obj)

            while True:
                # print(f"tokenizer generate request single wait for event rid: {rid}")
                await event.wait()
                # print(f"tokenizer generate request single wait for event done rid: {rid}")
                yield state.out_list[-1]
                state.out_list = []
                if state.finished:
                    with self.lock:
                        del self.rid_to_state[rid]
                    break
                event.clear()
        else:
            # print(f"tokenizer generate_request multiple request")
            assert obj.stream is False
            bs = len(obj.text)
            for i in range(bs):
                rid = obj.rid[i]
                input_ids = self.tokenizer.encode(obj.text[i])
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
                    input_text=obj.text[i],
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    image_hash=image_hash,
                    image_size=image_size,
                    sampling_params=sampling_params,
                    return_logprob=obj.return_logprob[i],
                    logprob_start_len=obj.logprob_start_len[i],
                    stream=obj.stream,
                )

                # print(f"tokenizer generate_request router_chan put")
                self.router_chan.put_nowait(tokenized_obj)
                # print(f"tokenizer generate_request router_chan put done")

                event = asyncio.Event()
                state = ReqState([], False, event)
                with self.lock:
                    self.rid_to_state[rid] = state

            output_list = []
            for i in range(bs):
                rid = obj.rid[i]
                with self.lock:
                    state = self.rid_to_state[rid]
                # print("tokenizer generate request multiple wait for event")
                await state.event.wait()
                # print("tokenizer generate request multiple wait for event complete")
                output_list.append(state.out_list[-1])
                assert state.finished
                with self.lock:
                    del self.rid_to_state[rid]

            yield output_list

    def detokenize(self, obj: DetokenizeReqInput):
        token_texts = self.tokenizer.convert_ids_to_tokens(obj.input_ids)
        return [t.decode() if isinstance(t, bytes) else t for t in token_texts]

    async def flush_cache(self):
        flush_cache_req = FlushCacheReq()
        # self.send_to_router.send_pyobj(flush_cache_req)
        self.router_chan.put_nowait(flush_cache_req)

    def model_output_loop(self):
        # tokenizer_get = make_async_thread(self.tokenizer_chan.get)

        while True:
            # print(f"tokenizer manager tokenizer_chan get waiting...")
            output = self.tokenizer_chan.get()
            # print(f"tokenizer manager tokenizer_chan get done: {recv_obj}")

            for i, rid in enumerate(output.rids):
                output.meta_info[i]["id"] = rid
                out_dict = {
                    "text": output.output_str[i],
                    "meta_info": output.meta_info[i],
                }
                with self.lock:
                    state = self.rid_to_state[rid]

                state.out_list.append(out_dict)
                state.finished = output.finished[i]
                # print(f"tokenizer state.event.set ready rid: {rid}")
                state.event.set()
                # print(f"tokenizer state.event.set ready done rid: {rid}")
