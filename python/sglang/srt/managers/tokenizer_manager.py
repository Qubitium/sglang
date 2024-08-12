"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""TokenizerManager is a process that tokenizes the text."""

import asyncio
import concurrent.futures
import dataclasses
import logging
import multiprocessing as mp
import os
from typing import Dict, List, Tuple, Union

import numpy as np
import transformers
import uvloop
from fastapi import BackgroundTasks
from sglang.srt.hf_transformers_utils import (
    get_config,
    get_context_length,
    get_processor,
    get_tokenizer,
)
from sglang.srt.managers.io_struct import (
    AbortReq,
    BatchEmbeddingOut,
    BatchStrOut,
    BatchTokenIDOut,
    EmbeddingReqInput,
    FlushCacheReq,
    GenerateReqInput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.schedule_batch import FINISH_MATCHED_STR
from sglang.srt.mm_utils import expand2square, process_anyres_image
from sglang.srt.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import is_generation_model, is_multimodal_model, load_image
from sglang.utils import find_printable_text, get_exception_traceback


asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class DecodeStatus:
    vid: int
    decoded_text: str
    decode_ids: List[int]
    surr_offset: int
    read_offset: int


@dataclasses.dataclass
class ReqState:
    out_list: List[dict]
    finished: bool
    event: asyncio.Event


THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=8)


class TokenizerManager:
    def __init__(
            self,
            server_args: ServerArgs,
            router_chan: mp.Queue,
            detokenizer_chan: mp.Queue,
            idle_chan: mp.Queue,
            model_overide_args: dict = None,
    ):
        # num of pending requests
        self.pending = 0

        self.server_args = server_args
        self.router_chan = router_chan
        self.detokenizer_chan = detokenizer_chan
        self.idle_chan = idle_chan

        self.model_path = server_args.model_path
        self.served_model_name = server_args.served_model_name
        self.hf_config = get_config(
            self.model_path,
            trust_remote_code=server_args.trust_remote_code,
            model_overide_args=model_overide_args,
        )
        self.is_generation = is_generation_model(self.hf_config.architectures)

        if server_args.context_length is not None:
            self.context_len = server_args.context_length
        else:
            self.context_len = get_context_length(self.hf_config)

        if server_args.skip_tokenizer_init:
            self.tokenizer = self.processor = None
        else:
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
        self.rid_to_state: Dict[str, ReqState] = {}
        self.decoder_task = None

        self.decode_status = {}

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

    async def generate_request(self, obj: GenerateReqInput, request=None):
        await self.start()

        obj.post_init()
        is_single = obj.is_single

        if is_single:
            async for response in self._handle_single_request(obj, request):
                yield response
        else:
            if hasattr(obj, "stream") and obj.stream:
                raise ValueError("Do not support stream for batch mode.")

            async for response in self._handle_batch_request(obj, request):
                yield response

    async def _handle_single_request(
            self,
            obj: Union[GenerateReqInput, EmbeddingReqInput],
            request,
            index=None,
            is_cache_for_prefill=False,
    ):
        if not is_cache_for_prefill:  # The normal case with a single prompt
            not_use_index = index is None

            rid = obj.rid if not_use_index else obj.rid[index]
            input_text = obj.text if not_use_index else obj.text[index]
            if obj.input_ids is None:
                assert self.tokenizer is not None
                input_ids = self.tokenizer.encode(input_text)
            else:
                input_ids = obj.input_ids if not_use_index else obj.input_ids[index]

            self._validate_input_length(input_ids)

            sampling_params = self._get_sampling_params(
                obj.sampling_params if not_use_index else obj.sampling_params[index]
            )

            if self.is_generation:
                pixel_values, image_hash, image_size = await self._get_pixel_values(
                    obj.image_data if not_use_index else obj.image_data[index]
                )
                return_logprob = (
                    obj.return_logprob if not_use_index else obj.return_logprob[index]
                )
                logprob_start_len = (
                    obj.logprob_start_len
                    if not_use_index
                    else obj.logprob_start_len[index]
                )
                top_logprobs_num = (
                    obj.top_logprobs_num
                    if not_use_index
                    else obj.top_logprobs_num[index]
                )
        else:  # A prefill request to cache the common prompt for parallel sampling
            assert self.is_generation
            if obj.text is not None:
                if isinstance(obj.text, list):
                    input_text = obj.text[index]
                    rid = obj.rid[index]
                else:
                    input_text = obj.text
                    rid = obj.rid[0]
                if self.tokenizer is not None:
                    input_ids = self.tokenizer.encode(input_text)
                else:
                    assert obj.input_ids is not None
                    input_ids = obj.input_ids
                    if isinstance(obj.input_ids, list) and isinstance(
                            obj.input_ids[0], list
                    ):
                        # when obj["input_ids"] is List[List[int]]
                        input_ids = obj.input_ids[index]
                        rid = obj.rid[index]
                    else:
                        input_ids = obj.input_ids
                        rid = obj.rid[0]
            else:
                input_text = None
                if isinstance(obj.input_ids, list) and isinstance(
                        obj.input_ids[0], list
                ):
                    # when obj["input_ids"] is List[List[int]]
                    input_ids = obj.input_ids[index]
                    rid = obj.rid[index]
                else:
                    input_ids = obj.input_ids
                    rid = obj.rid[0]

            sampling_params = SamplingParams(**obj.sampling_params[0])
            sampling_params.max_new_tokens = 0
            pixel_values, image_hash, image_size = await self._get_pixel_values(
                obj.image_data[0]
            )
            return_logprob = obj.return_logprob[0]
            logprob_start_len = obj.logprob_start_len[0]
            top_logprobs_num = obj.top_logprobs_num[0]

        if self.is_generation:
            tokenized_obj = TokenizedGenerateReqInput(
                rid,
                input_text,
                input_ids,
                pixel_values,
                image_hash,
                image_size,
                sampling_params,
                return_logprob,
                logprob_start_len,
                top_logprobs_num,
                obj.stream,
            )
        else:  # is embedding
            tokenized_obj = TokenizedEmbeddingReqInput(
                rid,
                input_text,
                input_ids,
                sampling_params,
            )

        # print("tokenized_obj",tokenized_obj)
        # no need to wait
        asyncio.get_event_loop().run_in_executor(THREAD_POOL, self.router_chan.put_nowait, tokenized_obj)

        event = asyncio.Event()
        state = ReqState([], False, event)
        self.rid_to_state[rid] = state
        self.pending += 1
        # print(f"PENDING {self.pending}")
        # print(f"tokenizer generate_request single request")

        if not is_cache_for_prefill:
            async for response in self._wait_for_response(
                    event, state, obj, rid, request
            ):
                yield response
        else:
            assert self.is_generation
            await self._wait_for_cache_prefill_response(event, state, obj, rid, request)
            yield input_ids

    async def _handle_batch_request(
        self, obj: Union[GenerateReqInput, EmbeddingReqInput], request
    ):
        # print(f"tokenizer generate_request multiple request")
        batch_size = obj.batch_size
        if self.is_generation:
            parallel_sample_num = obj.parallel_sample_num
            if parallel_sample_num != 1:
                # Send prefill requests to cache the common input
                parallel_sample_num += 1
                input_id_result = [] if obj.input_ids is None else None
                for i in range(batch_size):
                    async for input_id in self._handle_single_request(
                        obj, request, index=i, is_cache_for_prefill=True
                    ):
                        if input_id_result is not None:
                            input_id_result.append(input_id)
                if input_id_result is not None and len(input_id_result) > 1:
                    obj.input_ids = input_id_result
                elif input_id_result is not None:
                    obj.input_ids = input_id_result[0]
        else:
            parallel_sample_num = 1

        # First send out all requests
        for i in range(batch_size):
            for j in range(parallel_sample_num):
                if j == 0 and parallel_sample_num != 1:
                    continue
                index = i * parallel_sample_num + j
                if parallel_sample_num != 1:
                    # Here when using parallel sampling we should consider prefill stage so the index is :  j + i * (parallel_sample_num-1) + batch_size - 1
                    index += batch_size - 1 - i
                rid = obj.rid[index]
                if parallel_sample_num == 1:
                    ## select operation
                    if obj.input_ids is None:
                        input_text = obj.text[i]
                        input_ids = self.tokenizer.encode(obj.text[i])
                    else:
                        input_text = None
                        input_ids = obj.input_ids[i]
                else:
                    assert obj.input_ids is not None
                    if batch_size == 1:
                        input_text = None
                        input_ids = obj.input_ids
                    else:
                        input_text = None
                        input_ids = obj.input_ids[i]
                sampling_params = self._get_sampling_params(obj.sampling_params[index])

                if self.is_generation:
                    pixel_values, image_hash, image_size = await self._get_pixel_values(
                        obj.image_data[index]
                    )

                    tokenized_obj = TokenizedGenerateReqInput(
                        rid,
                        input_text,
                        input_ids,
                        pixel_values,
                        image_hash,
                        image_size,
                        sampling_params,
                        obj.return_logprob[index],
                        obj.logprob_start_len[index],
                        obj.top_logprobs_num[index],
                        obj.stream,
                    )
                else:
                    tokenized_obj = TokenizedEmbeddingReqInput(
                        rid,
                        input_text,
                        input_ids,
                        sampling_params,
                    )
                # print(f"tokenizer generate_request router_chan put")
                # no need to wait
                asyncio.get_event_loop().run_in_executor(THREAD_POOL, self.router_chan.put_nowait, tokenized_obj)
                # print(f"tokenizer generate_request router_chan put done")

                event = asyncio.Event()
                state = ReqState([], False, event)
                self.rid_to_state[rid] = state

        # Then wait for all responses
        output_list = []
        for i in range(batch_size):
            for j in range(parallel_sample_num):
                if j == 0 and parallel_sample_num != 1:
                    continue
                index = i * parallel_sample_num + j
                if parallel_sample_num != 1:
                    index += batch_size - 1 - i
                rid = obj.rid[index]
                state = self.rid_to_state[rid]
                # print("tokenizer generate request multiple wait for event")
                try:
                    await asyncio.wait_for(state.event.wait(), timeout=4)
                    break
                except asyncio.TimeoutError:
                    if request is not None and await request.is_disconnected():
                        for rid in obj.rid:
                            self.abort_request(rid)
                        raise ValueError(f"Abort request {rid}")

                # print("tokenizer generate request multiple wait for event complete")

                while True:
                    try:
                        await asyncio.wait_for(state.event.wait(), timeout=4)
                        break
                    except asyncio.TimeoutError:
                        if request is not None and await request.is_disconnected():
                            for rid in obj.rid:
                                self.abort_request(rid)
                            raise ValueError(f"Abort request {rid}")
                        continue

                if self.is_generation:
                    output_list.append(
                        self.convert_logprob_style(
                            state.out_list[-1],
                            obj.return_logprob[index],
                            obj.top_logprobs_num[index],
                            obj.return_text_in_logprobs,
                        )
                    )
                else:
                    output_list.append(state.out_list[-1])
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
        yield output_list

    def _validate_input_length(self, input_ids: List[int]):
        if len(input_ids) >= self.context_len:
            raise ValueError(
                f"The input ({len(input_ids)} tokens) is longer than the "
                f"model's context length ({self.context_len} tokens)."
            )

    def _get_sampling_params(self, sampling_params_data: dict):
        sampling_params = SamplingParams(**sampling_params_data)
        if sampling_params.max_new_tokens != 0:
            sampling_params.normalize(self.tokenizer)
            sampling_params.verify()
        return sampling_params

    async def _get_pixel_values(self, image_data):
        if isinstance(image_data, list) and len(image_data) > 0:
            return await self.get_pixel_values(image_data[0])
        elif isinstance(image_data, str):
            return await self.get_pixel_values(image_data)
        else:
            return None, None, None

    async def _wait_for_response(
            self,
            event: asyncio.Event,
            state: ReqState,
            obj: Union[GenerateReqInput, EmbeddingReqInput],
            rid: str,
            request,
    ):
        # print(f"tokenizer _wait_for_response for event rid: {rid}")
        while True:
            try:
                await asyncio.wait_for(event.wait(), timeout=4)
            except asyncio.TimeoutError:
                if request is not None and await request.is_disconnected():
                    self.abort_request(rid)
                    raise ValueError(f"Abort request {rid}")
                continue

            if self.is_generation:
                out = self.convert_logprob_style(
                    state.out_list[-1],
                    obj.return_logprob,
                    obj.top_logprobs_num,
                    obj.return_text_in_logprobs,
                )
            else:  # isinstance(obj, EmbeddingReqInput)
                out = state.out_list[-1]

            # Log requests
            if self.server_args.log_requests and state.finished:
                if obj.text is None:
                    in_obj = {"input_ids": obj.input_ids}
                else:
                    in_obj = {"text": obj.text}
                logger.info(f"in={in_obj}, out={out}")

            # print(f"tokenizer generate request single wait for event done rid: {rid}")
            state.out_list = []
            if state.finished:
                self.pending -= 1
                assert self.pending >= 0
                if self.pending == 0:
                    # print("PENDING state.finished => empty rid_stats! signal!")
                    asyncio.get_event_loop().run_in_executor(THREAD_POOL, self.idle_chan.put_nowait, [True])
                else:
                    pass
                    # print(f"PENDING size: {self.pending}")

                del self.rid_to_state[rid]

                yield out

                break

            event.clear()
            yield out

    async def _wait_for_cache_prefill_response(
            self,
            event: asyncio.Event,
            state: ReqState,
            obj: GenerateReqInput,
            rid: str,
            request,
    ):
        # print(f"tokenizer _wait_for_prefill_response for event rid: {rid}")
        while True:
            try:
                await asyncio.wait_for(state.event.wait(), timeout=4)
                break
            except asyncio.TimeoutError:
                if request is not None and await request.is_disconnected():
                    for rid in obj.rid:
                        self.abort_request(rid)
                    raise ValueError(f"Abort request {rid}")
                continue

        assert state.finished
        self.pending -= 1
        assert self.pending >= 0
        if self.pending == 0:
            # print("PENDING state.finished => empty rid_stats! signal!")
            asyncio.get_event_loop().run_in_executor(THREAD_POOL, self.idle_chan.put_nowait, [True])
        else:
            pass
            # print(f"PENDING size: {self.pending}")

        del self.rid_to_state[rid]

    async def flush_cache(self):
        flush_cache_req = FlushCacheReq()
        # self.send_to_router.send_pyobj(flush_cache_req)
        asyncio.get_event_loop().run_in_executor(THREAD_POOL, self.router_chan.put_nowait, flush_cache_req)

    def abort_request(self, rid: str):
        if rid not in self.rid_to_state:
            return
        del self.rid_to_state[rid]
        req = AbortReq(rid)
        # self.send_to_router.send_pyobj(req)
        asyncio.get_event_loop().run_in_executor(THREAD_POOL, self.router_chan.put_nowait, req)

        # Pause the loop in ControllerSingle.loop_for_forward() to avoid wasting CPU resources.
        self.pending = 0
        # print("Abort request, signal to IDLE Chan!")
        asyncio.get_event_loop().run_in_executor(THREAD_POOL, self.idle_chan.put_nowait, [True])

    def create_abort_task(self, obj: GenerateReqInput):
        # Abort the request if the client is disconnected.
        async def abort_request():
            await asyncio.sleep(3)
            if obj.is_single:
                self.abort_request(obj.rid)
            else:
                for rid in obj.rids:
                    self.abort_request(rid)

        background_tasks = BackgroundTasks()
        background_tasks.add_task(abort_request)
        return background_tasks

    # def create_handle_loop(self):
    #     self.to_create_loop = False
    #     loop = asyncio.get_event_loop()
    #     loop.create_task(self.handle_loop())

    async def decoder_loop(self):
        print("in decoder_loop ")
        while True:
            recv_obj: Union[BatchStrOut, BatchEmbeddingOut, BatchTokenIDOut] = await self.recv_from_detokenizer()
            assert isinstance(
                recv_obj, (BatchStrOut, BatchEmbeddingOut, BatchTokenIDOut)
            ), f"Unexpected obj received: {type(recv_obj)}"
            # previously in another thread/loop
            for i, rid in enumerate(recv_obj.rids):
                state = self.rid_to_state.get(rid, None)
                if state is None:
                    continue

                recv_obj.meta_info[i]["id"] = rid
                if isinstance(recv_obj, BatchStrOut):
                    out_dict = {
                        "text": recv_obj.output_strs[i],
                        "meta_info": recv_obj.meta_info[i],
                    }
                elif isinstance(recv_obj, BatchTokenIDOut):
                    read_start = 0 if i == 0 else recv_obj.read_offsets[i - 1]
                    out_dict = {
                        "token_ids": recv_obj.decode_ids[
                                     read_start: recv_obj.read_offsets[i]
                                     ],
                        "meta_info": recv_obj.meta_info[i],
                    }

                else:
                    assert isinstance(recv_obj, BatchEmbeddingOut)
                    out_dict = {
                        "embedding": recv_obj.embeddings[i],
                        "meta_info": recv_obj.meta_info[i],
                    }

                state.out_list.append(out_dict)
                state.finished = recv_obj.finished_reason[i] is not None
                # print(f"tokenizer state.event.set ready rid: {rid}")
                state.event.set()
                # print(f"tokenizer state.event.set ready done rid: {rid}")

    async def recv_from_detokenizer(self) -> Union[BatchStrOut, BatchEmbeddingOut, BatchTokenIDOut]:
        # The following code is from handle_loop() in detokenizer_manager.py
        # print(f"detokenizer detokenizer_chan get wait...")
        recv_obj_detokenizer: BatchTokenIDOut = await asyncio.get_event_loop().run_in_executor(THREAD_POOL,
                                                                                               self.detokenizer_chan.get)
        if isinstance(recv_obj_detokenizer, BatchEmbeddingOut):
            return BatchEmbeddingOut(
                rids=recv_obj_detokenizer.rids,
                embeddings=recv_obj_detokenizer.embeddings,
                meta_info=recv_obj_detokenizer.meta_info,
                finished_reason=recv_obj_detokenizer.finished_reason,
            )
        else:
            assert isinstance(recv_obj_detokenizer, BatchTokenIDOut)
            # print(f"detokenizer detokenizer_chan get done: {recv_obj_detokenizer}")

            # The following code is from handle_loop() in detokenizer_manager.py
            bs = len(recv_obj_detokenizer.rids)

            if self.tokenizer is None:
                # Send BatchTokenIDOut if no tokenizer init'ed.
                return recv_obj_detokenizer
            else:
                # Initialize decode status
                read_ids, surr_ids = [], []
                for i in range(bs):
                    rid = recv_obj_detokenizer.rids[i]
                    vid = recv_obj_detokenizer.vids[i]
                    if rid not in self.decode_status or self.decode_status[rid].vid != vid:
                        s = DecodeStatus(
                            vid=vid,
                            decoded_text=recv_obj_detokenizer.decoded_texts[i],
                            decode_ids=recv_obj_detokenizer.decode_ids[i],
                            surr_offset=0,
                            read_offset=recv_obj_detokenizer.read_offsets[i],
                        )
                        self.decode_status[rid] = s
                    else:
                        s = self.decode_status[rid]
                        s.decode_ids = recv_obj_detokenizer.decode_ids[i]

                    read_ids.append(s.decode_ids[s.surr_offset:])
                    surr_ids.append(s.decode_ids[s.surr_offset: s.read_offset])

                    # TODO(lmzheng): handle skip_special_tokens/spaces_between_special_tokens per request
                    def batch_decode_surr():
                        return self.tokenizer.batch_decode(surr_ids,
                                                           skip_special_tokens=
                                                           recv_obj_detokenizer.skip_special_tokens[
                                                               0],
                                                           spaces_between_special_tokens=
                                                           recv_obj_detokenizer.spaces_between_special_tokens[0])

                    def batch_decode_read():
                        return self.tokenizer.batch_decode(read_ids,
                                                           skip_special_tokens=
                                                           recv_obj_detokenizer.skip_special_tokens[
                                                               0],
                                                           spaces_between_special_tokens=
                                                           recv_obj_detokenizer.spaces_between_special_tokens[0])

                    surr_texts = await asyncio.get_event_loop().run_in_executor(THREAD_POOL, batch_decode_surr)
                    read_texts = await asyncio.get_event_loop().run_in_executor(THREAD_POOL, batch_decode_read)

                    # Trim stop str
                    # TODO(lmzheng): handle the case where multiple stop strs are hit
                    output_strs = []
                    for i in range(bs):
                        s = self.decode_status[recv_obj_detokenizer.rids[i]]
                        new_text = read_texts[i][len(surr_texts[i]):]
                        if recv_obj_detokenizer.finished_reason[i] is None:
                            # Streaming chunk: update the decode status
                            if len(new_text) > 0 and not new_text.endswith("�"):
                                s.decoded_text = s.decoded_text + new_text
                                s.surr_offset = s.read_offset
                                s.read_offset = len(s.decode_ids)
                                new_text = ""
                            else:
                                new_text = find_printable_text(new_text)

                        output_strs.append(s.decoded_text + new_text)

                        if isinstance(recv_obj_detokenizer.finished_reason[i], FINISH_MATCHED_STR):
                            pos = output_strs[i].find(recv_obj_detokenizer.finished_reason[i].matched)
                            if pos != -1:
                                output_strs[i] = output_strs[i][:pos]

                    # print(f"detokenizer tokenizer_chan put",recv_obj_detokenizer.rids)
                    return BatchStrOut(
                        rids=recv_obj_detokenizer.rids,
                        output_strs=output_strs,
                        meta_info=recv_obj_detokenizer.meta_info,
                        finished_reason=recv_obj_detokenizer.finished_reason,
                    )

    def convert_logprob_style(
            self,
            ret: dict,
            return_logprob: bool,
            top_logprobs_num: int,
            return_text_in_logprobs: bool,
    ):
        if return_logprob:
            ret["meta_info"]["input_token_logprobs"] = self.detokenize_logprob_tokens(
                ret["meta_info"]["input_token_logprobs"], return_text_in_logprobs
            )
            ret["meta_info"]["output_token_logprobs"] = self.detokenize_logprob_tokens(
                ret["meta_info"]["output_token_logprobs"], return_text_in_logprobs
            )

            if top_logprobs_num > 0:
                ret["meta_info"]["input_top_logprobs"] = (
                    self.detokenize_top_logprobs_tokens(
                        ret["meta_info"]["input_top_logprobs"],
                        return_text_in_logprobs,
                    )
                )
                ret["meta_info"]["output_top_logprobs"] = (
                    self.detokenize_top_logprobs_tokens(
                        ret["meta_info"]["output_top_logprobs"], return_text_in_logprobs
                    )
                )
        return ret

    def detokenize_logprob_tokens(
            self, token_logprobs: List[Tuple[float, int]], decode_to_text: bool
    ):
        if not decode_to_text:
            return [(logprob, token_id, None) for logprob, token_id in token_logprobs]

        assert self.tokenizer is not None
        token_ids = [tid for _, tid in token_logprobs]
        token_texts = self.tokenizer.batch_decode(token_ids)
        return [
            (logprob, token_id, token_text)
            for (logprob, token_id), token_text, in zip(token_logprobs, token_texts)
        ]


    def detokenize_top_logprobs_tokens(self, top_logprobs, decode_to_text: bool):
        # TODO: The current implementation only batches the detokenization for top-k tokens per single position.
        # We should batch all top-k tokens in all positions.
        for i, token_top_logprobs in enumerate(top_logprobs):
            if token_top_logprobs:
                top_logprobs[i] = self.detokenize_logprob_tokens(
                    token_top_logprobs, decode_to_text
                )
        return top_logprobs


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
        image, image_size = load_image(image_data)
        if image_size is not None:
            image_hash = hash(image_data)
            pixel_values = processor.image_processor(image)["pixel_values"]
            for _ in range(len(pixel_values)):
                pixel_values[_] = pixel_values[_].astype(np.float16)
            pixel_values = np.stack(pixel_values, axis=0)
            return pixel_values, image_hash, image_size
        else:
            image_hash = hash(image_data)
            if image_aspect_ratio == "pad":
                image = expand2square(
                    image,
                    tuple(int(x * 255) for x in processor.image_processor.image_mean),
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
