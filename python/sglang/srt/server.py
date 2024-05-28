"""SRT: SGLang Runtime"""

import asyncio
import dataclasses
import json
import logging
import multiprocessing as mp
import os
import sys
import threading
import time
from http import HTTPStatus
from typing import Optional

# Fix a bug of Python threading
setattr(threading, "_register_atexit", lambda *args, **kwargs: None)
mp.set_start_method('spawn', force=True)

import aiohttp
import psutil
import requests
import uvloop
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from sglang.backend.runtime_endpoint import RuntimeEndpoint
from sglang.srt.constrained import disable_cache
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.managers.detokenizer_manager import start_detokenizer_process
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.controller.manager_single import start_controller_process as start_controller_process_single
from sglang.srt.managers.controller.manager_multi import start_controller_process as start_controller_process_multi
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.openai_api_adapter import (
    load_chat_template_for_openai_api,
    v1_chat_completions,
    v1_completions,
)
from sglang.srt.server_args import ModelPortArgs, PortArgs, ServerArgs
from sglang.srt.utils import (
    API_KEY_HEADER_NAME,
    APIKeyValidatorMiddleware,
    allocate_init_ports,
    assert_pkg_version,
    enable_show_time_cost,
)
from sglang.utils import get_exception_traceback

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

API_KEY_HEADER_NAME = "X-API-Key"


# class APIKeyValidatorMiddleware(BaseHTTPMiddleware):
#     def __init__(self, app, api_key: str):
#         super().__init__(app)
#         self.api_key = api_key
#
#     async def dispatch(self, request: Request, call_next):
#         # extract API key from the request headers
#         api_key_header = request.headers.get(API_KEY_HEADER_NAME)
#         if not api_key_header or api_key_header != self.api_key:
#             return JSONResponse(
#                 status_code=403,
#                 content={"detail": "Invalid API Key"},
#             )
#         response = await call_next(request)
#         return response


# app = FastAPI()
tokenizer_manager = None

# @app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


# @app.get("/get_model_info")
async def get_model_info():
    result = {
        "model_path": tokenizer_manager.model_path,
    }
    return result


# @app.get("/get_server_args")
async def get_server_args():
    return dataclasses.asdict(tokenizer_manager.server_args)


# @app.get("/flush_cache")
async def flush_cache():
    tokenizer_manager.flush_cache()
    return Response(
        content="Cache flushed.\nPlease check backend logs for more details. "
                "(When there are running or waiting requests, the operation will not be performed.)\n",
        status_code=200,
    )


async def detokenize_logprob_tokens(token_logprobs, decode_to_text):
    if not decode_to_text:
        return [(logprob, token_id, None) for logprob, token_id in token_logprobs]

    token_ids = [tid for _, tid in token_logprobs]
    token_texts = await tokenizer_manager.detokenize(DetokenizeReqInput(token_ids))
    return [
        (logprob, token_id, token_text)
        for (logprob, token_id), token_text, in zip(token_logprobs, token_texts)
    ]


async def detokenize_top_logprobs_tokens(top_logprobs, decode_to_text):
    for i, t in enumerate(top_logprobs):
        if top_logprobs[i] is not None:
            top_logprobs[i] = await detokenize_logprob_tokens(t, decode_to_text)
    return top_logprobs


async def handle_token_logprobs_results(obj: GenerateReqInput, ret):
    """Handle the token logprobs results, convert token ids to text if needed.

    Args:
        obj (GenerateReqInput): The request object.
        ret (Union[Dict, List[Dict]]): The response object.
    """
    # NOTE: This is because the multiple requests in one http request.

    async def convert_style(r, return_text):
        r["meta_info"]["prefill_token_logprobs"] = await detokenize_logprob_tokens(
            r["meta_info"]["prefill_token_logprobs"], return_text
        )
        r["meta_info"]["decode_token_logprobs"] = await detokenize_logprob_tokens(
            r["meta_info"]["decode_token_logprobs"], return_text
        )
        r["meta_info"]["prefill_top_logprobs"] = await detokenize_top_logprobs_tokens(
            r["meta_info"]["prefill_top_logprobs"], return_text
        )
        r["meta_info"]["decode_top_logprobs"] = await detokenize_top_logprobs_tokens(
            r["meta_info"]["decode_top_logprobs"], return_text
        )

    if isinstance(obj.text, str):
        if obj.return_logprob:
            await convert_style(ret, obj.return_text_in_logprobs)
    else:
        for i, r in enumerate(ret):
            if obj.return_logprob[i]:
                await convert_style(r, obj.return_text_in_logprobs)


async def stream_generator(obj: GenerateReqInput):
    async for out in tokenizer_manager.generate_request(obj):
        await handle_token_logprobs_results(obj, out)
        yield out


async def generate_request(obj: GenerateReqInput, request: Request = None):
    if obj.stream:

        async def stream_results():
            try:
                async for out in tokenizer_manager.generate_request(obj, request):
                    yield f"data: {json.dumps(out, ensure_ascii=False)}\n\n"
            except ValueError as e:
                out = {"error": {"message": str(e)}}
                yield f"data: {json.dumps(out, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_results(), media_type="text/event-stream",
                                 background=tokenizer_manager.create_abort_task(obj))
    else:
        try:
            ret = await tokenizer_manager.generate_request(obj, request).__anext__()
            return ret
        except ValueError as e:
            return JSONResponse(
                {"error": {"message": str(e)}}, status_code=HTTPStatus.BAD_REQUEST
            )


# app.post("/generate")(generate_request)
# app.put("/generate")(generate_request)


#@app.post("/v1/completions")
async def openai_v1_completions(raw_request: Request):
    return await v1_completions(tokenizer_manager, raw_request)


#@app.post("/v1/chat/completions")
async def openai_v1_chat_completions(raw_request: Request):
    return await v1_chat_completions(tokenizer_manager, raw_request)


def launch_server(server_args, tokenizer_init_chan, pipe_finish_writer=None, model_overide_args=None):
    print("launch_server started.... ")
    global tokenizer_manager

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    # Set global environments
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    if server_args.show_time_cost:
        enable_show_time_cost()
    if server_args.disable_disk_cache:
        disable_cache()
    if server_args.enable_flashinfer:
        assert_pkg_version("flashinfer", "0.0.4")
    if server_args.chat_template:
        # TODO: replace this with huggingface transformers template
        load_chat_template_for_openai_api(server_args.chat_template)

    # Allocate ports
    server_args.port, server_args.additional_ports = allocate_init_ports(
        server_args.port,
        server_args.additional_ports,
        server_args.tp_size,
        server_args.dp_size,
    )

    # Init local models port args
    ports = server_args.additional_ports
    tp = server_args.tp_size
    model_port_args = []
    for i in range(server_args.dp_size):
        model_port_args.append(
            ModelPortArgs(
                nccl_port=ports[3 + i * (tp + 1)],
                model_tp_ports=ports[3 + i * (tp + 1) + 1 : 3 + (i + 1) * (tp + 1)],
            )
        )
    port_args = PortArgs(
        tokenizer_port=ports[0],
        router_port=ports[1],
        detokenizer_port=ports[2],
        model_port_args=model_port_args,
    )
    router_chan = mp.Queue()
    detokenizer_chan = mp.Queue()
    idle_chan = mp.Queue()
    startup_chan = mp.Queue()


    # Launch processes
    tokenizer_manager = TokenizerManager(server_args, router_chan, detokenizer_chan, idle_chan, model_overide_args)

    tokenizer_init_chan.put_nowait("init ok")

    if server_args.dp_size == 1:
        start_process = start_controller_process_single
    else:
        start_process = start_controller_process_multi
    proc_router = mp.Process(
        args=(
            server_args,
            port_args,
            router_chan,
            detokenizer_chan,
            idle_chan,
            startup_chan,
            model_overide_args
        ),
        target=start_process,
    )
    proc_router.start()

    # Wait for the model to finish loading
    router_init_state = startup_chan.get()

    if router_init_state != "init ok":
        proc_router.kill()
        print(
            f"Initialization failed. router_init_state: {router_init_state}", flush=True
        )
        sys.exit(1)

    assert proc_router.is_alive()

    # if server_args.api_key and server_args.api_key != "":
    #     app.add_middleware(APIKeyValidatorMiddleware, api_key=server_args.api_key)

    # def _launch_server():
    #     uvicorn.run(
    #         app,
    #         host=server_args.host,
    #         port=server_args.port,
    #         log_level=server_args.log_level,
    #         timeout_keep_alive=5,
    #         loop="uvloop",
    #     )

    # Send a warmup request
    def _wait_and_warmup():
        headers = {}
        url = server_args.url()
        if server_args.api_key:
            headers[API_KEY_HEADER_NAME] = server_args.api_key

        # Wait until the server is launched
        for _ in range(120):
            time.sleep(0.5)
            try:
                requests.get(url + "/get_model_info", timeout=5, headers=headers)
                break
            except requests.exceptions.RequestException as e:
                pass

        # Send a warmup request
        try:
            res = requests.post(
                url + "/generate",
                json={
                    "text": "The capital city of France is",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 16,
                    },
                },
                headers=headers,
                timeout=600,
            )
            assert res.status_code == 200
        except Exception as e:
            if pipe_finish_writer is not None:
                pipe_finish_writer.send(get_exception_traceback())
            print(f"Initialization failed. warmup error: {e}")
            raise e

        if pipe_finish_writer is not None:
            pipe_finish_writer.send("init ok")

    # t = threading.Thread(target=_wait_and_warmup)
    # t.start()
    #
    # # Listen for requests
    # try:
    #     uvicorn.run(
    #         app,
    #         host=server_args.host,
    #         port=server_args.port,
    #         log_level=server_args.log_level,
    #         timeout_keep_alive=5,
    #         loop="uvloop",
    #     )
    # finally:
    #     t.join()


class Runtime:
    def __init__(
        self,
        tokenizer_init_chan: mp.Queue,
        log_level: str = "error",
        model_overide_args: Optional[dict] = None,
        *args,
        **kwargs,
    ):
        """See the arguments in server_args.py::ServerArgs"""
        self.server_args = ServerArgs(*args, log_level=log_level, **kwargs)

        # Pre-allocate ports
        self.server_args.port, self.server_args.additional_ports = allocate_init_ports(
            self.server_args.port,
            self.server_args.additional_ports,
            self.server_args.tp_size,
            self.server_args.dp_size,
        )

        self.url = self.server_args.url()
        self.generate_url = (
            f"http://{self.server_args.host}:{self.server_args.port}/generate"
        )

        # self.pid = None
        # pipe_reader, pipe_writer = mp.Pipe(duplex=False)
        # proc = mp.Process(
        #     target=launch_server,
        #     args=(self.server_args, pipe_writer, model_overide_args),
        # )
        # proc.start()
        # pipe_writer.close()
        # self.pid = proc.pid

        self.pid = os.getpid()

        threading.Thread(target=launch_server, args=[self.server_args, tokenizer_init_chan, None, model_overide_args], daemon=True).start()

        # try:
        #     init_state = pipe_reader.recv()
        # except EOFError:
        #     init_state = ""

        # if init_state != "init ok":
        #     self.shutdown()
        #     raise RuntimeError(
        #         "Initialization failed. Please see the error messages above."
        #     )

    def shutdown(self):
        if self.pid is not None:
            try:
                parent = psutil.Process(self.pid)
            except psutil.NoSuchProcess:
                return
            children = parent.children(recursive=True)
            for child in children:
                child.kill()
            psutil.wait_procs(children, timeout=5)
            # FIXME Why are we killing parent?
            # parent.kill()
            # parent.wait(timeout=5)
            self.pid = None

    # def get_tokenizer(self):
    #     return get_tokenizer(
    #         self.server_args.tokenizer_path,
    #         tokenizer_mode=self.server_args.tokenizer_mode,
    #         trust_remote_code=self.server_args.trust_remote_code,
    #     )

    async def add_request(
            self,
            prompt: str,
            sampling_params,
    ) -> None:
        json_data = {
            "text": prompt,
            "sampling_params": sampling_params,
            "stream": True,
        }
        pos = 0

        timeout = aiohttp.ClientTimeout(total=3 * 3600)
        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
            async with session.post(self.generate_url, json=json_data) as response:
                async for chunk, _ in response.content.iter_chunks():
                    chunk = chunk.decode("utf-8")
                    if chunk and chunk.startswith("data:"):
                        if chunk == "data: [DONE]\n\n":
                            break
                        data = json.loads(chunk[5:].strip("\n"))
                        cur = data["text"][pos:]
                        if cur:
                            yield cur
                        pos += len(cur)

    def __del__(self):
        self.shutdown()
