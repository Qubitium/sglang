import asyncio
import logging
import multiprocessing
import threading
import time

import uvloop
from sglang.srt.backend_config import GLOBAL_BACKEND_CONFIG
from sglang.srt.managers.router.model_rpc import ModelRpcClient
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import get_exception_traceback
from sglang.srt.utils import make_async_thread
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


class RouterManager:
    def __init__(self, model_client: ModelRpcClient, router_chan: multiprocessing.Queue, detokenzier_chan: multiprocessing.Queue):
        # Init status
        self.model_client = model_client
        self.router_chan = router_chan
        self.detokenizer_chan = detokenzier_chan

        self.recv_reqs = []
        self.lock = threading.Lock()

        # Init some configs
        self.extend_dependency_time = GLOBAL_BACKEND_CONFIG.extend_dependency_time

    def loop_for_forward(self):
        while True:
            with self.lock:
                next_step_input = list(self.recv_reqs)
                self.recv_reqs = []

            # print(f"model_client.step wait...")
            out_pyobjs = self.model_client.step(next_step_input)
            # print(f"model_client.step done: {out_pyobjs}")

            for obj in out_pyobjs:
                self.detokenizer_chan.put_nowait(obj)

            # async sleep for receiving the subsequent request and avoiding cache miss
            # if len(out_pyobjs) != 0:
            #     has_finished = any([obj.finished for obj in out_pyobjs])
            #     if has_finished:
            #         time.sleep(self.extend_dependency_time)

            # TODO FIXME
            time.sleep(0.0006)

    def loop_for_recv_requests(self):
       #  router_get = make_async_thread(self.router_chan.get)

        while True:
            # print(f"loop_for_recv_requests wait...")
            recv_req = self.router_chan.get()
            # print(f"loop_for_recv_requests got: {recv_req}")
            with self.lock:
                self.recv_reqs.append(recv_req)


def start_router_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    router_chan: multiprocessing.Queue,
    detokenizer_chan: multiprocessing.Queue,
    startup_chan: multiprocessing.Queue,
):
    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    try:
        model_client = ModelRpcClient(server_args, port_args)
        router = RouterManager(model_client, router_chan, detokenizer_chan)
    except Exception:
        startup_chan.put_nowait(get_exception_traceback())
        raise

    startup_chan.put_nowait("init ok")

    #loop = asyncio.new_event_loop()
    #asyncio.set_event_loop(loop)
    # loop.create_task(router.loop_for_recv_requests())
    threading.Thread(target=router.loop_for_recv_requests, daemon=True).start()
    router.loop_for_forward()
    # loop.run_until_complete(router.loop_for_forward())
