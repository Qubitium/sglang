import logging
import multiprocessing
import queue
import time

from sglang.srt.managers.router.model_rpc import ModelRpcClient
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import get_exception_traceback


class RouterManager:
    def __init__(self, model_client: ModelRpcClient, router_chan: multiprocessing.Queue, detokenzier_chan: multiprocessing.Queue):
        # Init status
        self.model_client = model_client
        self.router_chan = router_chan
        self.detokenizer_chan = detokenzier_chan

    def loop_for_forward(self):
        idle = False
        while True:
            next_step_input = []

            # TODO FIX ME this blocks processing
            # if idle and self.router_chan.qsize() == 0:
            #     next_step_input.append(self.router_chan.get())

            # non-blocking queue flush
            while self.router_chan.qsize() > 0:
                try:
                    next_step_input.append(self.router_chan.get_nowait())
                except queue.Empty:
                    break

            # print(f"model_client.step wait...")
            out_pyobjs = self.model_client.step(next_step_input)
            # print(f"model_client.step done: {out_pyobjs}")

            for obj in out_pyobjs:
                self.detokenizer_chan.put_nowait(obj)

            # the model inference is empty
            if len(out_pyobjs) == 0:
                idle = True
                time.sleep(0.0001)
            else:
                idle = False


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

    # blocking
    router.loop_for_forward()