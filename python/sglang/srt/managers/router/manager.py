import logging
import multiprocessing
import queue
import time

from sglang.srt.managers.router.model import ModelClient
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import get_exception_traceback


class RouterManager:
    def __init__(self, model_client: ModelClient, router_chan: multiprocessing.Queue, detokenzier_chan: multiprocessing.Queue):
        # Init status
        self.model_client = model_client
        self.router_chan = router_chan
        self.detokenizer_chan = detokenzier_chan

    def loop_for_forward(self):
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
            output = self.model_client.step(next_step_input)
            # print(f"model_client.step done: {out_pyobjs}")

            for item in output:
                self.detokenizer_chan.put_nowait(item)

            # the model inference is empty
            if len(output) == 0:
                # prevent spin loop causing too much cpu usage
                time.sleep(0.0004)


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
        model_client = ModelClient(server_args, port_args)
        router = RouterManager(model_client, router_chan, detokenizer_chan)
    except Exception:
        startup_chan.put_nowait(get_exception_traceback())
        raise

    startup_chan.put_nowait("init ok")

    # blocking
    router.loop_for_forward()