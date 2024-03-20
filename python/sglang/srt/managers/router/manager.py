import logging
import multiprocessing as mp
import queue
import time

from sglang.srt.managers.router.model import ModelClient
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import get_exception_traceback

def flush_queue(q: mp.Queue):
    while q.qsize() > 0:
        try:
            q.get_nowait()
        except queue.Empty:
            pass

class RouterManager:
    def __init__(self, model_client: ModelClient, router_chan: mp.Queue, detokenzier_chan: mp.Queue, idle_chan: mp.Queue):
        # Init status
        self.model_client = model_client
        self.router_chan = router_chan
        self.detokenizer_chan = detokenzier_chan
        self.idle_chan = idle_chan

    def loop_for_forward(self):
        idle = True
        while True:
            if not idle:
                # print("forward check IDLE.....")
                if self.idle_chan.qsize() > 0:
                    print("forward GOT IDLE signal")
                    flush_queue(self.idle_chan)
                    idle = True


            next_step_input = []

            if idle:
                print("forward IDLE WAIT")
                next_step_input.append(self.router_chan.get())
                idle = False
                print("forward IDLE WAIT complete")
            else:
                pass
                # print("CPU SPIN LOOP")

            # non-blocking queue flush
            while self.router_chan.qsize() > 0:
                try:
                    next_step_input.append(self.router_chan.get_nowait())
                    idle = False
                    # try to merge requests
                    if self.router_chan.qsize() == 0:
                        # TODO FIXME make this configurable
                        time.sleep(0.01)
                except queue.Empty:
                    break

            # print(f"model_client.step wait...")
            if len(next_step_input) > 0:
                print(f"Forward Requests batch size: {len(next_step_input)}")

            output = self.model_client.step(next_step_input)
            # print(f"model_client.step done: {out_pyobjs}")

            for item in output:
                self.detokenizer_chan.put_nowait(item)

            # time.sleep(0.0004)
            # the model inference is empty
            # if len(output) == 0:
            #     # TODO FIXME make this configurable
            #     # prevent spin loop causing too much cpu usage
            #     time.sleep(0.0004)


def start_router_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    router_chan: mp.Queue,
    detokenizer_chan: mp.Queue,
    idle_chan: mp.Queue,
    startup_chan: mp.Queue,
):
    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    try:
        model_client = ModelClient(server_args, port_args)
        router = RouterManager(model_client, router_chan, detokenizer_chan, idle_chan)
    except Exception:
        startup_chan.put_nowait(get_exception_traceback())
        raise

    startup_chan.put_nowait("init ok")

    # blocking
    router.loop_for_forward()