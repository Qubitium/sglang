"""A controller that manages a group of tensor parallel workers."""
import asyncio
import logging
import multiprocessing as mp
import os
import queue
import psutil


from sglang.global_config import global_config
from sglang.srt.managers.controller.tp_worker import ModelTpClient
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.utils import get_exception_traceback
from sglang.srt.utils import flush_queue


class ControllerSingle:
    def __init__(self, model_client: ModelTpClient, router_chan: mp.Queue, detokenzier_chan: mp.Queue,
                 idle_chan: mp.Queue):
        # Init status
        self.model_client = model_client
        self.router_chan = router_chan
        self.detokenizer_chan = detokenzier_chan
        self.idle_chan = idle_chan

        # Init some configs
        self.request_dependency_delay = global_config.request_dependency_delay

    async def loop_for_forward(self):
        idle = True
        while True:
            if not idle:
                # print("forward check IDLE.....")
                if not self.idle_chan.empty():
                    # print("forward GOT IDLE signal")
                    flush_queue(self.idle_chan)
                    idle = True

            next_step_input = []

            if idle:
                # print("forward IDLE WAIT")
                next_step_input.append(self.router_chan.get())
                idle = False
                # print("forward IDLE WAIT complete")
            else:
                pass
                # print("CPU SPIN LOOP")

            # if not idle, model is doing work, disable wait and set to 0ms
            wait_timeout = 0.010 if len(next_step_input) == 1 else 0.0
            while True:
                try:
                    if wait_timeout == 0.0:
                        next_step_input.append(self.router_chan.get(block=False))
                    else:
                        next_step_input.append(self.router_chan.get(block=True, timeout=wait_timeout))
                        wait_timeout = max(0.0, wait_timeout - 0.02)
                except queue.Empty:
                    break

            # print(f"model_client.step wait...")
            if len(next_step_input) > 0:
                print(f"manager_single Forward Requests batch size: {len(next_step_input)}")

            output = await self.model_client.step(next_step_input)

            for item in output:
                self.detokenizer_chan.put_nowait(item)


def start_controller_process(
        server_args: ServerArgs,
        port_args: PortArgs,
        router_chan: mp.Queue,
        detokenizer_chan: mp.Queue,
        idle_chan: mp.Queue,
        startup_chan: mp.Queue,
        model_overide_args,
):
    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    try:
        model_client = ModelTpClient(
            list(range(server_args.tp_size)),
            server_args,
            port_args.model_port_args[0],
            model_overide_args,
        )
        controller = ControllerSingle(model_client, router_chan, detokenizer_chan, idle_chan)
    except Exception:
        startup_chan.put_nowait(get_exception_traceback())
        raise

    startup_chan.put_nowait("init ok")

    # blocking
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(controller.loop_for_forward())
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, terminating sglang process")
        try:
            cur_process = psutil.Process(os.getpid())
            parent = cur_process.parent()
        except psutil.NoSuchProcess:
            return
        children = cur_process.children(recursive=True)
        for child in children:
            child.kill()
        psutil.wait_procs(children, timeout=5)
        parent.kill()
        parent.wait(timeout=5)
