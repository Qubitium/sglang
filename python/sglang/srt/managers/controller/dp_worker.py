"""A data parallel worker thread."""
import asyncio
import logging
import queue
import threading
from typing import List, Callable

import uvloop
# import zmq

from sglang.global_config import global_config
from sglang.srt.managers.controller.tp_worker import ModelTpClient
from sglang.srt.managers.io_struct import BatchTokenIDOut
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.utils import get_exception_traceback



logger = logging.getLogger("srt.controller")
CHECKING_INTERVAL = 5

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


class DataParallelWorkerThread(threading.Thread):
    def __init__(
        self,
        worker_id: int,
        request_queue: queue.Queue,
        detokenizer_port: int,
        step_func: Callable,
        detokenzier_chan,
    ):
        super(DataParallelWorkerThread, self).__init__()
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.liveness = True
        self.request_dependency_delay = global_config.request_dependency_delay

        self.detokenzier_chan = detokenzier_chan

        self.step = step_func

    async def loop_for_forward(self):
        while self.liveness:
            requests = []
            while not self.request_queue.empty():
                requests.append(self.request_queue.get())

            out_pyobjs: List[BatchTokenIDOut] = []
            try:
                out_pyobjs = await self.step(requests)
            except Exception:
                for r in requests:
                    self.request_queue.put(r)
                logger.error(
                    f"Worker thread {self.worker_id}: "
                    f"failed to get back from Model Server\n"
                    f"{get_exception_traceback()}"
                )
                self.liveness = False

            # TODO remove code after testing in DP
            if len(out_pyobjs) > 0:
                print(f"out_pyobjs: type: {type(out_pyobjs[0])}")

            for obj in out_pyobjs:
                self.detokenzier_chan.put_nowait(obj)

            # async sleep for receiving the subsequent request and avoiding cache miss
            if len(out_pyobjs) != 0:
                # TODO remove comment after testing in DP
                has_finished = any([obj.finished_reason is not None for obj in out_pyobjs])
                if has_finished:
                    await asyncio.sleep(self.request_dependency_delay)
            await asyncio.sleep(global_config.wait_for_new_request_delay)

    async def monitoring(self):
        while True:
            await asyncio.sleep(CHECKING_INTERVAL)
            # can plug in monitoring logic here

    def run(self):
        logger.info(f"DataParallelWorkerThread {self.worker_id} start")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.create_task(self.monitoring())
        loop.run_until_complete(self.loop_for_forward())


def start_data_parallel_worker(
    server_args: ServerArgs,
    port_args: PortArgs,
    model_overide_args,
    gpu_ids: List[int],
    worker_id: int,
    detokenzier_chan,
):
    model_tp_client = ModelTpClient(
        gpu_ids,
        server_args,
        port_args.model_port_args[worker_id],
        model_overide_args,
    )
    worker_thread = DataParallelWorkerThread(
        worker_id=worker_id,
        request_queue=queue.Queue(),
        detokenizer_port=port_args.detokenizer_port,
        step_func=model_tp_client.step,
        detokenzier_chan=detokenzier_chan,
    )
    worker_thread.start()
    return worker_thread