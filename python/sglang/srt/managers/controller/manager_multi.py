"""
A controller that manages multiple data parallel workers.
Each data parallel worker can manage multiple tensor parallel workers.
"""

import dataclasses
import asyncio
import logging
import queue
from enum import Enum, auto
from typing import Dict
import multiprocessing
import os
import psutil

import numpy as np

from sglang.srt.managers.controller.manager_single import (
    start_controller_process as start_controller_process_single,
)
from sglang.srt.managers.io_struct import (
    AbortReq,
    FlushCacheReq,
    TokenizedGenerateReqInput,
)
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import kill_parent_process
from sglang.utils import get_exception_traceback

from sglang.srt.utils import flush_queue

logger = logging.getLogger("srt.controller")


class LoadBalanceMethod(Enum):
    """Load balance method."""

    ROUND_ROBIN = auto()
    SHORTEST_QUEUE = auto()

    @classmethod
    def from_str(cls, method: str):
        method = method.upper()
        try:
            return cls[method]
        except KeyError as exc:
            raise ValueError(f"Invalid load balance method: {method}") from exc


@dataclasses.dataclass
class WorkerHandle:
    """Store the handle of a data parallel worker."""

    proc: multiprocessing.Process
    queue: multiprocessing.Queue


class ControllerMulti:
    """A controller that manages multiple data parallel workers."""

    def __init__(
            self,
            load_balance_method: str,
            server_args: ServerArgs,
            port_args: PortArgs,
            model_overide_args,
            router_chan: multiprocessing.Queue,
            detokenzier_chan: multiprocessing.Queue,
            idle_chan: multiprocessing.Queue
    ):
        self.load_balance_method = LoadBalanceMethod.from_str(load_balance_method)
        self.server_args = server_args

        # self.port_args = port_args
        self.router_chan = router_chan
        self.detokenizer_chan = detokenzier_chan
        self.idle_chan = idle_chan

        if self.load_balance_method == LoadBalanceMethod.ROUND_ROBIN:
            self.round_robin_counter = 0

        self.dispatch_lookup = {
            LoadBalanceMethod.ROUND_ROBIN: self.round_robin_scheduler,
            LoadBalanceMethod.SHORTEST_QUEUE: self.shortest_queue_scheduler,
        }
        self.dispatching = self.dispatch_lookup[self.load_balance_method]

        # Init status
        self.recv_reqs = []

        # Start data parallel workers
        self.workers = []

        for i in range(server_args.dp_size):
            self.start_dp_worker(i)

    def start_dp_worker(self, dp_worker_id: int):
        tp_size = self.server_args.tp_size

        pipe_controller_reader, pipe_controller_writer = multiprocessing.Pipe(
            duplex=False
        )

        gpu_ids = list(range(dp_worker_id * tp_size, (dp_worker_id + 1) * tp_size))
        queue = multiprocessing.Queue()
        proc = multiprocessing.Process(
            target=start_controller_process_single,
            args=(
                self.server_args,
                self.port_args,
                pipe_controller_writer,
                self.model_overide_args,
                True,
                gpu_ids,
                dp_worker_id,
                queue,
            ),
        )
        proc.start()

        controller_init_state = pipe_controller_reader.recv()
        if controller_init_state != "init ok":
            raise RuntimeError(
                f"Initialization failed. controller_init_state: {controller_init_state}"
            )
        self.workers.append(
            WorkerHandle(
                proc=proc,
                queue=queue,
            )
        )

    def round_robin_scheduler(self, input_requests):
        for r in input_requests:
            self.workers[self.round_robin_counter].queue.put(r)
            self.round_robin_counter = (self.round_robin_counter + 1) % len(
                self.workers
            )

    def shortest_queue_scheduler(self, input_requests):
        for r in input_requests:
            queue_sizes = [worker.queue.qsize() for worker in self.workers]
            wid = np.argmin(queue_sizes)
            self.workers[wid].queue.put(r)


    async def remove_dead_workers(self):
        for i in list(self.workers.keys()):
            worker_thread = self.workers[i]
            if not worker_thread.liveness:
                worker_thread.join()
                # move unsuccessful requests back to the queue
                while not worker_thread.request_queue.empty():
                    self.recv_reqs.append(worker_thread.request_queue.get())
                del self.workers[i]
                logger.info(f"Stale worker {i} removed")

    # async def loop_for_forward(self):
    #     while True:
    #         print("multi loop_for_forward 1")
    #         await self.remove_dead_workers()
    #         print("multi loop_for_forward 2")
    #         if self.have_any_live_worker():
    #             next_step_input = list(self.recv_reqs)
    #             print("multi loop_for_forward 3",next_step_input)
    #             self.recv_reqs = []
    #             if next_step_input:
    #                 await self.dispatching(next_step_input)
    #         # else:
    #         #    logger.error("There is no live worker.")
    #
    #         await asyncio.sleep(global_config.wait_for_new_request_delay)

    async def loop_for_forward(self):
        idle = True
        while True:
            if not idle:
                # print("forward check IDLE.....")
                if not self.idle_chan.empty():
                    # print("forward GOT IDLE signal")
                    flush_queue(self.idle_chan)
                    idle = True

            recv_req = None
            if idle:
                # print("forward IDLE WAIT")
                recv_req = self.router_chan.get()
                idle = False
                # print("forward IDLE WAIT complete", recv_req)
            else:
                pass
                # print("CPU SPIN LOOP")

            self.put_to_request_queue(recv_req)

            wait_timeout = 0.010 if recv_req is not None else 0.0
            while True:
                try:
                    if wait_timeout == 0.0:
                        self.put_to_request_queue(self.router_chan.get(block=False))
                    else:
                        self.put_to_request_queue(self.router_chan.get(block=True, timeout=wait_timeout))
                        wait_timeout = max(0.0, wait_timeout - 0.02)
                except queue.Empty:
                    break

            await self.remove_dead_workers()

            next_step_input = list(self.recv_reqs)
            self.recv_reqs = []
            self.dispatching(next_step_input)

    def put_to_request_queue(self, recv_req):
        if isinstance(recv_req, FlushCacheReq):
            # TODO(lsyin): apply more specific flushCacheReq
            for worker in self.workers:
                worker.queue.put(recv_req)
        elif isinstance(recv_req, AbortReq):
            in_queue = False
            for i, req in enumerate(self.recv_reqs):
                if req.rid == recv_req.rid:
                    self.recv_reqs[i] = recv_req
                    in_queue = True
                    break
            if not in_queue:
                # Send abort req to all TP groups
                for worker in self.workers:
                    worker.queue.put(recv_req)
        elif isinstance(recv_req, TokenizedGenerateReqInput):
            self.recv_reqs.append(recv_req)
        # else:
        #     logger.error(f"Invalid object: {recv_req}")


def start_controller_process(
        server_args: ServerArgs,
        port_args: PortArgs,
        router_chan: multiprocessing.Queue,
        detokenizer_chan: multiprocessing.Queue,
        idle_chan: multiprocessing.Queue,
        startup_chan: multiprocessing.Queue,
        model_overide_args,
):
    """Start a controller process."""

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    try:
        controller = ControllerMulti(
            server_args.load_balance_method, server_args, port_args, model_overide_args, router_chan, detokenizer_chan,
            idle_chan
        )
    except Exception:
        startup_chan.put_nowait(get_exception_traceback())
        raise

    startup_chan.put_nowait("init ok")

    # blocking
    try:
        loop = asyncio.get_event_loop()
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
    except Exception:
        logger.error("Exception in ControllerMulti:\n" + get_exception_traceback())
    finally:
        for w in controller.workers:
            os.kill(w.proc.pid, 9)
        kill_parent_process()
