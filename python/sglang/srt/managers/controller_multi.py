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

"""
A controller that manages multiple data parallel workers.
Each data parallel worker can manage multiple tensor parallel workers.
"""

import dataclasses
import logging
import queue
import multiprocessing
import os
import psutil
from enum import Enum, auto

import numpy as np

from sglang.srt.managers.controller_single import (
    start_controller_process as start_controller_process_single,
)
from sglang.srt.managers.io_struct import (
    AbortReq,
    FlushCacheReq,
    TokenizedGenerateReqInput,
)
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import configure_logger, kill_parent_process
from sglang.utils import get_exception_traceback

from sglang.srt.utils import flush_queue

logger = logging.getLogger(__name__)



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
            server_args: ServerArgs,
            port_args: PortArgs,
            router_chan: multiprocessing.Queue,
            detokenzier_chan: multiprocessing.Queue,
            idle_chan: multiprocessing.Queue
    ):
        # Parse args
        self.server_args = server_args
        self.port_args = port_args
        self.load_balance_method = LoadBalanceMethod.from_str(
            server_args.load_balance_method
        )

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

        startup_chan = multiprocessing.Queue()

        gpu_ids = list(range(dp_worker_id * tp_size, (dp_worker_id + 1) * tp_size))
        queue = multiprocessing.Queue()
        proc = multiprocessing.Process(
            target=start_controller_process_single,
            args=(
                self.server_args,
                self.port_args,
                self.model_overide_args,
                self.router_chan,
                self.detokenizer_chan,
                self.idle_chan,
                startup_chan,
                True,
                gpu_ids,
                dp_worker_id,
                queue,
            ),
        )
        proc.start()

        controller_init_state = startup_chan.get()
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

    def loop_for_forward(self):
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
):
    """Start a controller process."""

    configure_logger(server_args)

    try:
        controller = ControllerMulti(server_args, port_args, router_chan, detokenizer_chan, idle_chan)
    except Exception:
        startup_chan.put_nowait(get_exception_traceback())
        raise

    startup_chan.put_nowait("init ok")

    # blocking
    try:
        controller.loop_for_forward()
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
        kill_parent_process()
