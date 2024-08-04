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

"""A controller that manages a group of tensor parallel workers."""

import logging
import multiprocessing
import os
import queue
from typing import List

from sglang.srt.managers.tp_worker import (
    ModelTpServer,
    broadcast_recv_input,
    launch_tp_servers,
)
from sglang.srt.managers.io_struct import AbortReq
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import flush_queue, kill_parent_process
from sglang.utils import get_exception_traceback

logger = logging.getLogger(__name__)

class ControllerSingle:
    """A controller that manages a group of tensor parallel workers."""

    def __init__(self,
                 server_args: ServerArgs,
                 port_args: PortArgs,
                 model_overide_args: dict,
                 gpu_ids: List[int],
                 is_data_parallel_worker: bool,
                 dp_worker_id: int,
                 mp_queue: multiprocessing.Queue,
                 router_chan: multiprocessing.Queue,
                 detokenzier_chan: multiprocessing.Queue,
                 idle_chan: multiprocessing.Queue,
                 ):
        # Init status
        self.router_chan = router_chan
        self.detokenizer_chan = detokenzier_chan
        self.idle_chan = idle_chan

        # Parse args
        self.tp_size = server_args.tp_size
        self.is_dp_worker = is_data_parallel_worker
        self.dp_worker_id = dp_worker_id
        self.mp_queue = mp_queue

        # Launch other tp ranks
        tp_size_local = server_args.tp_size // server_args.nnodes
        self.tp_procs = []
        if tp_size_local > 1:
            tp_rank_range = range(1, tp_size_local)
            self.tp_procs = launch_tp_servers(
                gpu_ids,
                tp_rank_range,
                server_args,
                port_args.nccl_ports[dp_worker_id],
                model_overide_args,
            )

        # Launch tp rank 0
        self.tp_server = ModelTpServer(
            gpu_ids[0],
            0,
            server_args,
            port_args.nccl_ports[dp_worker_id],
            model_overide_args,
        )
        self.tp_cpu_group = self.tp_server.model_runner.tp_group.cpu_group

    def loop_for_forward(self):
        idle = True
        while True:
            if not self.is_dp_worker:
                recv_reqs, idle = self.recv_requests(idle)
            else:
                recv_reqs = self.recv_requests_from_mp_queue()

            if self.tp_size > 1:
                broadcast_recv_input(recv_reqs, 0, self.tp_cpu_group)

            out_pyobjs = self.tp_server.exposed_step(recv_reqs)

            for obj in out_pyobjs:
                self.detokenizer_chan.put_nowait(obj)

    def recv_requests(self, idle: bool):
        recv_reqs = []
        if not self.is_dp_worker:
            if not idle:
                # print("forward check IDLE.....")
                if not self.idle_chan.empty():
                    # print("forward GOT IDLE signal")
                    flush_queue(self.idle_chan)
                    idle = True

            if idle:
                # print("forward IDLE WAIT")
                recv_obj = self.router_chan.get()
                recv_reqs.append(recv_obj)

                if isinstance(recv_obj, AbortReq):
                    pass
                    # print("Received AbortReq, forward IDLE WAIT")
                else:
                    idle = False
                    # print("forward IDLE WAIT complete")
            else:
                pass
                # print("CPU SPIN LOOP")

            # if not idle, model is doing work, disable wait and set to 0ms
            wait_timeout = 0.010 if len(recv_reqs) == 1 else 0.0
            while True:
                try:
                    if wait_timeout == 0.0:
                        recv_reqs.append(self.router_chan.get(block=False))
                    else:
                        recv_reqs.append(self.router_chan.get(block=True, timeout=wait_timeout))
                        wait_timeout = max(0.0, wait_timeout - 0.02)
                except queue.Empty:
                    break

            # print(f"model_client.step wait...")
            if len(recv_reqs) > 0:
                print(f"manager_single Forward Requests batch size: {len(recv_reqs)}")

        return recv_reqs, idle

    def recv_requests_from_mp_queue(self):
        recv_reqs = []
        while not self.mp_queue.empty():
            recv_reqs.append(self.mp_queue.get())
        return recv_reqs


def start_controller_process(
        server_args: ServerArgs,
        port_args: PortArgs,
        model_overide_args: dict,
        router_chan: multiprocessing.Queue,
        detokenizer_chan: multiprocessing.Queue,
        idle_chan: multiprocessing.Queue,
        startup_chan: multiprocessing.Queue,
        is_data_parallel_worker: bool = False,
        gpu_ids: List[int] = None,
        dp_worker_id: int = None,
        queue: multiprocessing.connection.Connection = None,
):
    """Start a controller process."""

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    if not is_data_parallel_worker:
        tp_size_local = server_args.tp_size // server_args.nnodes
        gpu_ids = [i for _ in range(server_args.nnodes) for i in range(tp_size_local)]
        dp_worker_id = 0
        queue = None

    try:
        controller = ControllerSingle(
            server_args,
            port_args,
            model_overide_args,
            gpu_ids,
            is_data_parallel_worker,
            dp_worker_id,
            queue,
            router_chan, detokenizer_chan, idle_chan,
        )
    except Exception:
        startup_chan.put_nowait(get_exception_traceback())
        raise

    startup_chan.put("init ok")

    try:
        controller.loop_for_forward()
    except Exception:
        logger.error("Exception in ControllerSingle:\n" + get_exception_traceback())
    finally:
        for t in controller.tp_procs:
            os.kill(t.pid, 9)
        kill_parent_process()
