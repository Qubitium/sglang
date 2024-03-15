import asyncio
import logging

import uvloop
import zmq
import zmq.asyncio
from sglang.srt.managers.router.model_rpc import ModelRpcClient
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import get_exception_traceback

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


class RouterManager:
    def __init__(self, model_client: ModelRpcClient, port_args: PortArgs, next_request_delay: float = 0.03):
        # Init communication
        context = zmq.asyncio.Context(2)
        self.recv_from_tokenizer = context.socket(zmq.PULL)
        self.recv_from_tokenizer.bind(f"tcp://127.0.0.1:{port_args.router_port}")

        self.send_to_detokenizer = context.socket(zmq.PUSH)
        self.send_to_detokenizer.connect(
            f"tcp://127.0.0.1:{port_args.detokenizer_port}"
        )

        # sleep time in ms to increase cache hit-rate, default=30ms
        # tip: reduce delay for system with low cpu <-> gpu latency
        self.next_request_delay = next_request_delay

        # Init status
        self.model_client = model_client
        self.recv_reqs = []

    async def loop_for_forward(self):
        # req_pending = 0
        while True:
            #print(f"Router processing next requests: {len(self.recv_reqs)}")
            next_step_input = list(self.recv_reqs)
            #if len(next_step_input) > 0:
                #req_pending += len(next_step_input)
                #print(f"Router Request got more, Pending: {req_pending}")

            self.recv_reqs = []
            #print(f"Router waiting for out_pyobs...")
            out_pyobjs = await self.model_client.step(next_step_input)
            #print(f"Router got out_pyobs...")

            # async sleep before processing next request to increase cache hit-rate
            if len(out_pyobjs) != 0:
                for obj in out_pyobjs:
                    self.send_to_detokenizer.send_pyobj(obj)

                has_finished = False
                for obj in out_pyobjs:
                    if obj.finished:
                        has_finished = True
                        #req_pending -= 1
                        #print(f"Router Request finished, Pending: {req_pending}")

                if has_finished:
                    #print(f"Router going to sleep for: {self.next_request_delay}, Pending: {req_pending}")
                    await asyncio.sleep(self.next_request_delay)
                else:
                    pass #print(f"Router not sleeping, Pending: {req_pending}")
            else:
                await asyncio.sleep(0.005)
            # if req_pending == 0:
            #     #print(f"Router sleep due to req_pending == 0")
            #     await asyncio.sleep(0.005)
            # else:
            #     # TODO why this delay?
            #     pass # await asyncio.sleep(0.0005)

            #assert req_pending >= 0

    async def loop_for_recv_requests(self):
        while True:
            recv_req = await self.recv_from_tokenizer.recv_pyobj()
            self.recv_reqs.append(recv_req)


def start_router_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    pipe_writer,
    **kwargs,
):
    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )

    try:
        model_client = ModelRpcClient(server_args, port_args)
        router = RouterManager(model_client, port_args, **kwargs)
    except Exception:
        pipe_writer.send(get_exception_traceback())
        raise

    pipe_writer.send("init ok")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(router.loop_for_recv_requests())
    loop.run_until_complete(router.loop_for_forward())
