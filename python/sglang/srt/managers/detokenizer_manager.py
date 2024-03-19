import multiprocessing
import queue
import threading

from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.srt.managers.io_struct import BatchStrOut, BatchTokenIDOut
from sglang.srt.server_args import ServerArgs


class DetokenizerManager:
    def __init__(
            self,
            server_args: ServerArgs,
            detokenizer_chan: multiprocessing.Queue,
            output_chan: queue.Queue,
    ):
        self.detokenizer_chan = detokenizer_chan
        self.output_chan = output_chan

        self.tokenizer = get_tokenizer(
            server_args.tokenizer_path,
            tokenizer_mode=server_args.tokenizer_mode,
            trust_remote_code=server_args.trust_remote_code,
        )

        self.work_thread = None

    def start(self):
        if self.work_thread is None:
            self.work_thread = threading.Thread(target=self.work_loop, daemon=True)
            self.work_thread.start()

    def work_loop(self):
        # detokenizer_get = make_async_thread(self.detokenizer_chan.get)

        while True:
            # print(f"detokenizer detokenizer_chan get wait...")
            recv_obj = self.detokenizer_chan.get()
            # print(f"detokenizer detokenizer_chan get done: {recv_obj}")

            if isinstance(recv_obj, BatchTokenIDOut):
                output_tokens = recv_obj.output_tokens

                # TODO(lmzheng): handle skip_special_tokens per request
                output_strs = self.tokenizer.batch_decode(
                    output_tokens,
                    skip_special_tokens=recv_obj.skip_special_tokens[0],
                )

                # Trim stop str
                # TODO(lmzheng): handle the case where multiple stop strs are hit
                for i in range(len(output_strs)):
                    if recv_obj.hit_stop_str[i] is not None:
                        pos = output_strs[i].find(recv_obj.hit_stop_str[i])
                        if pos != -1:
                            output_strs[i] = output_strs[i][:pos]

                    if len(output_tokens[i]) > 0:
                        first_token = self.tokenizer.convert_ids_to_tokens(
                            int(output_tokens[i][0])
                        )
                        if not isinstance(first_token, str):
                            first_token = first_token.decode("utf-8", errors="ignore")
                        if first_token.startswith("▁"):
                            output_strs[i] = " " + output_strs[i]

                    output_strs[i] = (
                            recv_obj.output_and_jump_forward_strs[i] + output_strs[i]
                    )

                # print(f"detokenizer tokenizer_chan put")
                self.output_chan.put_nowait(
                    BatchStrOut(
                        recv_obj.rids,
                        output_strs,
                        recv_obj.meta_info,
                        recv_obj.finished,
                    )
                )
                # print(f"detokenizer tokenizer_chan put done")
            else:
                raise ValueError(f"Invalid object: {recv_obj}")
