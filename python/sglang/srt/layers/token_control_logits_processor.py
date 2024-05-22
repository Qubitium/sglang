import math
import os
from typing import Optional, List, Tuple

import torch
import trainx

# The test found that it is greater than or equal to 11.97 to make the white tokens more likely to appear.
WHITE_TOKEN_BIAS_VALUE = 11.97


class TokenControlLogitsProcessor:
    def __init__(self):
        self.bias: Optional[torch.Tensor] = None

        # TODO FIXME: Temporarily use fixed command-r token.
        #   Need to fix the issue that different processes cannot obtain token_id
        # print("TokenControlLogitsProcessor init process", os.getpid())
        # self.key_start_token = trainx.TOKEN_KEY()
        # self.key_end_token = trainx.TOKEN_END()
        # self.eos_token = trainx.TOKEN_EOS()

        self.key_start_token = 255019
        self.value_token = 255020
        self.key_end_token = 255021
        self.eos_token = 255001
        self.bos_token = 5

    def __call__(self, last_ids: List[List[int]], scores: torch.Tensor) -> torch.Tensor:
        self.create_bias(scores)

        changed_tokens = []
        for i, last_id in enumerate(last_ids):
            white_tokens, black_tokens = self.compute_token(last_id)
            if white_tokens:
                self.bias[i][white_tokens] = WHITE_TOKEN_BIAS_VALUE
            elif black_tokens:
                self.bias[i][black_tokens] = -math.inf

            changed_tokens.append(white_tokens if white_tokens else black_tokens)

        scores.add_(self.bias)

        self.reset_bias(changed_tokens)
        return scores

    def compute_token(self, last_id) -> Tuple[List[int], List[int]]:
        white_tokens = None
        black_tokens = None
        if last_id == self.key_end_token:
            white_tokens = [self.key_start_token, self.key_end_token, self.eos_token]
        elif last_id == self.key_start_token:
            black_tokens = [self.key_start_token, self.key_end_token, self.value_token, self.eos_token]
        elif last_id == self.value_token:
            black_tokens = [self.key_end_token, self.value_token, self.eos_token]
        return white_tokens, black_tokens

    def create_bias(self, scores):
        if self.bias is None:
            # We create it here because full_like() also copies the device and dtype
            self.bias = torch.full_like(scores, 0)

    def reset_bias(self, changed_tokens: List[List[int]]):
        tokens = changed_tokens
        # reset last_block_tokens setting
        if self.bias is not None and len(tokens) > 0:
            for b in self.bias:
                b[tokens] = 0
