from typing import TypeAlias, TypedDict

import torch

input_t: TypeAlias = tuple[torch.Tensor, torch.Tensor, dict[str, str]]
output_t: TypeAlias = torch.Tensor


class TestSpec(TypedDict):
    B: int
    T: int
    D: int
    W: int
    seed: int
