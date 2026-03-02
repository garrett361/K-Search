from typing import Any, TypeAlias, TypedDict

import torch

input_t: TypeAlias = tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, dict[str, Any]]
output_t: TypeAlias = torch.Tensor


class TestSpec(TypedDict):
    B: int
    T: int
    D: int
    W: int
    seed: int
    activation: str
    withbias: bool
    withresidual: bool
