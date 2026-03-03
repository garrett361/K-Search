from task import input_t, output_t
from utils import make_match_reference

import torch


def ref_kernel(data: input_t) -> output_t:
    from fla.modules.convolution import causal_conv1d_fwd

    x, weight, bias, residual, config = data
    y, _ = causal_conv1d_fwd(
        x=x,
        weight=weight,
        bias=bias,
        residual=residual,
        activation=config.get("activation"),
    )
    return y


def generate_input(
    B: int,
    T: int,
    D: int,
    W: int,
    seed: int,
) -> input_t:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)

    x = torch.randn(B, T, D, device="cuda", dtype=torch.bfloat16, generator=gen)
    weight = torch.randn(D, W, device="cuda", dtype=torch.bfloat16, generator=gen)

    config = {}

    return (x, weight, config)


check_implementation = make_match_reference(ref_kernel, rtol=2e-2, atol=2e-2)
