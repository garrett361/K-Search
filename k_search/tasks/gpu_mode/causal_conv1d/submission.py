from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
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
