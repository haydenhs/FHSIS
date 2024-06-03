import torch
from models.FHSIS import Net
from fvcore.nn.flop_count import flop_count
from typing import Any, Callable, List, Optional, Union
from collections import Counter, OrderedDict
from numbers import Number
from fvcore.nn.jit_handles import get_shape
from numpy import prod
from models.GPPNN import GPPNN


def softmax_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    This is a generic jit handle that counts the number of activations for any operation given the output shape.
    """
    out_shape = get_shape(outputs[0])
    ac_count = prod(out_shape)
    return ac_count * 4


def einsum_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for the einsum operation. We currently support
    two einsum operations: "nct,ncp->ntp" and "ntg,ncg->nct".
    """
    # Inputs of einsum should be a list of length 2.
    # Inputs[0] stores the equation used for einsum.
    # Inputs[1] stores the list of input shapes.
    assert len(inputs) == 2, len(inputs)
    equation = inputs[0].toIValue()
    # Get rid of white space in the equation string.
    equation = equation.replace(" ", "")
    # Re-map equation so that same equation with different alphabet
    # representations will look the same.
    letter_order = OrderedDict((k, 0) for k in equation if k.isalpha()).keys()
    mapping = {ord(x): 97 + i for i, x in enumerate(letter_order)}
    equation = equation.translate(mapping)
    input_shapes_jit = inputs[1].node().inputs()
    input_shapes = [get_shape(v) for v in input_shapes_jit]
    if equation == "abc,abd->acd":
        n, c, t = input_shapes[0]
        p = input_shapes[-1][-1]
        flop = n * c * t * p
        return flop

    elif equation == "abc,adc->adb":
        n, t, g = input_shapes[0]
        c = input_shapes[-1][1]
        flop = n * t * g * c
        return flop

    elif equation == "abcde,acfde->abfde":
        a, b, c, d, e = input_shapes[0]
        a, c, f, d, e = input_shapes[1]
        flop = a * b * c * d * e * f
        return flop

    else:
        raise NotImplementedError("Unsupported einsum operation.")


rgb = torch.randn(1, 3, 512, 512)
lr = torch.randn(1, 31, 16, 16)
lr_rgb = torch.randn(1, 3, 16, 16)
inputs = (lr, rgb)
#model = Net(n_basis=5)
model = GPPNN()
ops = {'aten::einsum':einsum_flop_jit, 'aten::softmax':softmax_flop_jit}
out1, out2 = flop_count(model, inputs, ops)
print("Total FLOPs in Gb is ", sum(out1.values()))
print(out2)
