from typing import overload, override

from torch import Tensor
from torch.nn import Linear

from modugant.layers.abstract import Layer
from modugant.matrix.matrix import Matrix


class Spherical[I: int, O: int](Linear, Layer[I, O]):
    def __init__(self, inputs: I, outputs: O) -> None:
        '''
        Initialize the spherical layer.

        Args:
            inputs (I: int): The number of input nodes.
            outputs (O: int): The number of output nodes.

        '''
        super().__init__(inputs, outputs)
    @overload
    def forward[N: int](self, input: Matrix[N, I]) -> Matrix[N, O]:
        ...
    @overload
    def forward[N: int](self, input: Tensor) -> Tensor:
        ...
    @override
    def forward[N: int](self, input: Matrix[N, I] | Tensor) -> Matrix[N, O] | Tensor:
        return super().forward(input)