from typing import override

from torch.nn import Module

from modugant.layers.protocol import Layer
from modugant.matrix.index import Index
from modugant.matrix.matrix import Matrix


class SubsetLayer[I: int, S: int, O: int](Module, Layer[I, O]):
    '''Layer that selects a subset of the input.'''

    def __init__(
        self,
        dim: O,
        index: Index[S, I],
        layer: Layer[S, O]
    ) -> None:
        '''
        Initialize the subset layer.

        Args:
            dim (int): The number of output nodes.
            index (Index[int, I]): The index.
            layer (Layer[S, O]): The following layer.

        '''
        self._dim = dim
        self._index = index
        self._layer = layer
    @override
    def forward[N: int](self, input: Matrix[N, I]) -> Matrix[N, O]:
        return self._layer.forward(input[..., self._index])
