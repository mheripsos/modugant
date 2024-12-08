from typing import override

from torch.nn import Module

from modugant.layers.protocol import Layer
from modugant.matrix.index import Index
from modugant.matrix.matrix import Matrix
from modugant.matrix.ops import cat


class ResidualLayer[I: int, S: int, L: int, O: int](Module, Layer[I, O]):
    '''Residual layer for a generator.'''

    def __init__(
        self,
        dim: O,
        index: Index[S, I],
        layer: Layer[S, L]
    ) -> None:
        '''
        Initialize the residual layer.

        Args:
            dim (int): The number of output nodes.
            index (Index[int, I]): The index.
            layer (Layer[S, O]): The following layer.

        '''
        assert index.dim + layer.dim == dim, 'The dimensions do not match.'
        super().__init__()
        self._dim = dim
        self._index = index
        self._layer = layer
    @override
    def forward[N: int](self, input: Matrix[N, I]) -> Matrix[N, O]:
        subset = input[..., self._index]
        return cat(
            (subset, self._layer.forward(subset)),
            dim = 1,
            shape = (input.shape[0], self._dim)
        )
