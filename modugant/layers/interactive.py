from typing import Sequence, override

from torch.nn import Module

from modugant.layers.linear.sphere import SphericalLayer
from modugant.layers.protocol import Layer
from modugant.matrix.index import Index
from modugant.matrix.matrix import Matrix
from modugant.matrix.ops import products


class InteractionLayer[I: int, O: int](Module, Layer[I, O]):
    '''Linear layer with interactive weights.'''

    def __init__(
        self,
        left: Sequence[Index[int, I]],
        right: Index[O, I]
    ) -> None:
        '''
        Initialize the interaction layer.

        Args:
            left (Sequence[Index[int, I]]): The left indices.
            right (Index[R, I]): The right index.

        '''
        super().__init__()
        self._left = left
        self._right = right
        self._layers = [SphericalLayer(right.dim, index) for index in left]
    @override
    def forward[N: int](self, input: Matrix[N, I]) -> Matrix[N, O]:
        return input[..., self._right] + products(tuple(layer.forward(input) for layer in self._layers))
