from math import sqrt
from typing import Optional, Sequence, override

from torch.nn import Module

from modugant.layers.protocol import Layer
from modugant.matrix.dim import Dim
from modugant.matrix.index import Index
from modugant.matrix.matrix import Matrix
from modugant.matrix.ops import normal, zeros


class LinearLayer[I: int, D: int, O: int](Module, Layer[I, O]):
    '''Abstract linear layer for a neural network.'''

    def __init__(
        self,
        dim: O,
        index: Index[D, I],
        follow: Optional[Sequence[Layer[O, O]]] = None,
        bias: bool = True
    ) -> None:
        '''
        Initialize the linear layer.

        Args:
            dim (int): The number of output nodes.
            index (Index[int, I]): The index.
            follow (Sequence[Layer[O, O]]): The following layers.
            bias (bool): Whether to include a bias term.

        '''
        super().__init__()
        self._dim = dim
        self._index = index
        self._weight = normal(
            mean = 0.0,
            std = 2 / (sqrt(index.dim)),
            shape = (index.dim, dim),
            requires_grad = True
        )
        self._bias = zeros((Dim.one(), dim), requires_grad = bias)
        self._follow = follow or []
    def _prepare[N: int](self, output: Matrix[N, O]) -> Matrix[N, O]:
        '''Finish the forward pass of the linear transformation.'''
        return output
    def _finish[N: int](self, output: Matrix[N, O]) -> Matrix[N, O]:
        '''Finish the forward pass of the linear transformation.'''
        return output
    @override
    def forward[N: int](self, input: Matrix[N, I]) -> Matrix[N, O]:
        '''Forward pass of the linear transformation.'''
        transform = self._prepare(input[..., self._index] @ self.weight + self._bias)
        for layer in self._follow:
            transform = layer.forward(transform)
        return self._finish(transform)
    @property
    def weight(self) -> Matrix[D, O]:
        '''Return the weight matrix.'''
        return self._weight
