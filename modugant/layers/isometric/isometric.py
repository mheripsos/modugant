from typing import Any, overload, override

from torch import Tensor
from torch.nn import Module

from modugant.layers.protocol import Layer
from modugant.matrix.matrix import Matrix


class IsometricLayer[D: int](Module, Layer[D, D]):
    '''Isometric layer.'''

    _dim: D
    @overload
    def forward[N: int](self, input: Matrix[N, D]) -> Matrix[N, D]:
        ...
    @overload
    def forward(self, input: Tensor) -> Matrix[Any, D]:
        ...
    @override
    def forward[N: int](self, input: Matrix[N, D] | Tensor) -> Matrix[Any, D]:
        output = super().forward(input)
        return Matrix.load(output, (input.shape[0], self._dim))
