from abc import ABC, abstractmethod
from typing import overload

from torch import Module, Tensor

from modugant.matrix.matrix import Matrix


class Layer[I: int, O: int](Module, ABC):
    '''Abstract layer for a neural network.'''

    @overload
    def forward[N: int](self, input: Matrix[N, I]) -> Matrix[N, O]:
        ...
    @overload
    def forward[N: int](self, input: Tensor) -> Tensor:
        ...
    @abstractmethod
    def forward[N: int](self, input: Matrix[N, I] | Tensor) -> Matrix[N, O] | Tensor:
        '''Forward pass through the layer.'''
        ...
