from typing import Protocol

from modugant.matrix.matrix import Matrix


class Layer[I: int, O: int](Protocol):
    '''Abstract layer for a neural network.'''

    _dim: O
    def forward[N: int](self, input: Matrix[N, I]) -> Matrix[N, O]:
        '''Forward pass through the layer.'''
        ...
    @property
    def dim(self) -> O:
        '''Return the number of output nodes.'''
        return self._dim

