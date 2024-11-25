from typing import Any, override

from modugant.loaders.connectors.transformers.protocol import Transformer
from modugant.matrix.index import Index
from modugant.matrix.matrix import Matrix


class StandardizeTransformer[S: int](Transformer[S]):
    '''Standardize Transformer for Connector composition. Standardizes the data.'''

    def __init__(
        self,
        data: Matrix[int, int],
        index: Index[S, int]
    ) -> None:
        '''
        Initialize the standardize transformer.

        Args:
            data (Matrix): The data from which to compute mean and variance.
            index (Index): The index of the data to standardize.

        '''
        self._index = index
        subset = data[..., index]
        self._mean = subset.mean(dim = 0, keepdim = True)
        self._std = subset.std(dim = 0, keepdim = True)
    @override
    def load[N: int](self, data: Matrix[N, Any]) -> Matrix[N, S]:
        '''Transform underlying data.'''
        return (data[..., self._index] - self._mean) / self._std
    @override
    def unload[N: int](self, data: Matrix[N, S]) -> Matrix[N, Any]:
        '''Revert to underlying data.'''
        return (data * self._std) + self._mean
