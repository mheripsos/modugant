from typing import Any, override

from torch import Tensor

from modugant.matrix import Matrix
from modugant.matrix.index import Index
from modugant.protocols import Loader


class StandardizeLoader[S: int, D: int](Loader[S, D]):
    '''Normalize encoder for GANs.'''

    def __init__(
        self,
        data: Tensor,
        index: Index[D, S]
    ) -> None:
        '''
        Initialize the normalize encoder.

        Args:
            sampled (int): The number of sampled dimensions.
            data (Tensor): The data to sample.
            index (Index): The index to load.

        '''
        self._sampled = index.cap
        self._outputs = index.dim
        self._index = index
        subset = Matrix(data, (data.shape[0], self._index.cap))
        self._mean = subset[..., self._index].mean(dim = 0, keepdim = True)
        self._std = subset[..., self._index].std(dim = 0, keepdim = True)
    @override
    def load[N: int](self, data: Matrix[N, S]) -> Matrix[N, D]:
        return (data[..., self._index] - self._mean) / self._std
    @override
    def unload[N: int](self, data: Matrix[N, D]) -> Matrix[N, Any]:
        return (data * self._std) + self._mean
