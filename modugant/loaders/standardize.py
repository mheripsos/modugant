from typing import Any, List, override

from torch import Tensor

from modugant.matrix import Matrix
from modugant.matrix.index import Index
from modugant.protocols import Loader


class StandardizeLoader[D: int](Loader[D]):
    '''Normalize encoder for GANs.'''

    def __init__(
        self,
        dim: D,
        data: Tensor,
        index: List[int]
    ) -> None:
        '''
        Initialize the normalize encoder.

        Args:
            dim (int): The number of inputs and outputs.
            data (torch.Tensor): The data to sample.
            index (List[int]): The indices to normalize.

        '''
        assert len(index) == dim
        self._outputs = dim
        self._index = Index.load(index, dim)
        subset = Matrix.load(data, (data.shape[0], data.shape[1]))
        self._mean = subset[..., self._index].mean(dim = 0, keepdim = True)
        self._std = subset[..., self._index].std(dim = 0, keepdim = True)
    @override
    def load[N: int](self, data: Matrix[N, int]) -> Matrix[N, D]:
        return (data[..., self._index] - self._mean) / self._std
    @override
    def unload[N: int](self, data: Matrix[N, D]) -> Matrix[N, Any]:
        return (data * self._std) + self._mean
