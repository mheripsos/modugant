from typing import override

from torch import Tensor

from modugant.matrix import Matrix
from modugant.matrix.index import Index
from modugant.matrix.ops import randperm
from modugant.protocols import Sampler


class IteratingSampler[D: int](Sampler[D]):
    '''Iterating sampler for GANs.'''

    def __init__(
        self,
        dim: D,
        data: Tensor,
        split: float = 0.8
    ) -> None:
        '''
        Initialize the iterating sampler.

        Args:
            dim (int): The number of inputs and outputs.
            data (torch.Tensor): The data to sample.
            split (float): The split between the training and test data.

        '''
        assert data.shape[1] == dim, f'Data {data} is not of dimension {dim}'
        self._outputs = dim
        n = data.shape[0]
        sample = randperm(n)
        cuttoff = int(n * split)
        self._train = Matrix.load(data[sample[:cuttoff], :], (cuttoff, dim))
        self._test = Matrix.load(data[sample[cuttoff:], :], (n - cuttoff, dim))
        self.__cursor = 0
    @override
    def sample[N: int](self, batch: N) -> Matrix[N, D]:
        index = Index.wrap(Index.slice(self.__cursor, batch), len(self._train))
        self.__cursor += batch
        return self._train[index, ...]
    @override
    def restart(self) -> None:
        self.__cursor = 0
    @property
    @override
    def holdout(self) -> Matrix[int, D]:
        return self._test
