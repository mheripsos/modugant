from typing import override

from torch import Tensor

from modugant.matrix import Matrix
from modugant.matrix.index import Index
from modugant.protocols import Sampler


class IteratingSampler[S: int](Sampler[S]):
    '''Iterating sampler for GANs.'''

    def __init__(
        self,
        dim: S,
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
        sample = Index.randperm(n)
        cuttoff = int(n * split)
        self._train = Matrix(data[sample[:cuttoff], :], (cuttoff, dim))
        self._test = Matrix(data[sample[cuttoff:], :], (n - cuttoff, dim))
        self.__cursor = 0
    @override
    def sample[N: int](self, batch: N) -> Matrix[N, S]:
        index = Index.slice(self.__cursor, batch, self.__cursor + batch)
        wrapped = index.wrap(self._train.shape[0])
        self.__cursor += batch
        return self._train[wrapped, ...]
    @override
    def restart(self) -> None:
        self.__cursor = 0
    @property
    @override
    def holdout(self) -> Matrix[int, S]:
        return self._test
