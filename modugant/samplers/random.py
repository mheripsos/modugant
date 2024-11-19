from typing import override

from torch import Tensor

from modugant.matrix import Matrix
from modugant.matrix.index import Index
from modugant.protocols import Sampler


class RandomSampler[D: int](Sampler[D]):
    '''Random sampler for GANs.'''

    def __init__(
        self,
        dim: D,
        data: Tensor,
        split: float = 0.8,
    ) -> None:
        '''
        Initialize the random sampler.

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
    @override
    def sample[N: int](self, batch: N) -> Matrix[N, D]:
        '''Generate the index for the batch.'''
        index = Index.sample(batch, self._train.shape[0], replacement = True)
        return self._train[index, ...]
    @override
    def restart(self) -> None:
        '''Restart the sampler.'''
        pass
    @property
    @override
    def holdout(self) -> Matrix[int, D]:
        '''The holdout test data.'''
        return self._test
