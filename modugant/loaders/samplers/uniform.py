from typing import override

from modugant.loaders.protocol import Sampler
from modugant.matrix.index import Index


class RandomSampler(Sampler):
    '''Random sampler for GANs.'''

    def __init__(
        self,
        size: int
    ) -> None:
        '''
        Initialize the random sampler.

        Args:
            size (int): The size of the data.

        '''
        self.__size = size
    @override
    def sample[N: int](self, batch: N) -> Index[N, int]:
        '''Generate the index for the batch.'''
        return Index.sample(batch, self.__size, replacement = True)
    @override
    def restart(self) -> None:
        '''Restart the sampler.'''
        pass
