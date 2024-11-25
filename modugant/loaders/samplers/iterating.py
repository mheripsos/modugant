from typing import override

from modugant.loaders.protocol import Sampler
from modugant.matrix.index import Index


class IteratingSampler(Sampler):
    '''Iterating sampler for GANs.'''

    def __init__(
        self,
        size: int,
    ) -> None:
        '''
        Initialize the iterating sampler.

        Args:
            size (int): The size of the data.

        '''
        self.__size = size
        self.__cursor = 0
    @override
    def sample[N: int](self, batch: N) -> Index[N, int]:
        return Index.slice(self.__cursor, batch, self.__cursor + batch)
    @override
    def restart(self) -> None:
        self.__cursor = 0
