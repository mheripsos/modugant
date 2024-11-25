from typing import override

from modugant.loaders.connectors.splitters.protocol import Splitter
from modugant.matrix.dim import Zero
from modugant.matrix.index import Index
from modugant.matrix.matrix import Matrix


class IdentitySplitter[S: int, C: int, D: int](Splitter[S, C, D]):
    '''
    Identity Splitter for Transformer composition.

    Type Parameters:
        S: The sampled dimensionality
        C: The dimensionality of the condition matrix
        D: The dimensionality of the discriminable data

    '''

    def __init__(
        self,
        select: Index[D, S],
        condition: Index[C, S],
    ) -> None:
        '''
        Initialize the IdentitySplitter.

        Args:
            select: the index of the selection.
            condition: the index of the condition.

        '''
        self._samples = select.cap
        self._conditions = condition.dim
        self._outputs = select.dim
        self.__select = select
        self.__condition = condition
    @override
    def prepare[N: int](self, data: Matrix[N, S]) -> Matrix[N, D]:
        return data[..., self.__select]
    @override
    def condition[N: int](self, data: Matrix[N, S]) -> Matrix[N, C]:
        return data[..., self.__condition]

class IdentitySelector[S: int, D: int](IdentitySplitter[S, Zero, D]):
    '''
    Subset Splitter for Transformer composition.

    Type Parameters:
        S: The sampled dimensionality
        D: The dimensionality of the discriminable data

    '''

    def __init__(self, index: Index[D, S]) -> None:
        '''
        Initialize the SubsetSplitter.

        Args:
            index: the index of the subset.

        '''
        super().__init__(index, Index.empty(index.cap))

class IdentityConditioner[S: int, C: int, P: int](IdentitySplitter[S, C, Zero]):
    '''
    Blankout Splitter for Transformer composition.

    Type Parameters:
        S: The sampled dimensionality
        C: The dimensionality of the condition matrix
        D: The dimensionality of the discriminable data

    '''

    def __init__(
        self,
        index: Index[C, S],
    ) -> None:
        '''
        Initialize the BlankoutSplitter.

        Args:
            index: the index of the blankout.

        '''
        super().__init__(Index.empty(index.cap), index)
