from typing import Any, override

from modugant.matrix import Matrix
from modugant.matrix.index import Index
from modugant.protocols import Loader


class DirectLoader[S: int, D: int](Loader[S, D]):
    '''Direct loader for GANs.'''

    def __init__(
        self,
        index: Index[D, S]
    ) -> None:
        '''
        Initialize the direct loader.

        Args:
            dim (int): The number of inputs and outputs.
            index (List[Tuple[int, int]]): The start and end block indices

        '''
        self._sampled = index.cap
        self._outputs = index.dim
        self._index = index
    @override
    def load[N: int](self, data: Matrix[N, S]) -> Matrix[N, D]:
        return data[..., self._index]
    @override
    def unload[N: int](self, data: Matrix[N, D]) -> Matrix[N, Any]:
        return data[..., ...]
