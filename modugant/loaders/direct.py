from typing import Any, override

from modugant.matrix import Matrix
from modugant.matrix.index import Index
from modugant.protocols import Loader


class DirectLoader[D: int](Loader[D]):
    '''Direct loader for GANs.'''

    def __init__(
        self,
        dim: D,
        index: Index[D, int]
    ) -> None:
        '''
        Initialize the direct loader.

        Args:
            dim (int): The number of inputs and outputs.
            index (List[Tuple[int, int]]): The start and end block indices

        '''
        self._outputs = dim
        self._index = index
    @override
    def load[N: int](self, data: Matrix[N, int]) -> Matrix[N, D]:
        return data[..., self._index]
    @override
    def unload[N: int](self, data: Matrix[N, D]) -> Matrix[N, Any]:
        return data[..., ...]
