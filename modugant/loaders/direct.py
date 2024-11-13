from typing import Any, List, Tuple, override

from modugant.matrix import Matrix
from modugant.matrix.index import Index
from modugant.matrix.ops import cat
from modugant.protocols import Loader


class DirectLoader[D: int](Loader[D]):
    '''Direct loader for GANs.'''

    def __init__(
        self,
        dim: D,
        index: List[Tuple[int, int]]
    ) -> None:
        '''
        Initialize the direct loader.

        Args:
            dim (int): The number of inputs and outputs.
            index (List[Tuple[int, int]]): The start and end block indices

        '''
        assert sum([end - start for (start, end) in index]) == dim
        self._outputs = dim
        self._index = [
            Index.slice(start, end - start)
            for (start, end) in index
        ]
    @override
    def load[N: int](self, data: Matrix[N, int]) -> Matrix[N, D]:
        loaded = tuple(
            data[..., i_index]
            for i_index in self._index
        )
        return cat(loaded, dim = 1, shape = (data.shape[0], self._outputs))
    @override
    def unload[N: int](self, data: Matrix[N, D]) -> Matrix[N, Any]:
        return data[..., ...]
