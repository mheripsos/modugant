from typing import List, Tuple, override

from modugant.matrix import Matrix
from modugant.matrix.index import Index
from modugant.matrix.ops import cat, one_hot
from modugant.protocols import Loader


class OneHotLoader[D: int, B: int](Loader[D]):
    '''Category loader for GANs.'''

    def __init__(
        self,
        dim: D,
        index: List[Tuple[int, int]]
    ) -> None:
        '''
        Initialize the one-hot loader.

        Args:
            dim (int): The number of outputs.
            index (List[Tuple[int, int]]): The block indices and sizes.

        '''
        sizes = [size for (_, size) in index]
        assert sum(sizes) == dim
        self._outputs = dim
        self._categories = [
            (*index[i], sum(sizes[:i]))
            for i in range(len(sizes))
        ]
    @override
    def load[N: int](self, data: Matrix[N, int]) -> Matrix[N, D]:
        output = tuple(
            one_hot(data[..., Index.at(idx, idx)].long(), size)
            for (idx, size, _) in self._categories
        )
        return cat(output, dim = 1, shape = (data.shape[0], self._outputs))
    @override
    def unload[N: int](self, data: Matrix[N, D]) -> Matrix[N, int]:
        output = tuple(
            data[..., Index.slice(idx, size, self._outputs)].argmax(dim = 1, keepdim = True)
            for (_, size, idx) in self._categories
        )
        return cat(output, dim = 1, shape = (data.shape[0], len(self._categories)))
