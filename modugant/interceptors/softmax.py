from typing import List, Tuple, override

from modugant.matrix import Matrix
from modugant.matrix.index import Index
from modugant.matrix.ops import cat
from modugant.protocols import Inteceptor


class SoftmaxInterceptor[C: int, G: int, D: int](Inteceptor[C, G, D]):
    '''Softmax interceptor for GANs.'''

    def __init__(
        self,
        conditions: C,
        intermediates: G,
        outputs: D,
        index: List[Tuple[int, int]]
    ) -> None:
        '''
        Initialize the softmax interceptor.

        Args:
            conditions (int): The number of conditions.
            intermediates (int): The number of generated outputs.
            outputs (int): The number of transformed outputs.
            index (List[Tuple[int, int]]): The start and size block indices

        '''
        assert sum([size for (_, size) in index]) == outputs
        self._conditions = conditions
        self._intermediates = intermediates
        self._outputs = outputs
        self._index = [
            Index.slice(i_index[0], i_index[1])
            for i_index in index
        ]
    @override
    def prepare[N: int](self, condition: Matrix[N, C], intermediate: Matrix[N, G]) -> Matrix[N, D]:
        prepared = tuple(
            intermediate[..., i_index].softmax(dim = 1)
            for i_index in self._index
        )
        return cat(prepared, dim = 1, shape = (condition.shape[0], self._outputs))
