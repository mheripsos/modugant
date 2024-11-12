from typing import List, override

from modugant.matrix import Matrix
from modugant.matrix.index import Index
from modugant.protocols import Inteceptor


class SubsetInterceptor[C: int, G: int, D: int](Inteceptor[C, G, D]):
    '''Subset interceptor for GANs.'''

    def __init__(
        self,
        conditions: C,
        intermediates: G,
        outputs: D,
        index: List[int]
    ) -> None:
        '''
        Initialize the subset interceptor.

        Args:
            conditions (C: int): The number of conditions.
            intermediates (G: int): The number of generated outputs.
            outputs (D: int): The number of transformed outputs.
            index (List[int]): The column indices.

        '''
        assert len(index) == outputs
        self._conditions = conditions
        self._intermediates = intermediates
        self._outputs = outputs
        self._index = Index.load(index, outputs)
    @override
    def prepare[N: int](self, condition: Matrix[N, C], intermediate: Matrix[N, G]) -> Matrix[N, D]:
        return intermediate[..., self._index]
