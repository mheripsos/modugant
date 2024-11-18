from typing import override

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
        index: Index[D, G]
    ) -> None:
        '''
        Initialize the subset interceptor.

        Args:
            conditions (C: int): The number of conditions.
            intermediates (G: int): The number of generated outputs.
            outputs (D: int): The number of transformed outputs.
            index (Index): The indices into the generated outputs.

        '''
        assert len(index) == outputs
        self._conditions = conditions
        self._intermediates = intermediates
        self._outputs = outputs
        self._index = index
    @override
    def prepare[N: int](self, condition: Matrix[N, C], intermediate: Matrix[N, G]) -> Matrix[N, D]:
        return intermediate[..., self._index]
