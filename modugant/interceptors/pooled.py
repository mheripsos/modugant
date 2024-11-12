from typing import Sequence, override

from modugant.matrix import Matrix
from modugant.matrix.ops import cat
from modugant.protocols import Inteceptor


class PooledInterceptor[C: int, G: int, D: int](Inteceptor[C, G, D]):
    '''Pooled interceptor for GANs.'''

    def __init__(
        self,
        conditions: C,
        intermediates: G,
        outputs: D,
        interceptors: Sequence[Inteceptor[C, G, int]]
    ) -> None:
        '''
        Initialize the pooled interceptor.

        Args:
            conditions (C: int): The number of conditions.
            intermediates (G: int): The number of generated outputs.
            outputs (D: int): The number of transformed outputs.
            interceptors (Sequence[Inteceptor]): The interceptors.

        '''
        assert all([inteceptor.conditions == conditions for inteceptor in interceptors])
        assert all([inteceptor.intermediates == intermediates for inteceptor in interceptors])
        assert sum([inteceptor.outputs for inteceptor in interceptors]) == outputs
        self.__interceptors = interceptors
        self._conditions = conditions
        self._intermediates = intermediates
        self._outputs = outputs
    @override
    def prepare[N: int](self, condition: Matrix[N, C], intermediate: Matrix[N, G]) -> Matrix[N, D]:
        prepared = tuple(
            interceptor.prepare(condition, intermediate)
            for interceptor in self.__interceptors
        )
        return cat(prepared, dim = 1, shape = (condition.shape[0], self._outputs))
