from typing import Sequence, override

from modugant.matrix import Matrix
from modugant.matrix.dim import One
from modugant.matrix.ops import sums
from modugant.protocols import Penalizer


class PooledPenalizer[C: int, G: int](Penalizer[C, G]):
    '''Pooled penalizer for GANs.'''

    def __init__(
        self,
        conditions: C,
        intermediates: G,
        penalizers: Sequence[Penalizer[C, G]]
    ) -> None:
        '''
        Initialize the pooled penalizer.

        Args:
            conditions (C: int): The number of conditions.
            intermediates (G: int): The number of generated outputs
            penalizers (Sequence[Penalizer[C, G]]): The penalizers.

        '''
        assert all([penalizer.conditions == conditions for penalizer in penalizers])
        assert all([penalizer.intermediates == intermediates for penalizer in penalizers])
        self._conditions = conditions
        self._intermediates = intermediates
        self.__penalizers = penalizers
    @override
    def loss[N: int](self, condition: Matrix[N, C], intermediate: Matrix[N, G]) -> Matrix[One, One]:
        losses = tuple(
            penalizer.loss(condition, intermediate)
            for penalizer in self.__penalizers
        )
        return sums(losses)
    @override
    def update(self) -> None:
        for penalizer in self.__penalizers:
            penalizer.update()
