'''Pooled conditioner for GANs.'''
from typing import Sequence, override

from modugant.matrix import Matrix
from modugant.matrix.ops import cat
from modugant.protocols import Conditioner


class PooledConditioner[C: int, D: int](Conditioner[C, D]):
    '''Pooled conditioner for GANs.'''

    def __init__(
        self,
        conditions: C,
        outputs: D,
        conditioners: Sequence[Conditioner[int, D]]
    ) -> None:
        '''
        Initialize the pooled conditioner.

        Args:
            conditions (C: int): The number of conditions.
            outputs (D: int): The number of generated outputs
            conditioners (List[Conditioner]): The conditioners.

        '''
        assert sum([conditioner.conditions for conditioner in conditioners]) == conditions
        self.__conditioners = conditioners
        self._conditions = conditions
        self._outputs = outputs
    @override
    def condition[N: int](self, data: Matrix[N, D]) -> Matrix[N, C]:
        conditioned = tuple(
            conditioner.condition(data)
            for conditioner in self.__conditioners
        )
        return cat(conditioned, dim = 1, shape = (data.shape[0], self._conditions))
