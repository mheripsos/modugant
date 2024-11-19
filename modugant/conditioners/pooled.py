'''Joint Conditioner.'''
from typing import Sequence, override

from modugant.matrix import Matrix
from modugant.matrix.ops import cat
from modugant.protocols import Conditioner


class PooledConditioner[S: int, C: int](Conditioner[S, C]):
    '''Pooled conditioner for GANs.'''

    def __init__(
        self,
        conditions: C,
        conditioners: Sequence[Conditioner[S, int]]
    ) -> None:
        '''
        Initialize the joint conditioner.

        Args:
            conditions (C: int): The number of conditions.
            conditioners (List[Conditioner]): The conditioners.

        '''
        assert sum([conditioner.conditions for conditioner in conditioners]) == conditions
        self._sampled = conditioners[0].sampled
        self._conditions = conditions
        self.__conditioners = conditioners
    @override
    def condition[N: int](self, data: Matrix[N, S]) -> Matrix[N, C]:
        conditioned = tuple(
            conditioner.condition(data)
            for conditioner in self.__conditioners
        )
        return cat(conditioned, dim = 1, shape = (data.shape[0], self._conditions))
