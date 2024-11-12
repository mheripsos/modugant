'''Joint Conditioner.'''
from typing import Sequence, override

from modugant.matrix import Index, Matrix
from modugant.matrix.ops import cat
from modugant.protocols import Conditioner


class JointConditioner[C: int, D: int](Conditioner[C, D]):
    '''Joint conditioner for GANs.'''

    def __init__(
        self,
        conditions: C,
        outputs: D,
        conditioners: Sequence[Conditioner[int, int]]
    ) -> None:
        '''
        Initialize the joint conditioner.

        Args:
            conditions (C: int): The number of conditions.
            outputs (D: int): The number of generated outputs
            conditioners (List[Conditioner]): The conditioners.

        '''
        assert sum([conditioner.conditions for conditioner in conditioners]) == conditions
        assert sum([conditioner.outputs for conditioner in conditioners]) == outputs
        self.__conditioners = conditioners
        self._conditions = conditions
        self._outputs = outputs
        sizes = [conditioner.outputs for conditioner in conditioners]
        self.__backmap = [
            Index.slice(sum(sizes[:i]), sizes[i])
            for i in range(len(conditioners))
        ]
    @override
    def condition[N: int](self, data: Matrix[N, D]) -> Matrix[N, C]:
        conditioned = tuple(
            self.__conditioners[i].condition(data[..., slice])
            for (i, slice) in enumerate(self.__backmap)
        )
        return cat(conditioned, dim = 1, shape = (data.shape[0], self._conditions))
