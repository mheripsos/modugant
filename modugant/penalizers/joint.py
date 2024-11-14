from typing import Sequence, override

from modugant.matrix import Matrix
from modugant.matrix.dim import One
from modugant.matrix.index import Index
from modugant.matrix.ops import sums
from modugant.protocols import Penalizer


class JointPenalizer[C: int, G: int](Penalizer[C, G]):
    '''Joint penalizer for GANs.'''

    def __init__(
        self,
        conditions: C,
        intermediates: G,
        penalizers: Sequence[Penalizer[int, int]]
    ) -> None:
        '''
        Initialize the joint penalizer.

        Args:
            conditions (C: int): The number of conditions.
            intermediates (G: int): The number of generated outputs
            penalizers (Sequence[Penalizer[int, int]]): The penalizers.

        '''
        assert sum([penalizer.conditions for penalizer in penalizers]) == conditions
        assert sum([penalizer.intermediates for penalizer in penalizers]) == intermediates
        self._conditions = conditions
        self._intermediates = intermediates
        self.__penalizers = penalizers
        sizes = [
            [penalizer.conditions for penalizer in penalizers],
            [penalizer.intermediates for penalizer in penalizers]
        ]
        self.__backmap = [
            (
                Index.slice(sum(sizes[0][:i]), sizes[0][i]),
                Index.slice(sum(sizes[1][:i]), sizes[1][i])
            )
            for i in range(len(penalizers))
        ]
    @override
    def loss[N: int](self, condition: Matrix[N, C], intermediate: Matrix[N, G]) -> Matrix[One, One]:
        losses = tuple(
            self.__penalizers[i].loss(
                condition[..., c_idx],
                intermediate[..., d_idx]
            )
            for (i, (c_idx, d_idx)) in enumerate(self.__backmap)
        )
        return sums(losses)
    @override
    def update(self) -> None:
        for penalizer in self.__penalizers:
            penalizer.update()
