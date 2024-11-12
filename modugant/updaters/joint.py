from typing import Sequence, override

from modugant.matrix import Matrix
from modugant.matrix.dim import One
from modugant.matrix.index import Index
from modugant.matrix.ops import sums
from modugant.protocols import Updater


class JointUpdater[C: int, G: int](Updater[C, G]):
    '''Joint updater for GANs.'''

    def __init__(
        self,
        conditions: C,
        intermediates: G,
        updaters: Sequence[Updater[int, int]]
    ) -> None:
        '''
        Initialize the joint updater.

        Args:
            conditions (C: int): The number of conditions.
            intermediates (G: int): The number of generated outputs
            updaters (Sequence[Updater]): The updaters.

        '''
        assert sum([updater.conditions for updater in updaters]) == conditions
        assert sum([updater.intermediates for updater in updaters]) == intermediates
        self._conditions = conditions
        self._intermediates = intermediates
        self.__updaters = updaters
        sizes = [
            [updater.conditions for updater in updaters],
            [updater.intermediates for updater in updaters]
        ]
        self.__backmap = [
            (
                Index.slice(sum(sizes[0][:i]), sizes[0][i]),
                Index.slice(sum(sizes[1][:i]), sizes[1][i])
            )
            for i in range(len(updaters))
        ]
    @override
    def loss[N: int](self, condition: Matrix[N, C], intermediate: Matrix[N, G]) -> Matrix[One, One]:
        losses = tuple(
            self.__updaters[i].loss(
                condition[..., c_idx],
                intermediate[..., d_idx]
            )
            for (i, (c_idx, d_idx)) in enumerate(self.__backmap)
        )
        return sums(losses)
    @override
    def update(self) -> None:
        for updater in self.__updaters:
            updater.update()
