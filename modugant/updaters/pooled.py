from typing import Sequence, override

from modugant.matrix import Matrix
from modugant.matrix.dim import One
from modugant.matrix.ops import sums
from modugant.protocols import Updater


class PooledUpdater[C: int, G: int](Updater[C, G]):
    '''Pooled updater for GANs.'''

    def __init__(
        self,
        conditions: C,
        intermediates: G,
        updaters: Sequence[Updater[C, G]]
    ) -> None:
        '''
        Initialize the pooled updater.

        Args:
            conditions (C: int): The number of conditions.
            intermediates (G: int): The number of generated outputs
            updaters (Sequence[Updater]): The updaters.

        '''
        assert all([updater.conditions == conditions for updater in updaters])
        assert all([updater.intermediates == intermediates for updater in updaters])
        self._conditions = conditions
        self._intermediates = intermediates
        self.__updaters = updaters
    @override
    def loss[N: int](self, condition: Matrix[N, C], intermediate: Matrix[N, G]) -> Matrix[One, One]:
        losses = tuple(
            updater.loss(condition, intermediate)
            for updater in self.__updaters
        )
        return sums(losses)
    @override
    def update(self) -> None:
        for updater in self.__updaters:
            updater.update()
