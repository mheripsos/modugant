from typing import override

from modugant.matrix import Matrix
from modugant.matrix.dim import Dim, One
from modugant.matrix.ops import zeros
from modugant.protocols import Penalizer


class StaticPenalizer[C: int, G: int](Penalizer[C, G]):
    '''Static penalizer for GANs.'''

    def __init__(self, conditions: C, intermediates: G) -> None:
        '''Initialize the static penalizer.'''
        self._conditions = conditions
        self._intermediates = intermediates

    @override
    def loss[N: int](self, condition: Matrix[N, C], intermediate: Matrix[N, G]) -> Matrix[One, One]:
        return zeros((Dim.one(), Dim.one()))
    @override
    def update(self) -> None:
        pass
