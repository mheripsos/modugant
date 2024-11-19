'''NoneConditioner class with no conditions.'''
from typing import override

from torch import float32

from modugant.matrix import Matrix
from modugant.matrix.dim import Dim, Zero
from modugant.matrix.ops import zeros
from modugant.protocols import Conditioner


class NoneConditioner[S: int](Conditioner[S, Zero]):
    '''Unconditional conditioner for GANs.'''

    def __init__(self, sampled: S) -> None:
        '''
        Initialize the unconditional conditioner.

        Args:
            sampled (int): The number of sampled dimensions.

        '''
        super().__init__()
        self._sampled = sampled
        self._conditions = Dim.zero()

    @override
    def condition[N: int](self, data: Matrix[N, S]) -> Matrix[N, Zero]:
        return zeros((data.shape[0], Dim.zero()), dtype = float32)
