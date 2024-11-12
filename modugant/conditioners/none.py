'''NoneConditioner class with no conditions.'''
from typing import override

from torch import float32

from modugant.matrix import Matrix
from modugant.matrix.dim import Dim, Zero
from modugant.matrix.ops import zeros
from modugant.protocols import Conditioner


class NoneConditioner[D: int](Conditioner[Zero, D]):
    '''Unconditional conditioner for GANs.'''

    def __init__(self, outputs: D) -> None:
        '''
        Initialize the unconditional conditioner.

        Args:
            outputs (D: int): The number of generated outputs.

        '''
        super().__init__()
        self._conditions = Dim.zero()
        self._outputs = outputs

    @override
    def condition[N: int](self, data: Matrix[N, D]) -> Matrix[N, Zero]:
        return zeros((data.shape[0], Dim.zero()), dtype = float32)
