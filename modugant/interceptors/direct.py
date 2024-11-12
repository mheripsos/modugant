from typing import override

from modugant.matrix import Matrix
from modugant.protocols import Inteceptor


class DirectInterceptor[C: int, D: int](Inteceptor[C, D, D]):
    '''Direct interceptor for GANs.'''

    def __init__(
        self,
        conditions: C,
        dim: D
    ) -> None:
        '''
        Initialize the direct interceptor.

        Args:
            conditions (int): The number of conditions.
            dim (D: int): The number of inputs and outputs.

        '''
        self._conditions = conditions
        self._intermediates = dim
        self._outputs = dim
    @override
    def prepare[N: int](self, condition: Matrix[N, C], intermediate: Matrix[N, D]) -> Matrix[N, D]:
        return intermediate
