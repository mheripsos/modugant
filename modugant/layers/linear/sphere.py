from typing import Optional, Sequence, override

from modugant.layers.linear.linear import LinearLayer
from modugant.layers.protocol import Layer
from modugant.matrix.index import Index
from modugant.matrix.matrix import Matrix
from modugant.matrix.ops import norm


class SphericalLayer[I: int, L: int, O: int](LinearLayer[I, L, O]):
    '''Linear layer with weights on the unit sphere.'''

    def __init__(
        self,
        dim: O,
        index: Index[L, I],
        follow: Optional[Sequence[Layer[O, O]]] = None,
        bias: bool = True
    ) -> None:
        '''
        Initialize the spherical layer.

        Args:
            dim (int): The number of output nodes.
            index (Index[int, I]): The index.
            follow (Sequence[Layer[O, O]]): The following layers.
            bias (bool): Whether to include a bias term.

        '''
        super().__init__(dim, index, follow, bias)
    @property
    @override
    def weight(self) -> Matrix[L, O]:
        '''Return the weight matrix.'''
        return self._weight / norm(self._weight, dim = 0)
