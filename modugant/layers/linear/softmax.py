from math import sqrt
from typing import Optional, Sequence, override

from modugant.layers.linear.linear import LinearLayer
from modugant.layers.protocol import Layer
from modugant.matrix.index import Index
from modugant.matrix.matrix import Matrix


class SoftMaxLayer[I: int, L: int, O: int](LinearLayer[I, L, O]):
    '''Linear layer with crossing effects.'''

    def __init__(
        self,
        dim: O,
        index: Index[L, I],
        follow: Optional[Sequence[Layer[O, O]]] = None,
        bias: bool = True
    ) -> None:
        '''
        Initialize the softmaxed linear layer.

        Args:
            dim (int): The number of output nodes.
            index (Index[int, I]): The index.
            follow (Sequence[Layer[O, O]]): The following layers.
            bias (bool): Whether to include a bias term.

        '''
        super().__init__(dim, index, follow, bias)
    @override
    def _finish[N: int](self, output: Matrix[N, O]) -> Matrix[N, O]:
        '''Finish the forward pass of the linear transformation.'''
        return (output / sqrt(self._dim)).softmax(dim = 1)
