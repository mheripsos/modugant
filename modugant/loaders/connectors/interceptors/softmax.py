from typing import Sequence, Tuple, override

from modugant.loaders.connectors.interceptors.protocol import Interceptor
from modugant.matrix.dim import One
from modugant.matrix.index import Index
from modugant.matrix.matrix import Matrix
from modugant.matrix.ops import cat, cross_entropy, sums


class SoftmaxInterceptor[C: int](Interceptor[C, C]):
    '''
    Softmax Interceptor for Transformer composition.

    Type Parameters:
        C: The dimensionality of the condition matrix

    '''

    def __init__(
        self,
        conditions: C,
        blocks: Sequence[Tuple[int, int]]
    ) -> None:
        '''Initialize the Softmax Interceptor.'''
        assert len(blocks) > 0, 'At least one block is required.'
        assert sum([size for (_, size) in blocks]) == conditions
        self._conditions = conditions
        self._outputs = conditions
        self._index = [
            Index.slice(i_index[0], i_index[1], conditions)
            for i_index in blocks
        ]
    @override
    def intercept[N: int](self, condition: Matrix[N, C], intermediate: Matrix[N, C]) -> Matrix[N, C]:
        '''Prepare the generated data into discriminable data.'''
        intercepted = tuple(
            intermediate[..., i_index].softmax(dim = 1)
            for i_index in self._index
        )
        return cat(intercepted, dim = 1, shape = (condition.shape[0], self._outputs))
    @override
    def loss[N: int](self, condition: Matrix[N, C], intermediate: Matrix[N, C]) -> Matrix[One, One]:
        '''Compute additional penalization loss on the generated data.'''
        losses = tuple(
            (
                condition[..., idx].sum(dim = 1, keepdim = True) *
                cross_entropy(
                    intermediate[..., idx],
                    condition[..., idx].argmax(dim = 1, keepdim = True),
                    reduction = 'none'
                )
            )
            for idx in self._index
        )
        return sums(losses).sum(dim = 0, keepdim = True) / condition.shape[0]
