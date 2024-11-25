from typing import Sequence, override

from modugant.loaders.connectors.interceptors.protocol import Interceptor
from modugant.matrix.dim import One
from modugant.matrix.index import Index
from modugant.matrix.matrix import Matrix
from modugant.matrix.ops import cat, sums


class JointInterceptor[C: int, D: int](Interceptor[C, D]):
    '''
    Joint Interceptor for Transformer composition.

    Type Parameters:
        C: The dimensionality of the condition matrix
        D: The dimensionality of the discriminable data

    '''

    def __init__(
        self,
        conditions: C,
        outputs: D,
        interceptors: Sequence[Interceptor[int, int]]
    ) -> None:
        '''Initialize the Joint Interceptor.'''
        assert len(interceptors) > 0, 'At least one interceptor is required.'
        assert sum(interceptor.conditions for interceptor in interceptors) == conditions
        assert sum(interceptor.outputs for interceptor in interceptors) == outputs
        self._conditions = conditions
        self._outputs = outputs
        self.__interceptors = interceptors
        sizes = (
            [interceptor.conditions for interceptor in interceptors],
            [interceptor.outputs for interceptor in interceptors]
        )
        self.__backmap = [
            (
                Index.slice(sum(sizes[0][:i]), sizes[0][i], conditions),
                Index.slice(sum(sizes[1][:i]), sizes[1][i], outputs)
            )
            for i in range(len(interceptors))
        ]
    @override
    def intercept[N: int](self, condition: Matrix[N, C], intermediate: Matrix[N, D]) -> Matrix[N, D]:
        '''Prepare the generated data into discriminable data.'''
        intercepted = tuple(
            self.__interceptors[i].intercept(
                condition[..., c_index],
                intermediate[..., d_index]
            )
            for (i, (c_index, d_index)) in enumerate(self.__backmap)
        )
        return cat(intercepted, dim = 1, shape = (condition.shape[0], self._outputs))
    @override
    def loss[N: int](self, condition: Matrix[N, C], intermediate: Matrix[N, D]) -> Matrix[One, One]:
        '''Compute additional penalization loss on the generated data.'''
        losses = tuple(
            self.__interceptors[i].loss(
                condition[..., c_index],
                intermediate[..., d_index]
            )
            for (i, (c_index, d_index)) in enumerate(self.__backmap)
        )
        return sums(losses)
    @override
    def update(self) -> None:
        '''Update the connector (default pass).'''
        for interceptor in self.__interceptors:
            interceptor.update()
