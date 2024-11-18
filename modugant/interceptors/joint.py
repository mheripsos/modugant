from typing import Sequence, override

from modugant.matrix import Matrix
from modugant.matrix.index import Index
from modugant.matrix.ops import cat
from modugant.protocols import Inteceptor


class JointInterceptor[C: int, G: int, D: int](Inteceptor[C, G, D]):
    '''Joint inteceptor for GANs.'''

    def __init__(
        self,
        conditions: C,
        intermediates: G,
        outputs: D,
        interceptors: Sequence[Inteceptor[int, int, int]]
    ) -> None:
        '''
        Initialize the joint inteceptor.

        Args:
            conditions (C: int): The number of conditions.
            intermediates (G: int): The number of generated outputs.
            outputs (D: int): The number of transformed outputs.
            interceptors (Sequence[Inteceptor]): The interceptors.

        '''
        assert sum([inteceptor.conditions for inteceptor in interceptors]) == conditions
        assert sum([inteceptor.intermediates for inteceptor in interceptors]) == intermediates
        assert sum([inteceptor.outputs for inteceptor in interceptors]) == outputs
        self.__interceptors = interceptors
        self._conditions = conditions
        self._intermediates = intermediates
        self._outputs = outputs
        sizes = (
            [interceptor.conditions for interceptor in interceptors],
            [interceptor.intermediates for interceptor in interceptors]
        )
        self.__backmap = [
            (
                Index.slice(sum(sizes[0][:i]), sizes[0][i], conditions),
                Index.slice(sum(sizes[1][:i]), sizes[1][i], intermediates)
            )
            for i in range(len(interceptors))
        ]
    @override
    def prepare[N: int](self, condition: Matrix[N, C], intermediate: Matrix[N, G]) -> Matrix[N, D]:
        prepared = tuple(
            self.__interceptors[i].prepare(
                condition[..., c_index],
                intermediate[..., d_index]
            )
            for (i, (c_index, d_index)) in enumerate(self.__backmap)
        )
        return cat(prepared, dim = 1, shape = (condition.shape[0], self._outputs))
