from typing import Any, Sequence, override

from modugant.loaders.connectors.splitters.protocol import Splitter
from modugant.matrix.matrix import Matrix
from modugant.matrix.ops import cat


class JointSplitter[S: int, C: int, D: int](Splitter[S, C, D]):
    '''
    Joint Splitter for Transformer composition.

    Type Parameters:
        S: The sampled dimensionality
        C: The dimensionality of the condition matrix
        D: The dimensionality of the discriminable data

    '''

    def __init__(
        self,
        samples: S,
        conditions: C,
        outputs: D,
        splitters: Sequence[Splitter[Any, int, int]],
    ) -> None:
        '''
        Initialize the JointSplitter.

        Args:
            samples: the sample dimensionality.
            conditions: the condition dimensionality.
            outputs: the output dimensionality.
            splitters: the splitters to join.

        '''
        assert sum(splitter.samples for splitter in splitters) == samples
        assert sum(splitter.conditions for splitter in splitters) == conditions
        assert sum(splitter.outputs for splitter in splitters) == outputs
        self._samples = samples
        self._conditions = conditions
        self._outputs = outputs
        self.__splitters = splitters
    def __getitem__(self, index: int) -> Splitter[int, int, int]:
        '''Get the Splitter at the given index.'''
        return self.__splitters[index]
    @override
    def prepare[N: int](self, data: Matrix[N, S]) -> Matrix[N, D]:
        prepared = tuple(
            splitter.prepare(data) for splitter in self.__splitters
        )
        return cat(prepared, dim=1, shape = (data.shape[0], self._outputs))
    @override
    def condition[N: int](self, data: Matrix[N, S]) -> Matrix[N, C]:
        conditioned = tuple(
            splitter.condition(data) for splitter in self.__splitters
        )
        return cat(conditioned, dim=1, shape = (data.shape[0], self._conditions))
    @override
    def update(self) -> None:
        for splitter in self.__splitters:
            splitter.update()
