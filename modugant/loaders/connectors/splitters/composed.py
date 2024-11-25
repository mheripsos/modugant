from typing import Any, override

from modugant.loaders.connectors.splitters.protocol import Splitter
from modugant.matrix.matrix import Matrix


class ComposedSplitter[S: int, C: int, D: int](Splitter[S, C, D]):
    '''
    Composed Splitter for Transformer composition.

    Type Parameters:
        S: The sampled dimensionality
        C: The dimensionality of the condition matrix
        D: The dimensionality of the discriminable data

    '''

    def __init__(
        self,
        prepare: Splitter[S, Any, D],
        condition: Splitter[S, C, Any],
    ) -> None:
        '''
        Initialize the ComposedSplitter.

        Args:
            prepare: The preparer for the Splitter.
            condition: The conditioner for the Splitter.

        '''
        self._samples = prepare._samples
        self._conditions = condition._conditions
        self._outputs = prepare._outputs
        self.__preparer = prepare
        self.__conditioner = condition
    @override
    def prepare[N: int](self, data: Matrix[N, S]) -> Matrix[N, D]:
        return self.__preparer.prepare(data)
    @override
    def condition[N: int](self, data: Matrix[N, S]) -> Matrix[N, C]:
        return self.__conditioner.condition(data)
    @override
    def update(self) -> None:
        self.__preparer.update()
        self.__conditioner.update()
