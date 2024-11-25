from typing import Sequence, override

from modugant.loaders.connectors.protocol import Connector, PreConnector
from modugant.matrix.dim import One
from modugant.matrix.index import Index
from modugant.matrix.matrix import Matrix
from modugant.matrix.ops import cat, sums


class JointPreConnector[S: int, C: int, D: int](PreConnector[S, C, D]):
    '''Joint connector for Loader composition.'''

    def __init__(
        self,
        samples: S,
        conditions: C,
        outputs: D,
        connectors: Sequence[PreConnector[int, int, int]],
    ) -> None:
        '''
        Initialize the joint connector.

        Args:
            samples (S: int): The number of samples.
            conditions (C: int): The number of conditions.
            outputs (D: int): The number of outputs.
            connectors (Sequence[Connector[int, int, int]]): The connectors to join.

        '''
        sizes = (
            [connector.samples for connector in connectors],
            [connector.conditions for connector in connectors],
            [connector.outputs for connector in connectors],
        )
        assert(sum(connector.samples for connector in connectors) == samples)
        assert(sum(connector.conditions for connector in connectors) == conditions)
        assert(sum(connector.outputs for connector in connectors) == outputs)
        self._samples = samples
        self._conditions = conditions
        self._outputs = outputs
        self._connectors = connectors
        self._map = [
            (sum(sizes[0][:i]), sum(sizes[1][:i]), sum(sizes[2][:i]))
            for i in range(len(connectors))
        ]
    def __getitem__(self, index: int) -> PreConnector[int, int, int]:
        '''Get the Connector at the given index.'''
        return self._connectors[index]
    @override
    def condition[N: int](self, data: Matrix[N, S]) -> Matrix[N, C]:
        conditioned = tuple(
            self._connectors[i].condition(
                data[
                    ...,
                    Index.slice(self._map[i][0], self._connectors[i].samples, self._samples)
                ]
            )
            for i in range(len(self._connectors))
        )
        return cat(conditioned, dim=1, shape=(data.shape[0], self._conditions))
    @override
    def prepare[N: int](self, data: Matrix[N, S]) -> Matrix[N, D]:
        prepared = tuple(
            self._connectors[i].prepare(
                data[
                    ...,
                    Index.slice(self._map[i][0], self._connectors[i].samples, self._samples)
                ]
            )
            for i in range(len(self._connectors))
        )
        return cat(prepared, dim=1, shape=(data.shape[0], self._outputs))
    @override
    def intercept[N: int](self, condition: Matrix[N, C], intermediate: Matrix[N, D]) -> Matrix[N, D]:
        intercepted = tuple(
            self._connectors[i].intercept(
                condition[
                    ...,
                    Index.slice(self._map[i][1], self._connectors[i].conditions, self._conditions)
                ],
                intermediate[
                    ...,
                    Index.slice(self._map[i][2], self._connectors[i].outputs, self._outputs)
                ]
            )
            for i in range(len(self._connectors))
        )
        return cat(intercepted, dim=1, shape=(condition.shape[0], self._outputs))
    @override
    def loss[N: int](self, condition: Matrix[N, C], intermediate: Matrix[N, D]) -> Matrix[One, One]:
        losses = tuple(
            self._connectors[i].loss(
                condition[
                    ...,
                    Index.slice(self._map[i][1], self._connectors[i].conditions, self._conditions)
                ],
                intermediate[
                    ...,
                    Index.slice(self._map[i][2], self._connectors[i].outputs, self._outputs)
                ]
            )
            for i in range(len(self._connectors))
        )
        return sums(losses)
    @override
    def update(self) -> None:
        for connector in self._connectors:
            connector.update()


class JointConnector[S: int, C: int, D: int](JointPreConnector[S, C, D], Connector[S, C, D]):
    '''Joint connector for Loader composition.'''

    def __init__(
        self,
        samples: S,
        conditions: C,
        outputs: D,
        connectors: Sequence[Connector[int, int, int]],
    ) -> None:
        '''
        Initialize the joint connector.

        Args:
            samples (S: int): The number of samples.
            conditions (C: int): The number of conditions.
            outputs (D: int): The number of outputs.
            connectors (Sequence[Connector[int, int, int]]): The connectors to join.

        '''
        super().__init__(samples, conditions, outputs, connectors)
        self._connectors = connectors
    @override
    def load[N: int](self, data: Matrix[N, S]) -> Matrix[N, D]:
        loaded = tuple(
            connector.load(data) for connector in self._connectors
        )
        return cat(loaded, dim=1, shape=(data.shape[0], self._outputs))
    @override
    def unload[N: int](self, data: Matrix[N, D]) -> Matrix[N, S]:
        unloaded = tuple(
            self._connectors[i].unload(
                data[
                    ...,
                    Index.slice(self._map[i][2], self._connectors[i].outputs, self._outputs)
                ]
            )
            for i in range(len(self._connectors))
        )
        return cat(unloaded, dim=1, shape=(data.shape[0], self._samples))
