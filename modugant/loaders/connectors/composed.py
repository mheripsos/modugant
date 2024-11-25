from typing import Any, override

from modugant.loaders.connectors.interceptors.protocol import Interceptor
from modugant.loaders.connectors.protocol import Connector, PreConnector
from modugant.loaders.connectors.splitters.protocol import Splitter
from modugant.loaders.connectors.transformers.protocol import Transformer
from modugant.matrix.dim import One
from modugant.matrix.matrix import Matrix


class ComposedPreConnector[S: int, C: int, D: int](PreConnector[S, C, D]):
    '''
    Connector for Loader composition.

    Type Parameters:
        S: The dimensionality of the raw underlying data
        C: The dimensionality of the condition matrix
        D: The dimensionality of the discriminable data

    '''

    def __init__(
        self,
        splitter: Splitter[S, C, D],
        interceptor: Interceptor[C, D],
    ) -> None:
        '''
        Initialize the composed connector.

        Args:
            transformer (Transformer[R]): The transformer.
            splitter (Splitter[R, C, D]): The splitter.
            interceptor (Interceptor[C, D]): The interceptor.

        '''
        self._samples = splitter._samples
        self._conditions = splitter._conditions
        self._outputs = interceptor._outputs
        self.__splitter = splitter
        self.__interceptor = interceptor
    @override
    def condition[N: int](self, data: Matrix[N, S]) -> Matrix[N, C]:
        return self.__splitter.condition(data)
    @override
    def prepare[N: int](self, data: Matrix[N, S]) -> Matrix[N, D]:
        return self.__splitter.prepare(data)
    @override
    def intercept[N: int](self, condition: Matrix[N, C], intermediate: Matrix[N, D]) -> Matrix[N, D]:
        return self.__interceptor.intercept(condition, intermediate)
    @override
    def loss[N: int](self, condition: Matrix[N, C], intermediate: Matrix[N, D]) -> Matrix[One, One]:
        return self.__interceptor.loss(condition, intermediate)
    @override
    def update(self) -> None:
        self.__splitter.update()
        self.__interceptor.update()

class ComposedConnector[S: int, C: int, D: int](ComposedPreConnector[S, C, D], Connector[S, C, D]):
    '''Composed connector for Loader composition.'''

    def __init__(
        self,
        transformer: Transformer[S],
        splitter: Splitter[S, C, D],
        interceptor: Interceptor[C, D],
    ) -> None:
        '''
        Initialize the composed connector.

        Args:
            transformer (Transformer[R]): The transformer.
            splitter (Splitter[R, C, D]): The splitter.
            interceptor (Interceptor[C, D]): The interceptor.

        '''
        self._samples = transformer._samples
        self._conditions = splitter._conditions
        self._outputs = interceptor._outputs
        self.__transformer = transformer
        self.__splitter = splitter
        self.__interceptor = interceptor
    @override
    def load[N: int](self, data: Matrix[N, Any]) -> Matrix[N, S]:
        return self.__transformer.load(data)
    @override
    def unload[N: int](self, data: Matrix[N, S]) -> Matrix[N, Any]:
        return self.__transformer.unload(data)
    @override
    def update(self) -> None:
        super().update()
        self.__transformer.update()
