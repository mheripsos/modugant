from typing import override

from modugant.loaders.connectors.interceptors.protocol import Interceptor
from modugant.matrix.dim import One
from modugant.matrix.matrix import Matrix


class ComposedInterceptor[C: int, D: int](Interceptor[C, D]):
    '''
    Composed Interceptor for Transformer composition.

    Type Parameters:
        C: The dimensionality of the condition matrix
        D: The dimensionality of the discriminable data

    '''

    def __init__(self, interceptor: Interceptor[C, D], penalizer: Interceptor[C, D]) -> None:
        '''Initialize the Identity Interceptor.'''
        self._conditions = interceptor.conditions
        self._outputs = interceptor.outputs
        self.__interceptor = interceptor
        self.__penalizer = penalizer
    @override
    def intercept[N: int](self, condition: Matrix[N, C], intermediate: Matrix[N, D]) -> Matrix[N, D]:
        '''Prepare the generated data into discriminable data.'''
        return self.__interceptor.intercept(condition, intermediate)
    @override
    def loss[N: int](self, condition: Matrix[N, C], intermediate: Matrix[N, D]) -> Matrix[One, One]:
        '''Compute additional penalization loss on the generated data.'''
        return self.__penalizer.loss(condition, intermediate)
    @override
    def update(self) -> None:
        '''Update the connector (default pass).'''
        self.__interceptor.update()
        self.__penalizer.update()

