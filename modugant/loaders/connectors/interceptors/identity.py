from typing import override

from modugant.loaders.connectors.interceptors.protocol import Interceptor
from modugant.matrix.dim import One
from modugant.matrix.matrix import Matrix


class IdentityInterceptor[C: int, D: int](Interceptor[C, D]):
    '''
    Identity Interceptor for Transformer composition.

    Type Parameters:
        C: The dimensionality of the condition matrix
        D: The dimensionality of the discriminable data

    '''

    def __init__(self, conditions: C, outputs: D) -> None:
        '''Initialize the Identity Interceptor.'''
        self._conditions = conditions
        self._outputs = outputs
    @override
    def intercept[N: int](self, condition: Matrix[N, C], intermediate: Matrix[N, D]) -> Matrix[N, D]:
        '''Prepare the generated data into discriminable data.'''
        return intermediate
    @override
    def loss[N: int](self, condition: Matrix[N, C], intermediate: Matrix[N, D]) -> Matrix[One, One]:
        '''Compute additional penalization loss on the generated data.'''
        return Matrix.cell(0.0)
