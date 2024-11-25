from modugant.matrix.dim import One
from modugant.matrix.matrix import Matrix
from modugant.protocols import Updatable, WithConditions, WithOutputs


class Interceptor[C: int, D: int](WithConditions[C], WithOutputs[D], Updatable):
    '''
    Interceptor for Transformer composition.

    Type Parameters:
        C: The dimensionality of the condition matrix
        D: The dimensionality of the discriminable data

    Abstract properties (must be assigned in subclass):
        _conditions: C; The number of conditions.
        _outputs: D; The number of outputs.

    Abstract methods (must be implemented in subclass):
        intercept: Prepare the generated data into discriminable data.
            [N: int](condition: Matrix[N, C], intermediate: Matrix[N, D]) -> Matrix[N, D]
        loss: Compute additional penalization loss on the generated data.
            [N: int](condition: Matrix[N, C], intermediate: Matrix[N, D]) -> Matrix[One, One]
    '''

    def intercept[N: int](self, condition: Matrix[N, C], intermediate: Matrix[N, D]) -> Matrix[N, D]:
        '''Prepare the generated data into discriminable data.'''
        ...
    def loss[N: int](self, condition: Matrix[N, C], intermediate: Matrix[N, D]) -> Matrix[One, One]:
        '''Compute additional penalization loss on the generated data.'''
        ...
