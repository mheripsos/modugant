from modugant.matrix.matrix import Matrix
from modugant.protocols import Updatable, WithConditions, WithOutputs, WithSamples


class Splitter[S: int, C: int, D: int](WithSamples[S], WithConditions[C], WithOutputs[D], Updatable):
    '''
    Splitter for Transformer composition.

    Type Parameters:
        S: The sampled dimensionality
        C: The dimensionality of the condition matrix
        D: The dimensionality of the discriminable data

    Abstract properties (must be assigned in subclass):
        _samples: S; The number of samples.
        _conditions: C; The number of conditions.
        _outputs: D; The number of outputs.

    Abstract methods (must be implemented in subclass):
        prepare: Convert sampled data into discriminable data.
            [N: int](data: Matrix[N, S]) -> Matrix[N, D]
        condition: Extract a condition matrix from the underlying data.
            [N: int](data: Matrix[N, S]) -> Matrix[N, C]
        update: Update the Splitter parameters (default pass).
            () -> None

    '''

    def prepare[N: int](self, data: Matrix[N, S]) -> Matrix[N, D]:
        '''Convert sampled data into discriminable data.'''
        ...
    def condition[N: int](self, data: Matrix[N, S]) -> Matrix[N, C]:
        '''Extract a condition matrix from the underlying data.'''
        ...
