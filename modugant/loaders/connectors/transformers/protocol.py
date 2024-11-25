from typing import Any

from modugant.matrix.matrix import Matrix
from modugant.protocols import Updatable, WithSamples


class Transformer[S: int](WithSamples[S], Updatable):
    '''
    Transformer for Connector composition.

    Type Parameters:
        S: The sampled dimension
        D: The data dimension

    Abstract properties (must be assigned in subclass):
        _samples: S; The number of samples.
        _outputs: D; The number of outputs.

    Abstract methods (must be implemented in subclass):
        load: Transform underlying data.
            [N: int](data: Matrix[N, Any]) -> Matrix[N, S]
        unload: Revert to underlying data.
            [N: int](data: Matrix[N, S]) -> Matrix[N, Any]
        update: Update the connector (default pass).
            () -> None

    '''

    def load[N: int](self, data: Matrix[N, Any]) -> Matrix[N, S]:
        '''Transform underlying data.'''
        ...
    def unload[N: int](self, data: Matrix[N, S]) -> Matrix[N, Any]:
        '''Revert to underlying data.'''
        ...
