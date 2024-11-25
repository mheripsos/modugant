from modugant.loaders.connectors.interceptors.protocol import Interceptor
from modugant.loaders.connectors.splitters.protocol import Splitter
from modugant.loaders.connectors.transformers.protocol import Transformer


class PreConnector[S: int, C: int, D: int](Splitter[S, C, D], Interceptor[C, D]):
    '''
    Connector for Loader composition.

    Type Parameters:
        S: The dimensionality of the raw underlying data
        C: The dimensionality of the condition matrix
        D: The dimensionality of the discriminable data

    Abstract properties (must be assigned in subclass):
        _samples: S; The number of samples.
        _conditions: C; The number of conditions.
        _outputs: D; The number of outputs

    Abstract methods (must be implemented in subclass):
        condition: Extract a condition matrix from the underlying data.
            [N: int](data: Matrix[N, S]) -> Matrix[N, C]
        prepare: Prepare the generated data into discriminable data.
            [N: int](data: Matrix[N, S]) -> Matrix[N, D]
        intercept: Prepare the generated data into discriminable data.
            [N: int](condition: Matrix[N, C], intermediate: Matrix[N, D]) -> Matrix[N, D]
        loss: Compute additional penalization loss on the generated data.
            [N: int](condition: Matrix[N, C], intermediate: Matrix[N, D]) -> Matrix[One, One]
        update: Update the connector (default pass).
            () -> None
    '''


class Connector[S: int, C: int, D: int](Transformer[S], PreConnector[S, C, D]):
    '''
    Connector for Loader composition.

    Type Parameters:
        S: The dimensionality of the raw underlying data
        C: The dimensionality of the condition matrix
        D: The dimensionality of the discriminable data

    Abstract properties (must be assigned in subclass):
        _samples: S; The number of samples.
        _conditions: C; The number of conditions.
        _outputs: D; The number of outputs

    Abstract methods (must be implemented in subclass):
        load: Transform underlying data.
            [N: int](data: Matrix[N, Any]) -> Matrix[N, R]
        unload: Revert to underlying data.
            [N: int](data: Matrix[N, S]) -> Matrix[N, Any]

    '''
