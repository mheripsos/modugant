from modugant.loaders.connectors.composed import ComposedConnector, ComposedPreConnector
from modugant.loaders.connectors.interceptors.identity import IdentityInterceptor
from modugant.loaders.connectors.splitters.identity import IdentitySelector
from modugant.loaders.connectors.transformers.protocol import Transformer
from modugant.matrix.dim import Dim, Zero
from modugant.matrix.index import Index


class DirectPreConnector[S: int](ComposedPreConnector[S, Zero, S]):
    '''Direct connector for Loader composition.'''

    def __init__(self, dim: S) -> None:
        '''
        Initialize the direct connector.

        Args:
            dim (S): The dimensionality of the raw data.

        '''
        super().__init__(
            IdentitySelector(Index.range(dim)),
            IdentityInterceptor(Dim.zero(), dim)
        )

class DirectConnector[S: int](ComposedConnector[S, Zero, S]):
    '''Direct connector for Loader composition.'''

    def __init__(
        self,
        transformer: Transformer[S],
    ) -> None:
        '''
        Initialize the direct connector.

        Args:
            transformer (Transformer[S]): The transformer.

        '''
        super().__init__(
            transformer,
            IdentitySelector(Index.range(transformer.samples)),
            IdentityInterceptor(Dim.zero(), transformer.samples)
        )
