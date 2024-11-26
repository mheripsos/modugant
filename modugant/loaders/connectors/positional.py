from typing import Tuple

from modugant.loaders.connectors.direct import DirectConnector
from modugant.loaders.connectors.transformers.positional import PositionalTransformer


class PositionalConnector[S: int](DirectConnector[S]):
    '''Standardize connector for Loader composition.'''

    def __init__(
        self,
        index: Tuple[int, int],
        dim: S,
    ) -> None:
        '''
        Initialize the standardize connector.

        Args:
            index (Tuple[int, int]): The index and size of the source column.
            dim (int): The number of outputs.

        '''
        super().__init__(PositionalTransformer(index, dim))
