from modugant.loaders.connectors.direct import DirectConnector
from modugant.loaders.connectors.transformers.standardize import StandardizeTransformer
from modugant.matrix.index import Index
from modugant.matrix.matrix import Matrix


class StandardizeConnector[S: int](DirectConnector[S]):
    '''Standardize connector for Loader composition.'''

    def __init__(
        self,
        data: Matrix[int, int],
        index: Index[S, int]
    ) -> None:
        '''
        Initialize the standardize connector.

        Args:
            data (Matrix): The data from which to compute mean and variance.
            index (Index): The index of the data to standardize.

        '''
        super().__init__(StandardizeTransformer(data, index))
