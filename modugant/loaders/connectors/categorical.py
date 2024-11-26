from typing import Sequence, Tuple

from modugant.loaders.connectors.composed import ComposedConnector
from modugant.loaders.connectors.interceptors.softmax import SoftmaxInterceptor
from modugant.loaders.connectors.splitters.composed import ComposedSplitter
from modugant.loaders.connectors.splitters.identity import IdentitySelector
from modugant.loaders.connectors.splitters.sampled import SampledConditioner
from modugant.loaders.connectors.transformers.onehot import OneHotBatchTransformer
from modugant.matrix.index import Index


class CategoricalConnector[S: int](ComposedConnector[S, S, S]):
    '''Categorical connector for Loader composition.'''

    def __init__(
        self,
        size: S,
        indices: Sequence[Tuple[int, int]],
        picks: int = 1
    ) -> None:
        '''
        Initialize the categorical connector.

        Args:
            size (int): The number of outputs.
            indices (List[Tuple[int, int]]): List of index and size tuples.
            picks (int): The number of blocks to sample.

        '''
        super().__init__(
            OneHotBatchTransformer(indices, size),
            ComposedSplitter(
                IdentitySelector(Index.slices(indices, size, size)),
                SampledConditioner(size, size, indices, picks)
            ),
            SoftmaxInterceptor(size, indices)
        )
