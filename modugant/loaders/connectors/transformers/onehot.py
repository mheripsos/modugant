from typing import Any, Sequence, Tuple, override

from modugant.loaders.connectors.transformers.pooled import PooledTransformer
from modugant.loaders.connectors.transformers.protocol import Transformer
from modugant.matrix.dim import One
from modugant.matrix.index import Index
from modugant.matrix.matrix import Matrix
from modugant.matrix.ops import one_hot


class OneHotTransformer[S: int](Transformer[S]):
    '''One-hot Transformer for Connector composition. Converts raw data to one-hot encoding.'''

    def __init__(self, index: Tuple[int, S]) -> None:
        '''
        Initialize the one-hot transformer.

        Args:
            index (Tuple[int, S]): The index and size of the category.

        '''
        self._index = index[0]
        self._samples = index[1]
    @override
    def load[N: int](self, data: Matrix[N, Any]) -> Matrix[N, S]:
        '''Transform underlying data.'''
        return one_hot(data[..., Index.at(self._index, data.shape[1])], self._samples)
    @override
    def unload[N: int](self, data: Matrix[N, S]) -> Matrix[N, One]:
        '''Revert to underlying data.'''
        return data.argmax(dim = 1, keepdim = True)

class OneHotBatchTransformer[S: int](PooledTransformer[S]):
    '''One-hot batch Transformer for Connector composition. Converts raw data to one-hot encoding.'''

    def __init__(self, indices: Sequence[Tuple[int, int]], dim: S) -> None:
        '''
        Initialize the one-hot batch transformer.

        Args:
            indices (List[Tuple[int, int]]): List of index and size tuples.
            dim (int): The number of outputs.

        '''
        super().__init__(dim, [OneHotTransformer(index) for index in indices])
