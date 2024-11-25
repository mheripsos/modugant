from typing import Any, Sequence, Tuple, override

from torch import pi

from modugant.loaders.connectors.transformers.pooled import PooledTransformer
from modugant.loaders.connectors.transformers.protocol import Transformer
from modugant.matrix.dim import Dim, One
from modugant.matrix.index import Index
from modugant.matrix.matrix import Matrix
from modugant.matrix.ops import arange, cat


class PositionalTransformer[S: int](Transformer[S]):
    '''Positional Transformer for Connector composition. Converts raw data to positional encoding.'''

    @staticmethod
    def encode[NS: int, DS: int](encoder: Matrix[One, DS], data: Matrix[NS, One]) -> Matrix[NS, DS]:
        '''
        Positionally encode a column of ordinal data.

        Args:
            encoder (Matrix[One, DS]): The encoding row-vector as Matrix.
            data (Matrix[NS, One]): The column-vector data as Matrix to encode.

        Returns:
            Matrix[NS, DS]: The encoded data.

        '''
        exp = data @ encoder
        dim = encoder.shape[1]
        outputs = tuple(
            exp[..., Index.at(i, dim)].sin() if i % 2 == 0 else exp[..., Index.at(i, dim)].cos()
            for i in range(dim)
        )
        return cat(outputs, dim = 1, shape = (data.shape[0], dim))
    def __init__(self, index: Tuple[int, int], dim: S) -> None:
        '''
        Initialize the positional transformer.

        Args:
            index (Tuple[int, int]): The index and size of the source column.
            dim (int): The number of outputs.

        '''
        self._samples = dim
        self.__start = index[0]
        size = index[1]
        base = Matrix.cell((size / pi) ** (dim / (dim - 2)))
        self._encoder = cat(
            tuple(
                base ** (- 2 * (i // 2) / dim)
                for i in range(dim)
            ),
            dim = 1,
            shape = (Dim.one(), dim)
        )
        self.__decoder = PositionalTransformer.encode(self._encoder, arange(size + 1)).T
    @override
    def load[N: int](self, data: Matrix[N, Any]) -> Matrix[N, S]:
        '''Transform underlying data.'''
        return PositionalTransformer.encode(self._encoder, data[..., Index.at(self.__start, data.shape[1])])
    @override
    def unload[N: int](self, data: Matrix[N, S]) -> Matrix[N, Any]:
        '''Revert to underlying data.'''
        candidate = data @ self.__decoder
        return candidate.argmax(dim = 1, keepdim = True)

class PositionalBatchTransformer[S: int](PooledTransformer[S]):
    '''Positional batch Transformer for Connector composition. Converts raw data to positional encoding.'''

    def __init__(self, indices: Sequence[Tuple[int, int]], dim: S) -> None:
        '''
        Initialize the positional batch transformer.

        Args:
            indices (List[Tuple[int, int]]): List of index and size tuples.
            dim (int): The number of outputs per transformer.

        '''
        super().__init__(dim, [PositionalTransformer(index, dim) for index in indices])
