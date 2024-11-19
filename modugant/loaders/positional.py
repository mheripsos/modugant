from math import pi
from typing import Any, override

from modugant.matrix import Dim, Matrix
from modugant.matrix.dim import One
from modugant.matrix.index import Index
from modugant.matrix.ops import arange, cat
from modugant.protocols import Loader


class PositionalLoader[S: int, D: int](Loader[S, D]):
    '''Positional loader for GANs.'''

    def __init__(
        self,
        sampled: S,
        dim: D,
        index: int,
        max: int
    ) -> None:
        '''
        Initialize the positional loader.

        Args:
            sampled (int): The number of sampled dimensions
            dim (int): The number of outputs, a power of two.
            index (int): The index to load.
            max (int): The maximum value.

        '''
        assert dim >= 2 and dim & (dim - 1) == 0, 'Dimension must be a power of two.'
        assert max > 1, 'Size must be greater than one.'
        assert 0 <= index < sampled, 'Index must be in range.'
        self._sampled = sampled
        self._outputs = dim
        self.__index = index
        base = Matrix.cell((max / pi) ** (dim / (dim - 2)))
        self.__encoder = cat(
            tuple(
                base ** (- 2 * (i // 2) / dim)
                for i in range(dim)
            ),
            dim = 1,
            shape = (Dim.one(), dim)
        )
        self.__decoder = self._encode(arange(max + 1)).T
    def _encode[N: int](self, data: Matrix[N, One]) -> Matrix[N, D]:
        exp = data @ self.__encoder
        outputs = tuple(
            exp[..., Index.at(i, self._outputs)].sin() if i % 2 == 0 else exp[..., Index.at(i, self._outputs)].cos()
            for i in range(self._outputs)
        )
        return cat(outputs, dim = 1, shape = (data.shape[0], self._outputs))
    @override
    def load[N: int](self, data: Matrix[N, S]) -> Matrix[N, D]:
        return self._encode(data[..., Index.at(self.__index, self._sampled)])
    @override
    def unload[N: int](self, data: Matrix[N, D]) -> Matrix[N, Any]:
        candidate = data @ self.__decoder
        return candidate.argmax(dim = 1, keepdim = True)
