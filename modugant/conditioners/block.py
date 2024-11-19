'''Block Conditioner.'''
from typing import List, Tuple, override

from modugant.matrix import Index, Matrix
from modugant.matrix.dim import Dim
from modugant.matrix.ops import cat, one_hot, ones, randint, sums
from modugant.protocols import Conditioner


class BlockConditioner[S: int, C: int, P: int](Conditioner[S, C]):
    '''Block conditioner for GANs.'''

    def __init__(
        self,
        sampled: S,
        conditions: C,
        index: List[Tuple[int, int]],
        picks: P
    ) -> None:
        '''
        Initialize the block conditioner.

        Args:
            sampled (int): The number of sampled dimensions.
            conditions (C: int): The number of conditions.
            index (List[Tuple[int, int]]): The start and size block indices.
            picks (int): The number of blocks to sample.

        '''
        assert sum([size for (_, size) in index]) == conditions
        assert picks <= len(index) and picks > 0
        self._sampled = sampled
        self._conditions = conditions
        self.__picks = picks
        self.__index = [
            Index.slice(start, size, sampled)
            for (start, size) in index
        ]
        self.__chunks = len(index)
        self.__map = cat(
            tuple(
                one_hot(i * ones((size, Dim.one())).long(), self.__chunks)
                for (i, (_, size)) in enumerate(index)
            ),
            dim = 0,
            shape = (conditions, self.__chunks)
        )
    def _clone[N: int](self, data: Matrix[N, S]) -> Matrix[N, C]:
        # clone the portion of the data from which conditions are drawn
        clones = tuple(
            data[..., slice].clone().detach()
            for slice in self.__index
        )
        return cat(clones, dim = 1, shape = (data.shape[0], self._conditions))
    def _coeffs[N: int](self, batch: N) -> Matrix[N, C]:
        # sample block indices for each (batch x size)
        sample = randint(0, len(self.__index), (batch, self.__picks))
        # one-hot encode the block indices and stack into new dimension
        coeffs = sums(
            tuple(
                one_hot(sample[..., Index.at(i, self.__picks)], self.__chunks)
                for i in Index.range(self.__picks)
            )
        )
        # sum and clamp the one-hot encodings to do a union of the samples (if same block was sampled twice)
        # then map the one-hot encodings to the original block indices, so each one-hot encoding component is
        # repeated enough times for each of the original data indices
        # e.g.
        #  index = [(0, 3), (6, 9), (12, 14)]
        #  size = 2
        #  batch = 3
        #  sample = [
        #   [0, 1],
        #   [1, 2],
        #   [2, 0]
        #  ]
        #
        #  one-hot encode
        #  [
        #   [[1, 0, 0], [0, 1, 0]],
        #   [[0, 1, 0], [0, 0, 1]],
        #   [[0, 0, 1], [1, 0, 0]]
        #  ]
        #  sum and clamp (union)
        #  [
        #   [1, 1, 0],
        #   [0, 1, 1],
        #   [1, 0, 1]
        #  ]
        #  expand to original indices by using the map
        #  [
        #   [1, 1, 1, 1, 1, 1, 0, 0, 0],
        #   [0, 0, 0, 1, 1, 1, 1, 1, 1],
        #   [1, 1, 1, 0, 0, 0, 1, 1, 1],
        #  ]
        return coeffs.clamp(max = 1) @ self.__map.t()
    @override
    def condition[N: int](self, data: Matrix[N, S]) -> Matrix[N, C]:
        clone = self._clone(data)
        coeffs = self._coeffs(data.shape[0])
        return clone * coeffs
