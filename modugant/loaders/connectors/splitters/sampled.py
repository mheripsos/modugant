from typing import Sequence, Tuple, override

from modugant.loaders.connectors.splitters.protocol import Splitter
from modugant.matrix.dim import Dim, Zero
from modugant.matrix.index import Index
from modugant.matrix.matrix import Matrix
from modugant.matrix.ops import cat, one_hot, ones, randint, sums, zeros


class SampledConditioner[S: int, C: int, P: int](Splitter[S, C, Zero]):
    '''
    Sampled Conditioner for Transformer composition.

    Type Parameters:
        S: The sampled dimensionality
        C: The dimensionality of the condition matrix
        P: The number of blocks to sample

    '''

    def __init__(
        self,
        conditions: C,
        samples: S,
        index: Sequence[Tuple[int, int]],
        picks: P,
    ) -> None:
        '''
        Initialize the SampledConditioner.

        Args:
            conditions: the condition dimensionality.
            samples: the sample dimensionality.
            index: the start and size block indices.
            picks: the number of blocks to sample.

        '''
        assert sum([size for (_, size) in index]) == conditions
        assert picks <= len(index) and picks > 0
        self._samples = samples
        self._conditions = conditions
        self._outputs = Dim.zero()
        self.__picks = picks
        self.__index = [
            Index.slice(start, size, samples)
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
    def prepare[N: int](self, data: Matrix[N, S]) -> Matrix[N, Zero]:
        return zeros((data.shape[0], Dim.zero()))
    @override
    def condition[N: int](self, data: Matrix[N, S]) -> Matrix[N, C]:
        clone = self._clone(data)
        coeffs = self._coeffs(data.shape[0])
        return clone * coeffs
