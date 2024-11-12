from typing import Sequence, override

from modugant.matrix import Matrix
from modugant.matrix.ops import cat
from modugant.protocols import Loader


class JointLoader[D: int](Loader[D]):
    '''Joint loader for GANs.'''

    def __init__(
        self,
        dim: D,
        loaders: Sequence[Loader[int]]
    ) -> None:
        '''
        Initialize the joint loader.

        Args:
            dim (int): The number of inputs and outputs.
            loaders (Sequence[Loader]): The loaders.

        '''
        assert sum([loader.outputs for loader in loaders]) == dim
        self._outputs = dim
        self.__loaders = loaders
    @override
    def load[N: int](self, data: Matrix[N, int]) -> Matrix[N, D]:
        loaded = tuple(
            loader.load(data)
            for loader in self.__loaders
        )
        return cat(loaded, dim = 1, shape = (data.shape[0], self._outputs))
