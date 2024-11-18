from typing import Sequence, override

from modugant.matrix import Matrix
from modugant.matrix.index import Index
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
        sizes = [loader.outputs for loader in loaders]
        assert sum(sizes) == dim
        self._outputs = dim
        self.__loaders = loaders
        self.__backmap = [
            sum(sizes[:i])
            for i in range(len(sizes))
        ]
    @override
    def load[N: int](self, data: Matrix[N, int]) -> Matrix[N, D]:
        loaded = tuple(
            loader.load(data)
            for loader in self.__loaders
        )
        return cat(loaded, dim = 1, shape = (data.shape[0], self._outputs))
    @override
    def unload[N: int](self, data: Matrix[N, D]) -> Matrix[N, int]:
        unloaded = tuple(
            self.__loaders[i].unload(
                data[
                    ...,
                    Index.slice(self.__backmap[i], self.__loaders[i].outputs, self._outputs)
                ]
            )
            for i in range(len(self.__loaders))
        )
        return cat(unloaded, dim = 1, shape = (data.shape[0], self._outputs))
