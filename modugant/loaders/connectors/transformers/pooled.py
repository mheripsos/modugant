from typing import Any, Sequence, override

from modugant.loaders.connectors.transformers.protocol import Transformer
from modugant.matrix.index import Index
from modugant.matrix.matrix import Matrix
from modugant.matrix.ops import cat


class PooledTransformer[S: int](Transformer[S]):
    '''Pooled Transformer for Connector composition. Converts raw data to one-hot encoding.'''

    def __init__(self, dim: S, transformers: Sequence[Transformer[Any]]) -> None:
        '''Initialize the pooled transformer.'''
        sizes = [transformer.samples for transformer in transformers]
        assert(dim == sum(sizes)), f'The sum of transormer sizes {sum(sizes)} does not match the dim {dim}.'
        self._samples = dim
        self._transformers = transformers
        self.__backmap = [
            sum(sizes[:i])
            for i in range(len(transformers))
        ]
    def __getitem__(self, index: int) -> Transformer[int]:
        '''Get the transformer at the index.'''
        return self._transformers[index]
    @override
    def load[N: int](self, data: Matrix[N, Any]) -> Matrix[N, S]:
        '''Transform underlying data.'''
        transformed = tuple(
            transformer.load(data)
            for transformer in self._transformers
        )
        return cat(transformed, dim = 1, shape = (data.shape[0], self._samples))
    @override
    def unload[N: int](self, data: Matrix[N, S]) -> Matrix[N, Any]:
        '''Revert to underlying data.'''
        unloaded = tuple(
            self._transformers[i].unload(
                data[
                    ...,
                    Index.slice(self.__backmap[i], self._transformers[i].samples, self._samples)
                ]
            )
            for i in range(len(self._transformers))
        )
        return cat(unloaded, dim = 1, shape = (data.shape[0], self._samples))
