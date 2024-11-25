from typing import Any, Sequence, override

from modugant.loaders.connectors.transformers.protocol import Transformer
from modugant.matrix.index import Index
from modugant.matrix.matrix import Matrix
from modugant.matrix.ops import arange, randn, sums, zeros


class RandomEffectTransformer[U: int, P: int, S: int](Transformer[S]):
    '''Random effect transformer for Connector composition. Transforms data into vector of random effects.'''

    @staticmethod
    def remap[US: int, GS: int](mapping: Index[GS, US]) -> Matrix[US, GS]:
        '''Transform an Index map to a Matrix map.'''
        mapper = zeros((mapping.cap, mapping.dim))
        mapper[arange(mapping.dim)[..., 0], mapping] = 1
        return mapper
    def view[N: int](self, data: Matrix[N, Any]) -> Matrix[N, U]:
        '''Get the underlying view of the data.'''
        ...
    def unview[N: int](self, data: Matrix[N, U]) -> Matrix[N, Any]:
        '''Get the underlying view of the data.'''
        ...
    @override
    def load[N: int](self, data: Matrix[N, U]) -> Matrix[N, S]:
        '''Transform underlying data.'''
        return self.view(data) @ self.encoder
    @override
    def unload[N: int](self, data: Matrix[N, S]) -> Matrix[N, Any]:
        '''Revert to underlying data.'''
        return self.unview(data @ self.encoder.T)
    @property
    def width(self) -> U:
        '''Get the width.'''
        ...
    @property
    def encoder(self) -> Matrix[U, S]:
        '''Get the encoder.'''
        ...

class RandomEffectViewer[U: int, P: int, S: int](Transformer[U]):
    '''Converter from RandomEffectTransformer to RandomEffectMapper.'''

    def __init__(
        self,
        transformer: RandomEffectTransformer[U, P, S]
    ) -> None:
        '''
        Initialize the random effect viewer.

        Args:
            transformer (RandomEffectTransformer): The transformer to view.

        '''
        self._samples = transformer.width
        self._transformer = transformer
    @override
    def load[N: int](self, data: Matrix[N, Any]) -> Matrix[N, U]:
        '''Transform underlying data.'''
        return self._transformer.view(data)
    @override
    def unload[N: int](self, data: Matrix[N, U]) -> Matrix[N, Any]:
        '''Revert to underlying data.'''
        return self._transformer.unview(data)

class RandomEffectMapper[U: int, P: int, S: int](RandomEffectTransformer[U, P, S]):
    '''Random effect abstract class to extract data vector.'''

    _parameter: Matrix[P, S]
    def __init__(
        self,
        viewer: Transformer[U],
        dim: S,
        lr: float = 0.001
    ) -> None:
        '''
        Initialize the random effect loader.

        Args:
            viewer (Transformer): The transformer to load the underlying one-hot data.
            dim (int): The number of outputs.
            lr (float): The learning rate.

        '''
        self._samples = dim
        self._width = viewer.samples
        self._viewer = viewer
        self._lr = lr
    @override
    def view[N: int](self, data: Matrix[N, Any]) -> Matrix[N, U]:
        '''Get the underlying view of the data.'''
        return self._viewer.load(data)
    @override
    def unview[N: int](self, data: Matrix[N, U]) -> Matrix[N, Any]:
        '''Get the underlying view of the data.'''
        return self._viewer.unload(data)
    @override
    def update(self) -> None:
        '''Update the encoder.'''
        if self._parameter.grad is not None:
            gradient = Matrix(self._parameter.grad, self._parameter.shape)
            self._parameter = self._parameter - self._lr * gradient
            norm = (self._parameter * self._parameter).sum(dim = 1, keepdim = True).sqrt()
            self._parameter = self._parameter / norm
            if self._parameter.grad is not None:
                _ = self._parameter.grad.zero_()
        self._viewer.update()

class SimpleEffectTransformer[U: int, D: int](RandomEffectMapper[U, U, D]):
    '''Random effect Transformer for Connector composition.'''

    def __init__(
        self,
        viewer: Transformer[U],
        dim: D,
        lr: float = 0.001
    ) -> None:
        '''
        Initialize the simple effect transformer.

        Args:
            viewer (Transformer): The transformer to load the underlying one-hot data.
            dim (int): The number of outputs.
            lr (float): The learning rate.

        '''
        super().__init__(viewer, dim, lr)
        self._parameter = randn((viewer.samples, dim), requires_grad = True)
    @property
    @override
    def encoder(self) -> Matrix[U, D]:
        return self._parameter

class GroupedEffectTransformer[U: int, G: int, D: int](RandomEffectMapper[U, G, D]):
    '''
    Random effect Transformer for Connector composition.

    Converts nominal data to random effects arrays, grouped by some mapping.
    '''

    def __init__(
        self,
        viewer: Transformer[U],
        dim: D,
        groups: Index[G, U],
        lr: float = 0.001
    ) -> None:
        '''
        Initialize the grouped effect transformer.

        Args:
            viewer (Transformer): The transformer to load the underlying one-hot data.
            dim (int): The number of outputs.
            groups (Index): The grouping index.
            lr (float): The learning rate.

        '''
        super().__init__(viewer, dim, lr)
        self._grouper = RandomEffectTransformer.remap(groups)
        self._parameter = randn((groups.dim, dim), requires_grad = True)
    @property
    @override
    def encoder(self) -> Matrix[U, D]:
        return self._grouper @ self._parameter

class BatchedEffectTransformer[U: int, C: int, G: int, D: int](RandomEffectTransformer[U, G, D]):
    '''Nested effect transformer batch for NestedEffectTransformer composition.'''

    def __init__(
        self,
        root: RandomEffectTransformer[U, C, D],
        groups: Sequence[Index[G, U]],
        lr: float = 0.001
    ) -> None:
        '''
        Initialize the batched effect transformer.

        Args:
            root (RandomEffectTransformer): The root transformer.
            groups (List[Index]): The group index assignments.
            lr (float): The learning rate.

        '''
        self._samples = root.samples
        self._width = root.width
        self.__root = root
        self.__transformers = [
            GroupedEffectTransformer(RandomEffectViewer(root), root.samples, group, lr)
            for group in groups
        ]
    def __getitem__(self, index: int) -> RandomEffectTransformer[U, G, D]:
        '''Get the transformer at the index.'''
        return self.__transformers[index]
    @override
    def view[N: int](self, data: Matrix[N, Any]) -> Matrix[N, U]:
        '''Get the underlying view of the data.'''
        return self.__root.view(data)
    @override
    def unview[N: int](self, data: Matrix[N, U]) -> Matrix[N, Any]:
        '''Get the underlying view of the data.'''
        return self.__root.unview(data)
    @override
    def update(self) -> None:
        '''Update the encoder.'''
        for transformer in self.__transformers:
            transformer.update()
        self.__root.update()
    @property
    @override
    def width(self) -> U:
        '''Get the width.'''
        return self._width
    @property
    @override
    def encoder(self) -> Matrix[U, D]:
        '''Get the encoder.'''
        return self.__root.encoder + sums(tuple(transformer.encoder for transformer in self.__transformers))
