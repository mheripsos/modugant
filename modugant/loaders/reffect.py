from abc import ABC
from typing import Any, Optional, Tuple, override

from modugant.matrix.index import Index
from modugant.matrix.matrix import Matrix
from modugant.matrix.ops import arange, one_hot, randn, zeros
from modugant.protocols import Loader


class RandomEffectLoader[S: int, U: int, D: int](Loader[Any, D], ABC):
    '''Random effect loader for GANs.'''

    @staticmethod
    def remap[US: int, GS: int](mapping: Index[GS, US]) -> Matrix[US, GS]:
        '''Transform an Index map to a Matrix map.'''
        mapper = zeros((mapping.cap, mapping.dim))
        mapper[arange(mapping.dim)[..., 0], mapping] = 1
        return mapper
    def __init__(
        self,
        sampled: S,
        dim: D,
        index: Tuple[int, U],
        lr: float = 0.001
    ) -> None:
        '''
        Initialize the random effect loader.

        Args:
            sampled (int): The number of sampled dimensions.
            dim (int): The number of outputs.
            index (Tuple[int, int]): The block index and size.
            lr (float): The learning rate.

        '''
        self._sampled = sampled
        self._outputs = dim
        (start, block) = index
        self._start = start
        self._block = block
        self._lr = lr
    def raw[N: int](self, data: Matrix[N, int]) -> Matrix[N, U]:
        '''Return the raw data as one-hot encoded.'''
        return one_hot(data[..., Index.at(self._start, self._sampled)], self._block)
    @override
    def load[N: int](self, data: Matrix[N, int]) -> Matrix[N, D]:
        return self.raw(data) @ self.encoder
    @override
    def unload[N: int](self, data: Matrix[N, D]) -> Matrix[N, Any]:
        candidate = data @ self.encoder.T
        return candidate.argmax(dim = 1, keepdim = True)
    @property
    def encoder(self) -> Matrix[U, D]:
        '''The encoding matrix, to transform from data to latent space.'''
        ...

class RandomEffectUpdater[G: int, D: int](Loader[Any, D], ABC):
    '''Random effect abstract class to apply updates to encoder.'''

    _lr: float
    _parameter: Matrix[G, D] # the updateable parameter
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

class GroupedEffectLoader[S: int, U: int, G: int, D: int](RandomEffectLoader[S, U, D], RandomEffectUpdater[G, D]):
    '''Random effect loader for GANs.'''

    def __init__(
        self,
        sampled: S,
        dim: D,
        index: Tuple[int, U],
        groups: Index[G, U],
        lr: float = 0.001
    ) -> None:
        '''
        Initialize the random effect loader.

        Args:
            sampled (int): The number of sampled dimensions
            dim (int): The number of outputs.
            index (Tuple[int, U]): The block index and size.
            groups (Index[G, U]): The number of groups and group indices
            lr (float): The learning rate.

        '''
        super().__init__(sampled, dim, index, lr)
        self._grouper = RandomEffectLoader.remap(groups)
        self._parameter = randn((groups.dim, dim), requires_grad = True)
    @property
    @override
    def encoder(self) -> Matrix[U, D]:
        return self._grouper @ self._parameter

class SimpleEffectLoader[S: int, U: int, D: int](RandomEffectLoader[S, U, D], RandomEffectUpdater[U, D]):
    '''Random effect loader for GANs.'''

    def __init__(
        self,
        sampled: S,
        dim: D,
        index: Tuple[int, U],
        lr: float = 0.001
    ) -> None:
        '''
        Initialize the random effect loader.

        Args:
            sampled (int): The number of sampled dimensions
            dim (int): The number of outputs.
            index (Tuple[int, int]): The block index and size.
            lr (float): The learning rate.

        '''
        super().__init__(sampled, dim, index, lr)
        self._parameter = randn((self._block, dim), requires_grad = True)
    @property
    @override
    def encoder(self) -> Matrix[U, D]:
        return self._parameter

class NestedEffectLoader[S: int, U: int, G: int, D: int](GroupedEffectLoader[S, U, G, D]):
    '''Nested effect loader for GANs.'''

    def __init__(
        self,
        child: RandomEffectLoader[S, U, D],
        groups: Index[G, U],
        lr: Optional[float] = None
    ) -> None:
        '''
        Initialize the nested effect loader.

        Args:
            child (EffectLoader[U, D]): The child loader.
            groups (Index[G, U]): The number of groups and group indices
            lr (Optional[float]): The learning rate.

        '''
        super().__init__(
            child._sampled,
            child._outputs,
            (child._start, child._block),
            groups,
            lr or child._lr
        )
        if lr is None:
            lr = child._lr
        self._outputs = child.outputs
        self.__child = child
        self._grouper = RandomEffectLoader.remap(groups)
        self._parameter = randn((groups.dim, child._outputs), requires_grad = True)
        self._lr = lr
    @property
    @override
    def encoder(self) -> Matrix[U, D]:
        return (self._grouper @ self._parameter) + self.__child.encoder

