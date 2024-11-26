from typing import Any, Sequence, Tuple, override

from modugant.loaders.connectors.transformers.protocol import Transformer
from modugant.matrix.dim import One
from modugant.matrix.index import Index
from modugant.matrix.matrix import Matrix
from modugant.matrix.ops import arange, means, randn, sums, zeros

type EffectGroup[G: int, U: int] = Tuple[Index[G, U], Sequence[EffectGroup[G, U]]]

class RandomEffectTransformer[U: int, G: int, S: int](Transformer[S]):
    '''Random effect transformer for Connector composition. Transforms data into vector of random effects.'''

    def __init__(
        self,
        viewer: Transformer[U],
        dim: S,
        group: EffectGroup[G, U],
        lr: float = 0.001
    ) -> None:
        '''
        Initialize the random effect loader.

        Args:
            viewer (Transformer): The transformer to load the underlying one-hot data.
            dim (int): The number of outputs.
            group (Tuple[Index, List[EffectGroup]]): The grouping index.
            lr (float): The learning rate.

        '''
        self._samples = dim
        self._width = viewer.samples
        self._viewer = viewer
        self._lr = lr
        (mapping, regroup) = group
        self._grouper = zeros((mapping.cap, mapping.dim))
        self._grouper[arange(mapping.dim)[..., 0], mapping] = 1
        self._parameter = randn((mapping.dim, dim), requires_grad = True)
        self._groups = [RandomEffectTransformer(viewer, dim, group, lr) for group in regroup]
    @override
    def load[N: int](self, data: Matrix[N, Any]) -> Matrix[N, S]:
        return self._viewer.load(data) @ self.encoder
    @override
    def unload[N: int](self, data: Matrix[N, S]) -> Matrix[N, Any]:
        return self._viewer.unload(data @ self.encoder.T)
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
        for group in self._groups:
            group.update()
        self._viewer.update()
    def penalty(self) -> Matrix[One, One]:
        '''Get the penalty based on distance from standard normal to 4th moment.'''
        mean = self._parameter.mean(dim = 0, keepdim = True)
        std = self._parameter.std(dim = 0, keepdim = True)
        divergence = (1 / std).log() + (std.square() + mean.square()) / 2 - 0.5
        skewness = (((self._parameter - mean) / std) ** 3).mean(dim = 0, keepdim = True)
        kurtosis = (((self._parameter - mean) / std) ** 4).mean(dim = 0, keepdim = True) - 3
        return (
            divergence.mean(dim = 1, keepdim = True) +
            skewness.mean(dim = 1, keepdim = True) * 0.5 +
            kurtosis.mean(dim = 1, keepdim = True) * 0.5 +
            means(tuple(loader.penalty() for loader in self._groups))
        )
    @property
    def encoder(self) -> Matrix[U, S]:
        '''Get the encoder.'''
        return self._grouper @ self._parameter + sums(tuple(loader.encoder for loader in self._groups))
