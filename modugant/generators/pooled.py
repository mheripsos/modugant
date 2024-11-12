from typing import Self, Sequence, override

from modugant.device import Device
from modugant.matrix import Matrix
from modugant.matrix.dim import One
from modugant.matrix.index import Index
from modugant.matrix.ops import cat
from modugant.protocols import Generator


class PooledGenerator[C: int, L: int, G: int](Generator[C, L, G]):
    '''
    Combine a Generator with a non-learning pool of Generators.

    Type parameters:
        C: The number of conditions.
        L: The number of latent inputs.
        G: The number of generated outputs.

    '''

    def __init__(
        self,
        conditions: C,
        latents: L,
        intermediates: G,
        main: Generator[C, L, G],
        pool: Sequence[Generator[C, L, G]],
    ) -> None:
        '''
        Initialize the generator model.

        Args:
            conditions (C: int): The number of condition nodes.
            latents (L: int): The number of input nodes.
            intermediates (G: int): The number of output nodes.
            main (Generator): The main generator.
            pool (Sequence[Generator]): The pool of generators.

        '''
        self._conditions = conditions
        self._latents = latents
        self._intermediates = intermediates
        self._main = main
        self._pool = pool
    @override
    def sample[N: int](self, condition: Matrix[N, C]) -> Matrix[N, G]:
        ## Use randperm to sample splits along the first dimension of the condtion
        ## Then sample from the main generator and the pool generators in those splits
        k = min(len(self._pool), condition.shape[0])
        splits = Index.partition(condition.shape[0], k + 1)
        main = self._main.sample(condition[splits[0], ...])
        pool = cat(
            tuple(
                self._pool[i].sample(condition[splits[i + 1], ...])
                for i in range(k)
            ),
            dim = 0,
            shape = (condition.shape[0] - len(splits[0]), self._intermediates)
        )
        return cat((main, pool), dim = 0, shape = (condition.shape[0], self._intermediates))
    @override
    def update(self, loss: Matrix[One, One]) -> None:
        return self._main.update(loss)
    @override
    def reset(self) -> None:
        self._main.reset()
        for generator in self._pool:
            generator.reset()
    @override
    def restart(self) -> None:
        self._main.restart()
        for generator in self._pool:
            generator.restart()
    @override
    def move(self, device: Device) -> Self:
        _ = self._main.move(device)
        for generator in self._pool:
            _ = generator.move(device)
        return self
    @property
    @override
    def rate(self) -> float:
        return self._main.rate

