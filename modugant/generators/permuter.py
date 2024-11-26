from typing import Self, override

from torch import Tensor

from modugant.device import Device, check_device
from modugant.generators.protocol import Generator
from modugant.matrix import Index, Matrix
from modugant.matrix.dim import One, Zero
from modugant.matrix.index import Vector
from modugant.matrix.ops import cat


class PermutingGenerator[C: int, G: int, F: int](Generator[C, Zero, G]):
    '''A generator that permutes the data without learning.'''

    def __init__(
        self,
        conditions: C,
        intermediates: G,
        data: Tensor,
        index: Index[G, int],
        folds: Vector[F, int]
    ) -> None:
        '''
        Initialize the generator model.

        Args:
            conditions (C: int): The number of condition nodes.
            intermediates (G: int): The number of output nodes.
            data (Tensor): The data.
            index (Optional[List[int]]): The column indices to select from the data.
            folds (Optional[List[int]]): The fold sizes to keep intact during permutation

        '''
        assert len(index) == intermediates
        assert sum(folds) == intermediates
        self._conditions = conditions
        self._intermediates = intermediates
        matrix = Matrix(data, shape = (data.shape[0], data.shape[1]))
        self._splits = matrix[..., index].split(folds, dim = 1)
        self._dim = data.shape[0]
        self._device = data.device
    @override
    def sample[N: int](self, condition: Matrix[N, C]) -> Matrix[N, G]:
        index = Index.sample(condition.shape[0], self._dim)
        perm = Index.randperm(self._splits.dim)
        permed = cat(
            tuple(self._splits[i][index, ...] for i in perm),
            dim = 1,
            shape = (condition.shape[0], self._intermediates)
        )
        return permed
    @override
    def update(self, loss: Matrix[One, One]) -> None:
        pass
    @override
    def reset(self) -> None:
        pass
    @override
    def restart(self) -> None:
        pass
    @override
    def move(self, device: Device) -> Self:
        device = check_device(device)
        self._splits = Vector((split.to(device) for split in self._splits), self._splits.dim)
        return self
    @property
    @override
    def rate(self) -> float:
        return 0.0
