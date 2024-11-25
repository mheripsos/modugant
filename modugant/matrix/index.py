from typing import Any, Iterable, List, Self, Tuple, TypeGuard, cast, overload, override

from torch import ones, randperm

from modugant.matrix.dim import Dim, One, Zero


class Vector[D: int, T](Tuple[T, ...]):
    '''A sized list type.'''

    @staticmethod
    def is_of[DS: int, TS](value: List[TS], dim: DS) -> TypeGuard['Vector[DS, TS]']:
        '''Check if a list is of a certain size.'''
        if isinstance(value, Vector):
            ## this shouldn't be a warning
            return value.dim == dim # pyright: ignore[reportUnknownVariableType]
        else:
            return False
    @staticmethod
    def of[DS: int, TS](value: TS, dim: DS) -> 'Vector[DS, TS]':
        '''Create a list of a single value.'''
        return Vector([value] * dim, dim)
    def __new__(cls, data: Iterable[T], size: D) -> Self:
        '''Create a new sized list.'''
        return super().__new__(cls, data)
    def __init__(self, data: Iterable[T], dim: D):
        '''Initialize the sized list.'''
        super().__init__()
        assert len(self) == dim, 'Data does not match size.'
        self._dim = dim
    @overload
    def __getitem__[N: int](self, key: Index[N, D]) -> 'Vector[N, T]':...
    @overload
    def __getitem__(self, key: int) -> T:...
    @overload
    def __getitem__(self, key: slice) -> Tuple[T, ...]:...
    @override
    def __getitem__(self, key: Any) -> Any: # pyright: ignore[reportIncompatibleMethodOverride]
        '''Get the value at an index.'''
        if isinstance(key, Index):
            return Vector(
                tuple(
                    super().__getitem__(i)
                    for i in key
                ),
                self.dim
            )
        elif isinstance(key, int):
            return super().__getitem__(key)
        else:
            return cast(Tuple[T, ...], super().__getitem__(key))
    def expand[DX: int](self, count: DX) -> 'Vector[DX, Vector[D, T]]':
        '''Repeat the list.'''
        return Vector(
            [
                Vector([value for value in self], self.dim)
                for _ in range(count)
            ],
            count
        )
    @property
    def dim(self) -> D:
        '''The size of the list.'''
        return self._dim

class Index[D: int, C: int](Vector[D, int]):
    '''An index type.'''

    @staticmethod
    def randperm[DS: int](dim: DS) -> 'Index[DS, DS]':
        '''Create a random permutation.'''
        perm = cast(List[int], randperm(dim).tolist())
        return Index(perm, dim, dim)
    @staticmethod
    def slice[DS: int, CS: int](offset: int, size: DS, cap: CS) -> 'Index[DS, CS]':
        '''Create a slice of the index.'''
        return Index(range(offset, offset + size), size, cap)
    @staticmethod
    def range[DS: int](size: DS) -> 'Index[DS, DS]':
        '''Create a range of indices.'''
        return Index(range(size), size, size)
    @staticmethod
    def at[CS: int](index: int, cap: CS) -> 'Index[One, CS]':
        '''Create an index at a single position.'''
        return Index([index], Dim.one(), cap)
    @staticmethod
    def empty[CS: int](cap: CS) -> 'Index[Zero, CS]':
        '''Create an empty index.'''
        return Index([], Dim.zero(), cap)
    @staticmethod
    def sample[DS: int, CS: int](size: DS, cap: CS, replacement: bool = True) -> 'Index[DS, CS]':
        '''Create a random index.'''
        unif = ones((1, cap))
        index = cast(List[int], unif.multinomial(size, replacement = replacement).squeeze().tolist())
        return Index(index, size, cap)
    @staticmethod
    def partition[DS: int, CS: int](count: DS, cap: CS) -> Vector[DS, 'Index[int, CS]']:
        '''Partition an range of indices.'''
        assert cap >= count, 'Not enough indices to partition.'
        splits: List[int] = randperm(cap)[:(count - 1)].sort().values.tolist()
        indices = [
            Index.slice(0, splits[0], cap),
            *[
                Index.slice(splits[i], splits[i + 1] - splits[i], cap)
                for i in range(count - 2)
            ],
            Index.slice(splits[-1], cap - splits[-1], cap)
        ]
        return Vector(indices, count)
    def __new__(cls, data: Iterable[int], dim: D, cap: C) -> Self:
        '''Create a new index.'''
        return super().__new__(cls, data, dim)
    def __init__(self, data: Iterable[int], dim: D, cap: C):
        '''Initialize the index.'''
        super().__init__(data, dim)
        assert all(0 <= i < cap for i in self), 'Index out of bounds.'
        self._cap = cap
    def wrap(self, cap: C) -> 'Index[D, C]':
        '''Wrap the index with a new capacity.'''
        return Index([i % cap for i in self], self.dim, cap)
    @property
    def cap(self) -> C:
        '''The capacity of the index.'''
        return self._cap
