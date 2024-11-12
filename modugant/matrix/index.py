from typing import List, cast, override

from torch import ones, randperm

from .dim import Dim, One


class Index[S: int](List[int]):
    '''An index type.'''

    @staticmethod
    def load[SS: int](data: List[int], size: SS) -> 'Index[SS]':
        '''Load an index from a list.'''
        assert len(data) == size, f'Index {data} is not of size {size}'
        return cast(Index[SS], data)
    @staticmethod
    def slice[SS: int](offset: int, size: SS) -> 'Index[SS]':
        '''Create an index of a certain size with an offset.'''
        return Index.load(list(range(offset, offset + size)), size)
    @staticmethod
    def wrap[N: int](data: 'Index[N]', max: int) -> 'Index[N]':
        '''Wrap the index around the maximum.'''
        size = len(data)
        return Index.load([i % max for i in data], cast(N, size))
    @staticmethod
    def at(index: int) -> 'Index[One]':
        '''Create an index of size one.'''
        return Index.load([index], Dim.one())
    @staticmethod
    def ones[SS: int](size: SS) -> 'Index[SS]':
        '''Create an index of ones.'''
        return Index.load([1] * size, size)
    @staticmethod
    def sample[SS: int](size: SS, pool: int, replacement: bool = True) -> 'Index[SS]':
        '''Create a random sample of the index.'''
        unif = ones((Dim.one(), pool))
        index = cast(List[int], unif.multinomial(size, replacement = replacement)[0].long().tolist())
        return Index.load(index, size)
    @staticmethod
    def partition(size: int, pool: int) -> 'List[Index[int]]':
        '''Create a partitioned index.'''
        assert pool >= size, f'Cannot partition {size} into {pool} parts'
        splits: List[int] = randperm(pool)[:(size - 1)].sort().values.tolist()
        return [
            Index.load(list(range(0, splits[0])), splits[0]),
            *[
                Index.load(list(range(splits[i], splits[i + 1])), splits[i + 1] - splits[i])
                for i in range(size - 1)
            ]
        ]
    @override
    def __len__(self) -> S:
        return cast(S, super().__len__())
