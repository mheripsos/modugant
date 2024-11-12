from typing import Literal, Self, override


class Dim[N: int](int):
    '''A fixed integer as an explicit type.'''

    @staticmethod
    def zero() -> 'Dim[Literal[0]]':
        '''Return the dimension zero.'''
        return Dim[0](0)
    @staticmethod
    def one() -> 'Dim[Literal[1]]':
        '''Return the dimension one.'''
        return Dim[1](1)
    @override
    def __new__(cls, value: N) -> Self:
        return super().__new__(cls, value)

Zero = Dim[Literal[0]]
One = Dim[Literal[1]]
