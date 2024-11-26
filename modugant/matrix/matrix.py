from types import EllipsisType
from typing import Any, Literal, Optional, Sequence, Tuple, Union, cast, overload, override

from torch import Tensor

from .dim import Dim, One
from .index import Index, Vector

type Operand[R: int, C: int] = (
    'Matrix[R, C]' |
    'Matrix[R, One]' |
    'Matrix[One, C]' |
    'Matrix[One, One]' |
    int |
    float
)

class Matrix[R: int, C: int](Tensor):
    '''Matrix type.'''

    @staticmethod
    def __cast[RS: int, CS: int](data: Tensor, shape: Tuple[RS, CS]) -> 'Matrix[RS, CS]':
        return cast(Matrix[RS, CS], data)
    @overload
    @staticmethod
    def __reducer(
        data: Tensor,
        shape: Tuple[R, C],
        keepdim: Literal[True],
        dim: Literal[0]
    ) -> 'Matrix[One, C]': ...
    @overload
    @staticmethod
    def __reducer(
        data: Tensor,
        shape: Tuple[R, C],
        keepdim: Literal[True],
        dim: Literal[1]
    ) -> 'Matrix[R, One]': ...
    @overload
    @staticmethod
    def __reducer(
        data: Tensor,
        shape: Tuple[R, C],
        keepdim: Literal[True],
        dim: None
    ) -> 'Matrix[One, One]': ...
    @overload
    @staticmethod
    def __reducer(
        data: Tensor,
        shape: Tuple[R, C],
        *args: Any,
        **kwargs: Any
    ) -> Tensor: ...
    @staticmethod
    def __reducer(
        data: Tensor,
        shape: Tuple[R, C],
        keepdim: bool = False,
        dim: Optional[int] = None
    ) -> 'Matrix[One, C] | Matrix[R, One] | Matrix[One, One] | Tensor':
        if keepdim:
            if dim is None:
                return Matrix(data, (Dim.one(), Dim.one()))
            elif dim == 0:
                return Matrix(data, (Dim.one(), shape[1]))
            else:
                return Matrix(data, (shape[0], Dim.one()))
        else:
            return data
    @staticmethod
    def cell(value: float) -> 'Matrix[One, One]':
        '''Create a cell matrix.'''
        return Matrix(Vector([Vector([value], Dim.one())], Dim.one()))
    @staticmethod
    def row[CS: int](values: Vector[CS, float]) -> 'Matrix[One, CS]':
        '''Create a row matrix.'''
        return Matrix(Vector([values], Dim.one()))
    @staticmethod
    def col[RS: int](values: Vector[RS, float]) -> 'Matrix[RS, One]':
        '''Create a column matrix.'''
        transpose = Vector([Vector([value], Dim.one()) for value in values], values.dim)
        return Matrix(transpose)
    @overload
    def __new__(cls, data: Vector[R, Vector[C, float]]) -> 'Matrix[R, C]': ...
    @overload
    def __new__(cls, data: Tensor, shape: Tuple[R, C]) -> 'Matrix[R, C]': ...
    def __new__(
        cls,
        data: Union[Vector[R, Vector[C, float]], Tensor],
        shape: Optional[Tuple[R, C]] = None
    ) -> 'Matrix[R, C]':
        '''Create a new matrix.'''
        if isinstance(data, Tensor):
            return super().__new__(cls, data.clone().detach())
        else:
            return super().__new__(cls, data)
    @overload
    def __init__(self, data: Vector[R, Vector[C, float]]) -> None: ...
    @overload
    def __init__(self, data: Tensor, shape: Tuple[R, C]) -> None: ...
    def __init__(
            self,
            data: Union[Vector[R, Vector[C, float]], Tensor],
            shape: Optional[Tuple[R, C]] = None
        ) -> None:
        '''Initialize the matrix.'''
        super().__init__(data)
        if isinstance(data, Tensor):
            assert data.shape == shape, f'Data {data} is not of shape {shape}'
            self._shape = shape
        else:
            self._shape = (data.dim, data[0].dim)
    @override
    def __add__(self, other: 'Operand[R, C]') -> 'Matrix[R, C]':
        return Matrix(super().__add__(other), self.shape)
    @override
    def __radd__(self, other: 'Operand[R, C]') -> 'Matrix[R, C]':
        return Matrix(super().__radd__(other), self.shape)
    @override
    def __sub__(self, other: 'Operand[R, C]') -> 'Matrix[R, C]':
        return Matrix(super().__sub__(other), self.shape)
    @override
    def __rsub__(self, other: 'Operand[R, C]') -> 'Matrix[R, C]':
        return Matrix(super().__rsub__(other), self.shape)
    @override
    def __mul__(self, other: 'Operand[R, C]') -> 'Matrix[R, C]':
        return Matrix(super().__mul__(other), self.shape)
    @override
    def __rmul__(self, other: 'Operand[R, C]') -> 'Matrix[R, C]':
        return Matrix(super().__rmul__(other), self.shape)
    @override
    def __matmul__[X: int](self, other: 'Matrix[C, X]') -> 'Matrix[R, X]':
        return Matrix(super().__matmul__(other), (self.shape[0], other.shape[1]))
    @override
    def __rmatmul__[X: int](self, other: 'Matrix[X, R]') -> 'Matrix[X, C]':
        return Matrix(super().__rmatmul__(other), (other.shape[0], self.shape[1]))
    @override
    def __truediv__(self, other: 'Operand[R, C]') -> 'Matrix[R, C]':
        return Matrix(super().__truediv__(other), self.shape)
    @override
    def __rtruediv__(self, other: 'Operand[R, C]') -> 'Matrix[R, C]':
        return Matrix(super().__rtruediv__(other), self.shape)
    @override
    def __pow__(self, other: 'Operand[R, C]') -> 'Matrix[R, C]':
        return Matrix(super().__pow__(other), self.shape)
    @override
    def __rpow__(self, other: 'Operand[R, C]') -> 'Matrix[R, C]':
        return Matrix(super().__rpow__(other), self.shape)
    @override
    def __neg__(self) -> 'Matrix[R, C]':
        return Matrix(super().__neg__(), self.shape)
    @overload
    def __getitem__(self, indices: Tuple[EllipsisType, EllipsisType]) -> 'Matrix[R, C]': ...
    @overload
    def __getitem__[CS: int](self, indices: Tuple[EllipsisType, Index[CS, C]]) -> 'Matrix[R, CS]': ...
    @overload
    def __getitem__[RS: int](self, indices: Tuple[Index[RS, R], EllipsisType]) -> 'Matrix[RS, C]': ...
    @overload
    def __getitem__[RS: int, CS: int](self, indices: Tuple[Index[RS, R], Index[CS, C]]) -> 'Matrix[RS, CS]': ...
    @overload
    def __getitem__(
        self,
        indices: Tuple[EllipsisType | Sequence[int] | int, EllipsisType | Sequence[int] | int]
    ) -> Tensor: ...
    @override
    def __getitem__(self, *args: Any, **kwargs: Any) -> Tensor:
        return Matrix.__cast(super().__getitem__(*args, **kwargs), self.shape)
    @overload
    def argmin(self, dim: Literal[0], keepdim: Literal[True]) -> 'Matrix[One, C]': ...
    @overload
    def argmin(self, dim: Literal[1], keepdim: Literal[True]) -> 'Matrix[R, One]': ...
    @overload
    def argmin(self, dim: None, keepdim: Literal[True]) -> 'Matrix[One, One]': ...
    @overload
    def argmin(self, dim: int, keepdim: bool = False) -> 'Tensor': ...
    @override
    def argmin(
        self,
        dim: Optional[int] = None,
        keepdim: bool = False
    ) -> 'Matrix[One, C] | Matrix[R, One] | Matrix[One, One] | Tensor':
        return Matrix.__reducer(
            super().argmin(dim = dim, keepdim = keepdim),
            self.shape,
            keepdim,
            dim
        )
    @overload
    def argmax(self, dim: Literal[0], keepdim: Literal[True]) -> 'Matrix[One, C]': ...
    @overload
    def argmax(self, dim: Literal[1], keepdim: Literal[True]) -> 'Matrix[R, One]': ...
    @overload
    def argmax(self, dim: None, keepdim: Literal[True]) -> 'Matrix[One, One]': ...
    @overload
    def argmax(self, dim: int, keepdim: bool = False) -> 'Tensor': ...
    @override
    def argmax(
        self,
        dim: Optional[int] = None,
        keepdim: bool = False
    ) -> 'Matrix[One, C] | Matrix[R, One] | Matrix[One, One] | Tensor':
        return Matrix.__reducer(
            super().argmax(dim = dim, keepdim = keepdim),
            self.shape,
            keepdim,
            dim
        )
    @overload
    def sum(self, dim: Literal[0], keepdim: Literal[True], *args: Any, **kwargs: Any) -> 'Matrix[One, C]': ...
    @overload
    def sum(self, dim: Literal[1], keepdim: Literal[True], *args: Any, **kwargs: Any) -> 'Matrix[R, One]': ...
    @overload
    def sum(self, dim: None, keepdim: Literal[True], *args: Any, **kwargs: Any) -> 'Matrix[One, One]': ...
    @overload
    def sum(self, *args: Any, **kwargs: Any) -> Tensor: ...
    @override
    def sum( # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        dim: Optional[int] = None,
        keepdim: bool = False,
        *args: Any,
        **kwargs: Any
    ) -> 'Matrix[One, C] | Matrix[R, One] | Matrix[One, One] | Tensor':
        return Matrix.__reducer(
            super().sum(dim = dim, keepdim = keepdim, *args, **kwargs),
            self.shape,
            keepdim,
            dim
        )
    @overload
    def mean(self, dim: Literal[0], keepdim: Literal[True], *args: Any, **kwargs: Any) -> 'Matrix[One, C]': ...
    @overload
    def mean(self, dim: Literal[1], keepdim: Literal[True], *args: Any, **kwargs: Any) -> 'Matrix[R, One]': ...
    @overload
    def mean(self, dim: None, keepdim: Literal[True], *args: Any, **kwargs: Any) -> 'Matrix[One, One]': ...
    @overload
    def mean(self, *args: Any, **kwargs: Any) -> Tensor: ...
    @override
    def mean( # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        dim: Optional[int] = None,
        keepdim: bool = False,
        *args: Any,
        **kwargs: Any
    ) -> 'Matrix[One, C] | Matrix[R, One] | Matrix[One, One] | Tensor':
        return Matrix.__reducer(
            super().mean(dim = dim, keepdim = keepdim, *args, **kwargs),
            self.shape,
            keepdim,
            dim
        )
    @overload
    def std(self, dim: Literal[0], keepdim: Literal[True] = True, *args: Any, **kwargs: Any) -> 'Matrix[One, C]': ...
    @overload
    def std(self, dim: Literal[1], keepdim: Literal[True] = True, *args: Any, **kwargs: Any) -> 'Matrix[R, One]': ...
    @overload
    def std(self, dim: None, keepdim: Literal[True] = True, *args: Any, **kwargs: Any) -> 'Matrix[One, One]': ...
    @overload
    def std(self, *args: Any, **kwargs: Any) -> Tensor: ...
    @override
    def std( # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        dim: Optional[int] = None,
        keepdim: bool = False,
        *args: Any,
        **kwargs: Any
    ) -> 'Matrix[One, C] | Matrix[R, One] | Matrix[One, One] | Tensor':
        return Matrix.__reducer(
            super().std(dim = dim, keepdim = keepdim, *args, **kwargs),
            self.shape,
            keepdim,
            dim
        )
    @overload
    def var(self, dim: Literal[0], keepdim: Literal[True] = True, *args: Any, **kwargs: Any) -> 'Matrix[One, C]': ...
    @overload
    def var(self, dim: Literal[1], keepdim: Literal[True] = True, *args: Any, **kwargs: Any) -> 'Matrix[R, One]': ...
    @overload
    def var(self, dim: None, keepdim: Literal[True] = True, *args: Any, **kwargs: Any) -> 'Matrix[One, One]': ...
    @overload
    def var(self, *args: Any, **kwargs: Any) -> Tensor: ...
    @override
    def var( # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        dim: Optional[int] = None,
        keepdim: bool = False,
        *args: Any,
        **kwargs: Any
    ) -> 'Matrix[One, C] | Matrix[R, One] | Matrix[One, One] | Tensor':
        return Matrix.__reducer(
            super().var(dim = dim, keepdim = keepdim, *args, **kwargs),
            self.shape,
            keepdim,
            dim
        )
    @override
    def round(self, **kwargs: Any) -> 'Matrix[R, C]':
        return Matrix(super().round(**kwargs), self.shape)
    @override
    def clamp(self, *args: Any, **kwargs: Any) -> 'Matrix[R, C]':
        return Matrix(super().clamp(*args, **kwargs), self.shape)
    @override
    def log(self) -> 'Matrix[R, C]':
        return Matrix(super().log(), self.shape)
    @override
    def exp(self) -> 'Matrix[R, C]':
        return Matrix(super().exp(), self.shape)
    @override
    def square(self) -> 'Matrix[R, C]':
        return Matrix(super().square(), self.shape)
    @override
    def sqrt(self) -> 'Matrix[R, C]':
        return Matrix(super().sqrt(), self.shape)
    @override
    def sin(self) -> 'Matrix[R, C]':
        return Matrix(super().sin(), self.shape)
    @override
    def cos(self) -> 'Matrix[R, C]':
        return Matrix(super().cos(), self.shape)
    @overload
    def split[D: int](self, split_size: Vector[D, int], dim: Literal[0]) -> Vector[D, 'Matrix[int, C]']: ...
    @overload
    def split[D: int](self, split_size: Vector[D, int], dim: Literal[1]) -> Vector[D, 'Matrix[R, int]']: ...
    @overload
    def split(self, split_size: Tuple[int, ...], dim: int = 0) -> Tuple[Tensor, ...]: ...
    @override
    def split[D: int](
        self,
        split_size: Union[Vector[D, int], Tuple[int, ...]],
        dim: int = 0
    ) -> 'Vector[D, Matrix[int, C] | Matrix[R, int]] | Tuple[Tensor, ...]':
        splits = cast(Tuple[Tensor, ...], super().split(split_size = split_size, dim = dim))
        if isinstance(split_size, Vector):
            if dim == 0:
                return Vector(
                    (
                        Matrix(split, (split.shape[0], self.shape[1]))
                        for split in splits
                    ),
                    split_size.dim # pyright: ignore[reportUnknownArgumentType] should not be a warning
                )
            else:
                return Vector(
                    (
                        Matrix(split, (self.shape[0], split.shape[1]))
                        for split in splits
                    ),
                    split_size.dim # pyright: ignore[reportUnknownArgumentType] should not be a warning
                )
        else:
            return splits
    @overload
    def repeat(self, repeats: Tuple[int, Literal[1]]) -> 'Matrix[int, C]': ...
    @overload
    def repeat(self, repeats: Tuple[Literal[1], int]) -> 'Matrix[R, int]': ...
    @overload
    def repeat(self,  repeats: Tuple[int, int]) -> 'Matrix[Any, Any]': ...
    @override
    def repeat(self, repeats: Tuple[int, int]) -> 'Matrix[Any, Any]': # type: ignore[override]
        return Matrix(super().repeat(repeats), shape = (self.shape[0] * repeats[0], self.shape[1] * repeats[1]))
    @override
    def multinomial[S: int](self, num_samples: S, *args: Any, **kwargs: Any) -> 'Matrix[R, S]':
        return Matrix(super().multinomial(*args, **kwargs), (self.shape[0], num_samples))
    @override
    def softmax(self, *args: Any, **kwargs: Any) -> 'Matrix[R, C]':
        return Matrix(super().softmax(*args, **kwargs), self.shape)
    @override
    def int(self) -> 'Matrix[R, C]':
        return Matrix(super().int(), self.shape)
    @override
    def long(self) -> 'Matrix[R, C]':
        return Matrix(super().long(), self.shape)
    @override
    def float(self) -> 'Matrix[R, C]':
        return Matrix(super().float(), self.shape)
    @override
    def double(self) -> 'Matrix[R, C]':
        return Matrix(super().double(), self.shape)
    @override
    def t(self) -> 'Matrix[C, R]':
        return Matrix(super().t(), (self.shape[1], self.shape[0]))
    @override
    def detach(self) -> 'Matrix[R, C]':
        return Matrix(super().detach(), self.shape)
    @override
    def clone(self, **kwargs: Any) -> 'Matrix[R, C]':
        return Matrix(super().clone(**kwargs), self.shape)
    @override
    def to(self, *args: Any, **kwargs: Any) -> 'Matrix[R, C]':
        return Matrix(super().to(*args, **kwargs), self.shape)
    @property
    @override
    def T(self) -> 'Matrix[C, R]': # type: ignore
        '''The transpose of the matrix.'''
        return cast(Matrix[C, R], super().T)
    @property
    @override
    def shape(self) -> Tuple[R, C]: # type: ignore
        '''The shape of the matrix.'''
        return cast(Tuple[R, C], super().shape)
