from types import EllipsisType
from typing import Any, List, Literal, Optional, Sequence, Tuple, Union, cast, overload, override

from torch import Tensor, tensor

from .dim import One
from .index import Index

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
    def load[RS: int, CS: int](data: Tensor, shape: Optional[Tuple[RS, CS]] = None) -> 'Matrix[RS, CS]':
        '''Load a matrix from data.'''
        if shape is None:
            shape = cast(Tuple[RS, CS], data.shape)
        assert data.shape == shape, f'Matrix {data} is not of shape {shape}'
        return cast(Matrix[RS, CS], data)
    @staticmethod
    def cell(value: float) -> 'Matrix[One, One]':
        '''Create a cell matrix.'''
        return cast(Matrix[One, One], tensor([[value]]))
    @override
    def __add__(self, other: 'Operand[R, C]') -> 'Matrix[R, C]':
        return Matrix.load(super().__add__(other), self.shape)
    @override
    def __radd__(self, other: 'Operand[R, C]') -> 'Matrix[R, C]':
        return Matrix.load(super().__radd__(other), self.shape)
    @override
    def __sub__(self, other: 'Operand[R, C]') -> 'Matrix[R, C]':
        return Matrix.load(super().__sub__(other), self.shape)
    @override
    def __rsub__(self, other: 'Operand[R, C]') -> 'Matrix[R, C]':
        return Matrix.load(super().__rsub__(other), self.shape)
    @override
    def __mul__(self, other: 'Operand[R, C]') -> 'Matrix[R, C]':
        return Matrix.load(super().__mul__(other), self.shape)
    @override
    def __rmul__(self, other: 'Operand[R, C]') -> 'Matrix[R, C]':
        return Matrix.load(super().__rmul__(other), self.shape)
    @override
    def __matmul__[X: int](self, other: 'Matrix[C, X]') -> 'Matrix[R, X]':
        return Matrix.load(super().__matmul__(other), (self.shape[0], other.shape[1]))
    @override
    def __rmatmul__[X: int](self, other: 'Matrix[X, R]') -> 'Matrix[X, C]':
        return Matrix.load(super().__rmatmul__(other), (other.shape[0], self.shape[1]))
    @override
    def __truediv__(self, other: 'Operand[R, C]') -> 'Matrix[R, C]':
        return Matrix.load(super().__truediv__(other), self.shape)
    @override
    def __rtruediv__(self, other: 'Operand[R, C]') -> 'Matrix[R, C]':
        return Matrix.load(super().__rtruediv__(other), self.shape)
    @override
    def __pow__(self, other: 'Operand[R, C]') -> 'Matrix[R, C]':
        return Matrix.load(super().__pow__(other), self.shape)
    @override
    def __rpow__(self, other: 'Operand[R, C]') -> 'Matrix[R, C]':
        return Matrix.load(super().__rpow__(other), self.shape)
    @override
    def __neg__(self) -> 'Matrix[R, C]':
        return Matrix.load(super().__neg__(), self.shape)
    @overload
    def __getitem__(self, indices: Tuple[EllipsisType, EllipsisType]) -> 'Matrix[R, C]': ...
    @overload
    def __getitem__[CS: int](self, indices: Tuple[EllipsisType, Index[CS]]) -> 'Matrix[R, CS]': ...
    @overload
    def __getitem__(self, indices: Tuple[EllipsisType, Sequence[int]]) -> 'Matrix[R, Any]': ...
    @overload
    def __getitem__[RS: int](self, indices: Tuple[Index[RS], EllipsisType]) -> 'Matrix[RS, C]': ...
    @overload
    def __getitem__[RS: int, CS: int](self, indices: Tuple[Index[RS], Index[CS]]) -> 'Matrix[RS, CS]': ...
    @overload
    def __getitem__[RS: int](self, indices: Tuple[Index[RS], Sequence[int]]) -> 'Matrix[RS, Any]': ...
    @overload
    def __getitem__(self, indices: Tuple[Sequence[int], EllipsisType]) -> 'Matrix[R, C]': ...
    @overload
    def __getitem__[CS: int](self, indices: Tuple[Sequence[int], Index[CS]]) -> 'Matrix[Any, CS]': ...
    @overload
    def __getitem__(self, indices: Tuple[Sequence[int], Sequence[int]]) -> 'Matrix[Any, Any]': ...
    @overload
    def __getitem__(self, indices: Tuple[EllipsisType, int]) -> Tensor: ...
    @overload
    def __getitem__(self, indices: Tuple[int, EllipsisType]) -> Tensor: ...
    @override
    def __getitem__(self, *args: Any, **kwargs: Any) -> Tensor:
        return cast(Matrix[int, int], super().__getitem__(*args, **kwargs))
    @override
    def round(self, **kwargs: Any) -> 'Matrix[R, C]':
        return cast(Matrix[R, C], super().round(**kwargs))
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
        return super().argmax(dim = dim, keepdim = keepdim)
    @overload
    def sum(self, dim: Literal[0], keepdim: Literal[True]) -> 'Matrix[One, C]': ...
    @overload
    def sum(self, dim: Literal[1], keepdim: Literal[True]) -> 'Matrix[R, One]': ...
    @overload
    def sum(self, dim: None, keepdim: Literal[True]) -> 'Matrix[One, One]': ...
    @overload
    def sum(self, *args: Any, **kwargs: Any) -> Tensor: ...
    @override
    def sum( # type: ignore[override]
        self,
        *args: Any,
        **kwargs: Any
    ) -> 'Matrix[One, C] | Matrix[R, One] | Matrix[One, One] | Tensor':
        return super().sum(*args, **kwargs)
    @overload
    def mean(self, dim: Literal[0], keepdim: Literal[True]) -> 'Matrix[One, C]': ...
    @overload
    def mean(self, dim: Literal[1], keepdim: Literal[True]) -> 'Matrix[R, One]': ...
    @overload
    def mean(self, dim: None, keepdim: Literal[True]) -> 'Matrix[One, One]': ...
    @overload
    def mean(self, *args: Any, **kwargs: Any) -> Tensor: ...
    @override
    def mean( # type: ignore[override]
        self,
        *args: Any,
        **kwargs: Any
    ) -> 'Matrix[One, C] | Matrix[R, One] | Matrix[One, One] | Tensor':
        return super().mean(*args, **kwargs)
    @overload
    def std(self, dim: Literal[0], unbiased: bool = True, keepdim: Literal[True] = True) -> 'Matrix[One, C]': ...
    @overload
    def std(self, dim: Literal[1], unbiased: bool = True, keepdim: Literal[True] = True) -> 'Matrix[R, One]': ...
    @overload
    def std(self, dim: None, unbiased: bool = True, keepdim: Literal[True] = True) -> 'Matrix[One, One]': ...
    @overload
    def std(self, *args: Any, **kwargs: Any) -> Tensor: ...
    @override
    def std( # type: ignore[override]
        self,
        *args: Any,
        **kwargs: Any
    ) -> 'Matrix[One, C] | Matrix[R, One] | Matrix[One, One] | Tensor':
        return super().std(*args, **kwargs)
    @overload
    def var(
        self,
        dim: Literal[0],
        unbiased: bool = True,
        correction: Optional[Union[int, float, bool, complex]] = None,
        keepdim: Literal[True] = True
    ) -> 'Matrix[One, C]': ...
    @overload
    def var(
        self,
        dim: Literal[1],
        unbiased: bool = True,
        correction: Optional[Union[int, float, bool, complex]] = None,
        keepdim: Literal[True] = True
    ) -> 'Matrix[R, One]': ...
    @overload
    def var(
        self,
        dim: None,
        unbiased: bool = True,
        correction: Optional[Union[int, float, bool, complex]] = None,
        keepdim: Literal[True] = True
    ) -> 'Matrix[One, One]': ...
    @overload
    def var(self, *args: Any, **kwargs: Any) -> Tensor: ...
    @override
    def var( # type: ignore[override]
        self,
        *args: Any,
        **kwargs: Any
    ) -> 'Matrix[One, C] | Matrix[R, One] | Matrix[One, One] | Tensor':
        return super().var(*args, **kwargs)
    @override
    def clamp(self, *args: Any, **kwargs: Any) -> 'Matrix[R, C]':
        return Matrix.load(super().clamp(*args, **kwargs), self.shape)
    @override
    def log(self) -> 'Matrix[R, C]':
        return cast(Matrix[R, C], super().log())
    @override
    def sin(self) -> 'Matrix[R, C]':
        return cast(Matrix[R, C], super().sin())
    @override
    def cos(self) -> 'Matrix[R, C]':
        return cast(Matrix[R, C], super().cos())
    @overload
    def split(self, split: List[int], dim: Literal[0]) -> Tuple['Matrix[int, C]', ...]: ...
    @overload
    def split(self, split: List[int], dim: Literal[1]) -> Tuple['Matrix[R, int]', ...]: ...
    @override
    def split(self, *args: Any, **kwargs: Any) -> Tuple[Tensor, ...]:
        return cast(Tuple[Tensor, ...], super().split(*args, **kwargs))
    @overload
    def repeat(self, repeats: Tuple[int, Literal[1]]) -> 'Matrix[int, C]': ...
    @overload
    def repeat(self, repeats: Tuple[Literal[1], int]) -> 'Matrix[R, int]': ...
    @overload
    def repeat(self,  repeats: Tuple[int, int]) -> 'Matrix[Any, Any]': ...
    @override
    def repeat(self, repeats: Tuple[int, int]) -> 'Matrix[Any, Any]': # type: ignore[override]
        return Matrix.load(super().repeat(repeats), shape = (self.shape[0] * repeats[0], self.shape[1] * repeats[1]))
    @override
    def multinomial[S: int](self, num_samples: S, *args: Any, **kwargs: Any) -> 'Matrix[R, S]':
        return Matrix.load(super().multinomial(*args, **kwargs), (self.shape[0], num_samples))
    @override
    def softmax(self, *args: Any, **kwargs: Any) -> 'Matrix[R, C]':
        return cast(Matrix[R, C], super().softmax(*args, **kwargs))
    @override
    def int(self) -> 'Matrix[R, C]':
        return cast(Matrix[R, C], super().int())
    @override
    def long(self) -> 'Matrix[R, C]':
        return cast(Matrix[R, C], super().long())
    @override
    def float(self) -> 'Matrix[R, C]':
        return cast(Matrix[R, C], super().float())
    @override
    def double(self) -> 'Matrix[R, C]':
        return cast(Matrix[R, C], super().double())
    @override
    def t(self) -> 'Matrix[C, R]':
        return cast(Matrix[C, R], super().t())
    @override
    def detach(self) -> 'Matrix[R, C]':
        return cast(Matrix[R, C], super().detach())
    @override
    def clone(self, **kwargs: Any) -> 'Matrix[R, C]':
        return cast(Matrix[R, C], super().clone(**kwargs))
    @override
    def to(self, *args: Any, **kwargs: Any) -> 'Matrix[R, C]':
        super().__getitem__
        return cast(Matrix[R, C], super().to(*args, **kwargs))
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
