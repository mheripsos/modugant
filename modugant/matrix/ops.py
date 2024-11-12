from typing import Any, List, Literal, Tuple, Union, cast, overload

from torch import Tensor, stack
from torch import cat as t_cat
from torch import normal as t_normal
from torch import ones as t_ones
from torch import rand as t_rand
from torch import randint as t_randint
from torch import randn as t_randn
from torch import randperm as t_randperm
from torch import zeros as t_zeros
from torch.nn.functional import cross_entropy as t_cross_entropy
from torch.nn.functional import one_hot as t_one_hot

from modugant.matrix.index import Index

from .dim import Dim, One
from .matrix import Matrix


@overload
def cat[R: int, C: int](
    tensors: Tuple[Matrix[Any, C], ...],
    dim: Literal[0],
    shape: Tuple[R, C]
) -> Matrix[R, C]:...
@overload
def cat[R: int, C: int](
    tensors: Tuple[Matrix[R, Any], ...],
    dim: Literal[1],
    shape: Tuple[R, C]
) -> Matrix[R, C]:...
def cat[R: int, C: int](
    tensors: Union[Tuple[Matrix[int, C], ...], Tuple[Matrix[R, int], ...]],
    dim: Literal[0, 1],
    shape: Tuple[R, C]
) -> Matrix[Any, Any]:
    '''Concatenate matrices.'''
    return Matrix.load(t_cat(tensors, dim = dim), shape)

def zeros[R: int, C: int](shape: Tuple[R, C], **kwargs: Any) -> Matrix[R, C]:
    '''Create a matrix of zeros.'''
    return Matrix.load(t_zeros(shape, **kwargs), shape)

def ones[R: int, C: int](shape: Tuple[R, C], **kwargs: Any) -> Matrix[R, C]:
    '''Create a matrix of ones.'''
    return Matrix.load(t_ones(shape, **kwargs), shape)

def normal[R: int, C: int](mean: float, std: float, shape: Tuple[R, C], **kwargs: Any) -> Matrix[R, C]:
    '''Create a matrix of normal random values.'''
    return Matrix.load(t_normal(mean, std, shape, **kwargs), shape)

def rand[R: int, C: int](shape: Tuple[R, C], **kwargs: Any) -> Matrix[R, C]:
    '''Create a matrix of random values.'''
    return Matrix.load(t_rand(shape, **kwargs), shape)

def randn[R: int, C: int](shape: Tuple[R, C], **kwargs: Any) -> Matrix[R, C]:
    '''Create a matrix of normal random values.'''
    return Matrix.load(t_randn(shape, **kwargs), shape)

def randint[R: int, C: int](low: int, high: int, shape: Tuple[R, C], **kwargs: Any) -> Matrix[R, C]:
    '''Create a matrix of random integers.'''
    return Matrix.load(t_randint(low, high, shape, **kwargs), shape)

def randperm[N: int](n: N, **kwargs: Any) -> Index[N]:
    '''Create a matrix of random permutations.'''
    perm = cast(List[int], t_randperm(n, **kwargs).tolist())
    return Index.load(perm, n)

## Functional operations
def one_hot[N: int, C: int](matrix: Matrix[N, One], num_classes: C) -> Matrix[N, C]:
    '''Convert a matrix of one-hot vectors to a matrix of one-hot vectors with a different number of classes.'''
    return Matrix.load(t_one_hot(matrix, num_classes)[:, 0], (matrix.shape[0], num_classes))

@overload
def cross_entropy[R: int, C: int](
    logits: Matrix[R, C],
    targets: Matrix[R, One],
    reduction: Literal['none']
) -> Matrix[R, One]:...
@overload
def cross_entropy[R: int, C: int](
    logits: Matrix[R, C],
    targets: Matrix[R, One],
    reduction: Literal['mean', 'sum']
) -> Matrix[One, One]:...
@overload
def cross_entropy[R: int, C: int](
    logits: Matrix[R, C],
    targets: Matrix[R, Any],
    reduction: str
) -> Tensor:...
def cross_entropy[R: int, C: int](
    logits: Matrix[R, C],
    targets: Matrix[R, Any],
    reduction: str = 'mean',
    *args: Any,
    **kwargs: Any
) -> Tensor:
    '''Compute the cross-entropy loss.'''
    if len(targets.shape) == 2 and targets.shape[1] == 1:
        entropy = t_cross_entropy(
            logits,
            targets[..., 0],
            reduction = reduction,
            *args,
            **kwargs
        ).reshape(targets.shape)
        if reduction == 'none':
            return Matrix.load(entropy, targets.shape)
        else:
            return Matrix.load(entropy, (Dim.one(), Dim.one()))
    else:
        return t_cross_entropy(logits, targets, reduction = reduction, *args, **kwargs)

## Custom operations
def sums[R: int, C: int](matrices: Tuple[Matrix[R, C], ...]) -> Matrix[R, C]:
    '''Compute the sum of a matrix.'''
    return Matrix.load(stack(matrices).sum(dim = 0))
