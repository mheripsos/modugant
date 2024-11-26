from typing import Any, Literal, Tuple, Union, overload

from torch import arange as t_arange
from torch import cat as t_cat
from torch import eye as t_eye
from torch import normal as t_normal
from torch import ones as t_ones
from torch import rand as t_rand
from torch import randint as t_randint
from torch import randn as t_randn
from torch import stack
from torch import zeros as t_zeros
from torch.nn.functional import cross_entropy as t_cross_entropy
from torch.nn.functional import one_hot as t_one_hot

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
    return Matrix.cast(t_cat(tensors, dim = dim), shape)

def arange[N: int](end: N) -> Matrix[N, One]:
    '''Create a matrix of range values.'''
    return Matrix.cast(t_arange(end).reshape(end, 1), (end, Dim.one()))

def zeros[R: int, C: int](shape: Tuple[R, C], **kwargs: Any) -> Matrix[R, C]:
    '''Create a matrix of zeros.'''
    return Matrix.cast(t_zeros(shape, **kwargs), shape)

def ones[R: int, C: int](shape: Tuple[R, C], **kwargs: Any) -> Matrix[R, C]:
    '''Create a matrix of ones.'''
    return Matrix.cast(t_ones(shape, **kwargs), shape)

def eye[D: int](dim: D) -> Matrix[D, D]:
    '''Create an identity matrix.'''
    return Matrix.cast(t_eye(dim, dim), (dim, dim))

def normal[R: int, C: int](mean: float, std: float, shape: Tuple[R, C], **kwargs: Any) -> Matrix[R, C]:
    '''Create a matrix of normal random values.'''
    return Matrix.cast(t_normal(mean, std, shape, **kwargs), shape)

def rand[R: int, C: int](shape: Tuple[R, C], **kwargs: Any) -> Matrix[R, C]:
    '''Create a matrix of random values.'''
    return Matrix.cast(t_rand(shape, **kwargs), shape)

def randn[R: int, C: int](shape: Tuple[R, C], **kwargs: Any) -> Matrix[R, C]:
    '''Create a matrix of normal random values.'''
    return Matrix.cast(t_randn(shape, **kwargs), shape)

def randint[R: int, C: int](low: int, high: int, shape: Tuple[R, C], **kwargs: Any) -> Matrix[R, C]:
    '''Create a matrix of random integers.'''
    return Matrix.cast(t_randint(low, high, shape, **kwargs), shape)

## Functional operations
def one_hot[N: int, C: int](matrix: Matrix[N, One], num_classes: C) -> Matrix[N, C]:
    '''Convert a matrix of one-hot vectors to a matrix of one-hot vectors with a different number of classes.'''
    return Matrix.cast(t_one_hot(matrix, num_classes)[:, 0], (matrix.shape[0], num_classes))

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
def cross_entropy[R: int, C: int](
    logits: Matrix[R, C],
    targets: Matrix[R, One],
    reduction: str = 'mean',
    *args: Any,
    **kwargs: Any
) -> Matrix[Any, Any]:
    '''Compute the cross-entropy loss.'''
    entropy = t_cross_entropy(
        logits,
        targets[..., 0],
        reduction = reduction,
        *args,
        **kwargs
    ).reshape(targets.shape)
    if reduction == 'none':
        return Matrix.cast(entropy, targets.shape)
    else:
        return Matrix.cast(entropy, (Dim.one(), Dim.one()))

## Custom operations
def sums[R: int, C: int](matrices: Tuple[Matrix[R, C], ...]) -> Matrix[R, C]:
    '''Compute the sum of a matrix.'''
    return Matrix.cast(stack(matrices).sum(dim = 0), matrices[0].shape)

def means[R: int, C: int](matrices: Tuple[Matrix[R, C], ...]) -> Matrix[R, C]:
    '''Compute the mean of a matrix.'''
    return Matrix.cast(stack(matrices).mean(dim = 0), matrices[0].shape)
