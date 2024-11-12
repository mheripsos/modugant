'''Basic Regimens.'''
import math
from typing import Optional, Tuple, override

from modugant.protocols import Action, Regimen


class BasicRegimen(Regimen):
    '''Basic Regimen.'''

    def __init__(
        self,
        batch: int,
        k: int = 1,
        d_factor: float = 1.0,
        g_factor: float = 1.0,
        max_iterations: int = 1000,
        target: Optional[Tuple[float, float, float]] = None,
        max_resets: int = 10
    ) -> None:
        '''
        Initialize the basic regimen.

        Args:
            batch (int): The batch size.
            k (int): The number of discriminator steps.
            d_factor (float): The discriminator factor.
            g_factor (float): The generator factor.
            max_iterations (int): The maximum number of iterations.
            target (Tuple[float, float, float]): The target loss and variance.
            max_resets (int): The maximum number of resets.

        '''
        super().__init__()
        self.__batch = batch
        self.__k = k
        self.__d_factor = d_factor
        self.__g_factor = g_factor
        self.__max_iterations = max_iterations
        self.__check_iter = max_iterations // 100
        self.__ramp_iter = max_iterations // 10
        if target is None:
            target = (0.0, 0.0, 0.0)
        self.__min_loss = target[0] or (math.log(2) - 0.001)
        self.__max_loss = target[1] or (math.log(2) + 0.001)
        self.__min_var = target[2] or 0.00001
        self.__sum_loss = 0
        self.__sum_sq_loss = 0
        self.__resets = 0
        self.__max_resets = max_resets
    @override
    def command(self, iteration: int, loss: Tuple[float, float]) -> Tuple[Action, str]:
        (d_loss, _) = loss
        self.__sum_loss = 0.9 * self.__sum_loss + 0.1 * d_loss
        self.__sum_sq_loss = 0.9 * self.__sum_sq_loss + 0.1 * d_loss ** 2
        if iteration >= self.__max_iterations:
            return ('stop', 'Max iterations reached.')
        if iteration % self.__check_iter == 0:
            if iteration >= self.__ramp_iter:
                var = self.__sum_sq_loss - self.__sum_loss ** 2
                if d_loss > self.__min_loss and d_loss < self.__max_loss and var < self.__min_var:
                    return ('stop', 'Target loss and variance reached.')
                elif var < self.__min_var:
                    self.__resets += 1
                    if self.__resets >= self.__max_resets:
                        return ('escape', 'Max resets reached.')
                    else:
                        return ('reset', 'Bad convergence detected.')
        return ('continue', 'Continue training.')
    @override
    def report(self, iteration: int, action: Action, message: str, d_loss: float, g_loss: float) -> None:
        '''Report the training progress.'''
        ignore = action == 'continue' and iteration % self.__check_iter != 0
        if not ignore:
            print(f'Iteration: {iteration}, D Loss: {d_loss}, G Loss: {g_loss}')
            if action != 'continue':
                print(f'\t{action}: {message}')
    @override
    def reset(self) -> None:
        '''Reset the regimen.'''
        self.__sum_loss = 0
        self.__sum_sq_loss = 0
        self.__resets = 0
    @property
    @override
    def batch(self) -> int:
        '''The batch size.'''
        return self.__batch
    @property
    @override
    def k(self) -> int:
        '''The number of discriminator steps.'''
        return self.__k
    @property
    @override
    def d_factor(self) -> float:
        '''The discriminator factor.'''
        return self.__d_factor
    @property
    @override
    def g_factor(self) -> float:
        '''The generator factor.'''
        return self.__g_factor

