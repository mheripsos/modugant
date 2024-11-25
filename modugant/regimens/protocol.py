from typing import Literal, Protocol, Tuple

type Action = Literal['stop', 'escape', 'reset', 'continue']


class Regimen(Protocol):
    '''
    Regimen for training GANs.

    Abstract methods (must be implemented in subclass):
        command: Get the action to take based on the iteration and loss.
            (iteration: int, loss: Tuple[float, float]) -> Tuple[Action, str]
        reset: Reset the regimen.
            () -> None

    '''

    def command(self, iteration: int, loss: Tuple[float, float]) -> Tuple[Action, str]:
        '''
        Get the action to take based on the iteration and loss.

        Args:
            iteration (int): The current iteration.
            loss (Tuple[float, float]): The loss of the discriminator and generator.

        Returns:
            Action: The action to take.

        '''
        ...
    def reset(self) -> None:
        '''Reset the regimen.'''
        ...
    def report(self, iteration: int, action: Action, message: str, d_loss: float, g_loss: float) -> None:
        '''
        Report the action taken.

        Args:
            iteration (int): The current iteration.
            action (Action): The action taken.
            message (str): The message to report.
            d_loss (float): The loss of the discriminator.
            g_loss (float): The loss of the generator.

        '''
        ...
    @property
    def batch(self) -> int:
        '''Batch size.'''
        ...
    @property
    def k(self) -> int:
        '''Number of steps to take before updating the generator.'''
        ...
    @property
    def d_factor(self) -> float:
        '''Ratio of the generated cases in discriminator step to use per real case.'''
        ...
    @property
    def g_factor(self) -> float:
        '''Ratio of the generated cases in generator step to use per real case.'''
        ...
