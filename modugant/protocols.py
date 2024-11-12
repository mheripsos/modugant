'''
Protocol classes for GANs.

Protocols:
    Generator: Protocol for GAN generator.
    Discriminator: Protocol for GAN discriminator.
    Sampler: Protocol for GAN sampler.
    Regimen: Protocol for GAN training regimen.
'''

from typing import Callable, Literal, Protocol, Self, Tuple

from modugant.device import Device
from modugant.matrix import Matrix
from modugant.matrix.dim import One

type Action = Literal['stop', 'escape', 'reset', 'continue']

type Reporter = Callable[[int, Action, str, float, float], None]


class WithLatent[L: int](Protocol):
    '''Abstract class with latent property.'''

    _latents: L
    @property
    def latents(self) -> L:
        '''The number of latent inputs.'''
        return self._latents

class WithConditions[C: int](Protocol):
    '''Abstract class with conditions property.'''

    _conditions: C
    @property
    def conditions(self) -> C:
        '''The number of conditions.'''
        return self._conditions

class WithIntermediates[G: int](Protocol):
    '''Abstract class with intermediates property.'''

    _intermediates: G
    @property
    def intermediates(self) -> G:
        '''The number of intermediates.'''
        return self._intermediates

class WithOutputs[D: int](Protocol):
    '''Abstract class with outputs property.'''

    _outputs: D
    @property
    def outputs(self) -> D:
        '''The number of outputs.'''
        return self._outputs

class Generator[C: int, L: int, G: int](WithConditions[C], WithLatent[L], WithIntermediates[G], Protocol):
    '''
    Generator for GANs.

    Type parameters:
        C: The number of conditions.
        L: The number of latent inputs.
        G: The number of generated outputs.

    Abstract properties (must be assigned in subclass):
        _conditions: C; The number of conditions.
        _latents: L; The number of latent inputs.
        _intermediates: G; The number of generated outputs.

    Abstract methods (must be implemented in subclass):
        sample: Generate a sample from conditions.
            [N:int](condition: Matrix[N, C]) -> Matrix[N, G]
        update: Update the generator with the given loss.
            (loss: Matrix[One, One]) -> None
        reset: Reset the generator.
            () -> None
        restart: Restart the learning rate scheduler.
            () -> None
        rate: The current learning rate of the generator.
            property: () -> float

    '''

    def sample[N: int](self, condition: Matrix[N, C]) -> Matrix[N, G]:
        '''
        Generate a sample from conditions.

        Args:
            condition (Tensor (N, C)): The condition for the sample.

        Returns:
            Tensor (N, G): The generated sample.

        '''
        ...
    def update(self, loss: Matrix[One, One]) -> None:
        '''
        Update the discriminator with the given loss.

        Args:
            loss (Tensor (1, 1)): The loss of the generator.

        '''
        ...
    def reset(self) -> None:
        '''Reset the discriminator.'''
        ...
    def restart(self) -> None:
        '''Restart the learning rate scheduler.'''
        ...
    def move(self, device: Device) -> Self:
        '''Move the generator to the device.'''
        ...
    def train(self, mode: bool) -> Self:
        '''Set the generator to training mode.'''
        ...
    @property
    def rate(self) -> float:
        '''The current learning rate of the generator.'''
        ...

class Discriminator[C: int, D: int](WithConditions[C], WithOutputs[D], Protocol):
    '''
    Discriminator for GANs.

    Type parameters:
        C: The number of conditions.
        D: The number of inputs of data.

    Abstract properties (must be assigned in subclass):
        _conditions: C; The number of conditions.
        _outputs: D; The number of inputs of data.

    Abstract methods (must be implemented in subclass):
        predict: Pass the data through the discriminator.
            [N:int](condition: Matrix[N, C], data: Matrix[N, D]) -> Matrix[N, One]
        loss: Calculate the loss of the discriminator on the given predictions and target.
            [N:int](condition: Matrix[N, C], data: Matrix[N, D], target: Matrix[N, One]) -> Matrix[One, One]
        step: Update the discriminator on the given data and target. Return the loss.
            [N:int](condition: Matrix[N, C], data: Matrix[N, D], target: Matrix[N, One]) -> Matrix[One, One]
        reset: Reset the discriminator parameters.
            () -> None
        restart: Restart the discriminators state
            () -> None
        rate: The current learning rate of the generator.
            property: () -> float

    '''

    def predict[N: int](self, condition: Matrix[N, C], data: Matrix[N, D]) -> Matrix[N, One]:
        '''
        Pass the data through the discriminator.

        Args:
            condition (Tensor (N, C)): The condition for the data.
            data (Tensor (N, D)): The data to pass through the
                discriminator.

        Returns:
            Tensor (N, 1): The output of the discriminator.

        '''
        ...
    def loss[N: int](self, condition: Matrix[N, C], data: Matrix[N, D], target: Matrix[N, One]) -> Matrix[One, One]:
        '''
        Calculate the loss of the discriminator on the given predictions and target.

        Args:
            condition (Tensor (N, C)): The condition for the data.
            data (Tensor (N, D)): The data to pass through the discriminator.
            target (Tensor (N, 1)): The target of the

        Returns:
            Tensor (1, 1): The loss of the discriminator.

        '''
        ...
    def step[N: int](self, condition: Matrix[N, C], data: Matrix[N, D], target: Matrix[N, One]) -> Matrix[One, One]:
        '''
        Update the discriminator on the given data and target. Return the loss.

        Args:
            condition (Tensor (N, C)): The condition for the data.
            data (Tensor (N, D)): The data to pass through the discriminator.
            target (Tensor (N, 1)): The target of the discriminator

        Returns:
            Tensor (1, 1): The loss of the discriminator.

        '''
        ...
    def reset(self) -> None:
        '''Reset the discriminator.'''
        ...
    def restart(self) -> None:
        '''Restart the learning rate scheduler.'''
        ...
    def move(self, device: Device) -> Self:
        '''Move the generator to the device.'''
        ...
    def train(self, mode: bool) -> Self:
        '''Set the generator to training mode.'''
        ...
    @property
    def rate(self) -> float:
        '''The current learning rate of the generator.'''
        ...

class Conditioner[C: int, D: int](WithConditions[C], WithOutputs[D], Protocol):
    '''
    Conditioner for GANs.

    Type parameters:
        C: The number of conditions.
        D: The number of inputs of data.

    Abstract properties (must be assigned in subclass):
        _conditions: C; The number of conditions.
        _outputs: D; The number of inputs of data.

    Abstract methods (must be implemented in subclass):
        condition: Condition the data.
            [N:int](data: Matrix[N, D]) -> Matrix[N, C]

    '''

    def condition[N: int](self, data: Matrix[N, D]) -> Matrix[N, C]:
        '''
        Condition the data.

        Args:
            data (Tensor (N, D)): The data to condition.

        Returns:
            Tensor (N, C): The conditioned data.

        '''
        ...

class Inteceptor[C: int, G: int, D: int](WithConditions[C], WithIntermediates[G], WithOutputs[D], Protocol):
    '''
    Inteceptor for GANs.

    Type parameters:
        C: The number of conditions.
        G: The number of generated outputs.
        D: The number of transformed outputs.

    Abstract properties (must be assigned in subclass):
        _conditions: C; The number of conditions.
        _intermediates: G; The number of generated outputs.
        _outputs: D; The number of transformed outputs

    Abstract methods (must be implemented in subclass):
        prepare: Transform the data based on the condition.
            [N:int](condition: Matrix[N, C], intermediate: Matrix[N, G]) -> Matrix[N, D]

    '''

    def prepare[N: int](self, condition: Matrix[N, C], intermediate: Matrix[N, G]) -> Matrix[N, D]:
        '''
        Transform the data based on the condition.

        Args:
            condition (Tensor (N, C)): The condition for the data.
            intermediate (Tensor (N, G)): The generated output to transform.

        Returns:
            Tensor (N, D): The transformed data.

        '''
        ...

class Updater[C: int, G: int](WithConditions[C], WithIntermediates[G], Protocol):
    '''
    Updater for GANs.

    Type parameters:
        C: The number of conditions.
        G: The number of generated outputs.

    Abstract properties (must be assigned in subclass):
        _conditions: C; The number of conditions.
        _intermediates: G; The number of generated outputs.

    Abstract methods (must be implemented in subclass):
        loss: Calculate the loss of the data source on the given condition and output.
            [N:int](condition: Matrix[N, C], intermediate: Matrix[N, G]) -> Matrix[One, One]
        update: Update the data source.
            () -> None

    '''

    def loss[N: int](self, condition: Matrix[N, C], intermediate: Matrix[N, G]) -> Matrix[One, One]:
        '''
        Calculate the loss of the data source on the given condition and output.

        Args:
            condition (Tensor (N, C)): The condition for the data source.
            intermediate (Tensor (N, G)): The generated output.

        Returns:
            Tensor (1, 1): The loss of the data source.

        '''
        ...
    def update(self) -> None:
        '''Update the data source.'''
        ...

class Loader[D: int](WithOutputs[D], Protocol):
    '''
    Loader for GANs.

    Type parameters:
        D: The number of inputs of data.

    Abstract properties (must be assigned in subclass):
        _outputs: D; The number of outputs of data.

    Abstract methods (must be implemented in subclass):
        load: Encode the data.
            [N:int](data: Matrix[N, int]) -> Matrix[N, D]

    '''

    def load[N: int](self, data: Matrix[N, int]) -> Matrix[N, D]:
        '''
        Encode the data.

        Args:
            data (Tensor (N, X)): The data to encode.

        Returns:
            Tensor (N, D): The condition and encoded data.

        '''
        ...

class Transformer[C: int, G: int, D: int](
    Conditioner[C, D],
    Inteceptor[C, G, D],
    Updater[C, G],
    Loader[D],
    Protocol
):
    '''
    Transformer for GANs.

    Type parameters:
        C: The number of conditions.
        G: The number of generated outputs.
        D: The number of transformed outputs.

    Abstract properties (must be assigned in subclass):
        _conditions: C; The number of conditions.
        _intermediates: G; The number of generated outputs.
        _outputs: D; The number of transformed outputs.

    Abstract methods (must be implemented in subclass):
        condition: Condition the data.
            [N:int](data: Matrix[N, D]) -> Matrix[N, C]
        prepare: Transform the data based on the condition.
            [N:int](condition: Matrix[N, C], intermediate: Matrix[N, G]) -> Matrix[N, D]
        loss: Calculate the loss of the data source on the given condition and output.
            [N:int](condition: Matrix[N, C], intermediate: Matrix[N, G]) -> Matrix[One, One]
        update: Update the data source.
            () -> None
        load: Encode the data.
            [N:int](data: Matrix[N, int]) -> Matrix[N, D]

    '''

    ...

class Sampler[D: int](WithOutputs[D], Protocol):
    '''
    Sampler for GANs.

    Type parameters:
        D: The number of inputs of data.

    Abstract properties (must be assigned in subclass):
        _outputs: D; The number of outputs of data.

    Abstract methods (must be implemented in subclass):
        sample: Sample the data.
            [N:int](batch: N) -> Matrix[N, D]
        restart: Restart the sampler.
            () -> None
    '''

    def sample[N: int](self, batch: N) -> Matrix[N, D]:
        '''
        Sample the data.

        Args:
            batch (int): The batch size.

        Returns:
            Tensor (N, D): Sampled data.

        '''
        ...
    def restart(self) -> None:
        '''Restart the sampler.'''
        ...
    @property
    def holdout(self) -> Matrix[int, D]:
        '''The holdout test data.'''
        ...

class Connector[C: int, G: int, D: int](
    Conditioner[C, D],
    Inteceptor[C, G, D],
    Updater[C, G],
    Sampler[D],
    Protocol
):
    '''
    Connector for GANs.

    Type parameters:
        C: The number of conditions.
        G: The number of generated outputs.
        D: The number of transformed outputs.

    Abstract properties (must be assigned in subclass):
        _conditions: C; The number of conditions.
        _intermediates: G; The number of generated outputs.
        _outputs: D; The number of transformed outputs.

    Abstract methods (must be implemented in subclass):
        condition: Condition the data.
            [N:int](data: Matrix[N, D]) -> Matrix[N, C]
        prepare: Transform the data based on the condition.
            [N:int](condition: Matrix[N, C], intermediate: Matrix[N, G]) -> Matrix[N, D]
        loss: Calculate the loss of the data source on the given condition and output.
            [N:int](condition: Matrix[N, C], intermediate: Matrix[N, G]) -> Matrix[One, One]
        update: Update the data source.
            () -> None
        sample: Sample the data.
            [N:int](batch: N) -> Matrix[N, D]
        restart: Restart the sampler.
            () -> None
    '''

    ...
    def move(self, device: Device) -> Self:
        '''Move the connector to the device.'''
        ...

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
