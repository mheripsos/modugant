'''
GAN Trainer for training GANs.

Classes:
    Trainer: Trainer for GANs.

'''
from contextlib import contextmanager
from typing import Iterator, Optional, Tuple, cast

from torch import device

from modugant.device import Device, check_device
from modugant.matrix import Dim
from modugant.matrix.ops import cat, ones, zeros
from modugant.protocols import Action, Connector, Discriminator, Generator, Regimen, Reporter


def no_op(_: int, __: Action, ___: str, ____: float, _____: float) -> None:
    '''No operation.'''
    pass

class Trainer[C: int, L: int, G: int, D: int]:
    '''
    Trainer for GANs.

    Type parameters:
        C: The number of conditions.
        L: The number of latent inputs.
        G: The number of generated outputs.
        D: The number of data inputs.

    '''

    def __init__(
        self,
        generator: Generator[C, L, G],
        discriminator: Discriminator[C, D],
        connector: Connector[C, G, D],
        device: Device = 'cpu'
    ) -> None:
        '''
        Initialize the trainer.

        Args:
            connector (Connector): The data connector.
            generator (Generator): The generator.
            discriminator (Discriminator): The discriminator.
            device (Device): The device to use.

        '''
        self.__device = check_device(device)
        self.__generator = generator.move(self.__device)
        self.__discriminator = discriminator.move(self.__device)
        self.__connector = connector.move(self.__device)
    def restart(self) -> None:
        '''Restart the learning rate scheduler.'''
        self.__connector.restart()
        self.__discriminator.reset()
        self.__discriminator.restart()
        self.__generator.reset()
        self.__generator.restart()
    def step[DN: int, GN: int](self, regimen: Regimen) -> Iterator[Tuple[int, Tuple[float, float]]]:
        '''
        Step the trainer.

        Args:
            regimen (Regimen): The training regimen.
            report (Optional[Callable[[str], None]]): The report function.

        Yields:
            Tuple[int, Tuple[float, float]]: The step number and the errors.

        '''
        d_sub = int(regimen.batch * regimen.d_factor)
        d_size = cast(DN, d_sub + regimen.batch)
        g_size = cast(GN, int(regimen.batch * regimen.g_factor))
        trues = ones((g_size, Dim.one()), device = self.__device)
        labels = cat(
            (
                ones((regimen.batch, Dim.one()), device = self.__device),
                zeros((d_sub, Dim.one()), device = self.__device)
            ),
            dim = 0,
            shape = (d_size, Dim.one())
        )
        i = 0
        with device(self.__device):
            while True:
                d_error = 0
                g_error = 0
                for _ in range(regimen.k):
                    r_data = self.__connector.sample(regimen.batch)
                    r_condition = self.__connector.condition(r_data)
                    f_condition = self.__connector.condition(
                        self.__connector.sample(int(regimen.batch * regimen.d_factor))
                    )
                    generated = self.__generator.sample(f_condition).detach()
                    f_data = self.__connector.prepare(f_condition, generated)
                    loss = self.__discriminator.step(
                        cat((r_condition, f_condition), dim = 0, shape = (d_size, r_condition.shape[1])),
                        cat((r_data, f_data), dim = 0, shape = (d_size, r_data.shape[1])),
                        labels
                    )
                    d_error = loss.item()
                f_condition = self.__connector.condition(self.__connector.sample(g_size))
                generated = self.__generator.sample(f_condition)
                fake = self.__connector.prepare(f_condition, generated)
                d_loss = self.__discriminator.loss(f_condition, fake, trues)
                c_loss = self.__connector.loss(f_condition, generated)
                self.__generator.update(d_loss + c_loss)
                self.__connector.update()
                g_error = d_loss.item()
                yield i, (d_error, g_error)
                i += 1
    def train(self, regimen: Regimen, report: Optional[Reporter] = None) -> None:
        '''
        Train the GAN.

        Args:
            regimen (Regimen): The training regimen.
            report (Optional[Callable[[str], None]]): The report function.

        '''
        report = report or regimen.report

        for i, (d_error, g_error) in self.step(regimen):
            (command, message) = regimen.command(i, (d_error, g_error))
            report(i, command, message, d_error, g_error)
            if command == 'stop':
                break
            elif command == 'escape':
                raise ValueError('Discriminator did not converge')
            elif command == 'reset':
                self.__discriminator.reset()
                self.__discriminator.restart()
    @contextmanager
    def test(self) -> Iterator[None]:
        '''Enter the test mode.'''
        with device(self.__device):
            _ = self.__generator.train(False)
            _ = self.__discriminator.train(False)
            yield
            _ = self.__generator.train(True)
            _ = self.__discriminator.train(True)
