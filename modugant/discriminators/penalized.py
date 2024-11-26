'''
SmoothedDiscriminator class for GAN discriminator with regularization penalty.

Classes:
    SmoothedDiscriminator: Discriminator abstract class with regularization penalty.
'''
from typing import cast, override

from torch import Tensor
from torch.autograd import grad

from modugant.discriminators.basic import BasicDiscriminator
from modugant.discriminators.extended import ExtendedDiscriminator
from modugant.matrix import Matrix
from modugant.matrix.dim import Dim, One
from modugant.matrix.ops import one_hot, ones, rand


class ReshapingDiscriminator[C: int, D: int](BasicDiscriminator[C, D]):
    '''Discriminator protocol class which can be smoothness regularized.'''

    def __init__(self, conditions: C, outputs: D) -> None:
        '''
        Initialize the discriminator with conditions and outputs.

        Args:
            conditions (C: int): The number of condition nodes.
            outputs (D: int): The number of output nodes.

        '''
        super().__init__(conditions, outputs)

    def reshape[N: int](self, condition: Matrix[N, C], data: Matrix[N, D]) -> Tensor:
        '''
        Reshape the incoming condition data to fit the discriminator's first layer.

        Args:
            condition (torch.Tensor): The conditional data.
            data (torch.Tensor): The data to reshape.

        Returns:
            torch.Tensor: The reshaped data.

        '''
        ...
    def unshape[N: int](self, data: Tensor, n: N) -> Matrix[N, One]:
        '''
        Reshape the discriminator's last layer to fit the output data.

        Args:
            data (torch.Tensor): The data to reshape.
            n (N: int): The number of rows to reshape.

        Returns:
            Matrix[N, D]: The reshaped data.

        '''
        ...
    @override
    def predict[N: int](self, condition: Matrix[N, C], data: Matrix[N, D]) -> Matrix[N, One]:
        folded = self.reshape(condition, data)
        predicted = self.forward(folded)
        unshaped = self.unshape(predicted, data.shape[0])
        return unshaped

class SmoothedDiscriminator[C: int, D: int](ExtendedDiscriminator[C, D]):
    '''Discriminator model for GANs with smoothness regularization.'''

    _discriminator: ReshapingDiscriminator[C, D]
    def __init__(
            self,
            discriminator: ReshapingDiscriminator[C, D],
            factor: float
        ) -> None:
        '''
        Initialize the discriminator with a regularization penalty.

        Args:
            discriminator (Discriminator[C, D]): The discriminator model.
            factor (float): The regularization penalty weight.

        '''
        super().__init__(discriminator)
        self._factor = factor
    def _blend[N: int](self, data: Matrix[N, D], target: Matrix[N, One]) -> Matrix[N, N]:
        # Generate interpolation points between the true and false data
        # sample probability of assigning each original data row to each blend row
        sample = rand((data.shape[0], data.shape[0]))
        # sample percentages of the true cases in the blend per row
        alpha = rand((data.shape[0], Dim.one()))
        # sample from the true cases, use arg-max of probability to select a true into each blend row
        # use one_hot to convert the arg-max index back into a one_hot representing rows
        # `trues` is a [size, data.shape[0]] matrix mapping with 0/1 the sampled true data into the blend
        # the true cases can be selected into the blend with `trues @ data`
        trues = one_hot(
            (sample * target.T).argmax(dim = 1, keepdim = True),
            num_classes = target.shape[0]
        )
        # repeat with false cases
        falses = one_hot(
            (sample * (1 - target).T).argmax(dim = 1, keepdim = True),
            num_classes = target.shape[0]
        )
        # use alpha/(1 - alpha) to weight the true and false cases into the blend
        # the resulting matrix can be used as a sampler with `blend @ data`
        return (alpha * trues) + ((1 - alpha) * falses)
    def penalty[N: int](self, condition: Matrix[N, C], data: Matrix[N, D], target: Matrix[N, One]) -> Matrix[One, One]:
        '''
        Calculate the regularization penalty for the discriminator.

        Args:
            condition (Matrix[N, C]): The condition data.
            data (Matrix[N, D]): The data to blend.
            target (Matrix[N, One]): The target data.

        Returns:
            Matrix[One, One]: The regularization penalty.

        '''
        # create an interpolated data set
        blend = self._blend(data, target)
        inputs = self._discriminator.reshape(
            blend @ condition,
            blend @ data
        )
        inputs.requires_grad = True
        # feed forward the blend data
        output = self._discriminator.forward(inputs)
        # get the gradients of the output with respect to the blend data
        gradient = grad(
            outputs = output,
            inputs = inputs,
            grad_outputs = ones((output.shape[0], output.shape[1])),
            create_graph = True,
            retain_graph = True
        )[0]
        # calculate the norm of the gradients per row
        norm = cast(Tensor, gradient.norm(dim = 1, keepdim = True))
        # penalize the distance of the norm from 1
        penalty = self._factor * ((norm - 1) ** 2).mean(dim = None, keepdim = True)
        return Matrix.cast(penalty, (Dim.one(), Dim.one()))
    @override
    def step[N: int](self, condition: Matrix[N, C], data: Matrix[N, D], target: Matrix[N, One]) -> Matrix[One, One]:
        self._discriminator.zero_grad()
        loss = self._discriminator.loss(condition, data, target)
        _ = loss.backward()
        penalty = self.penalty(condition, data, target)
        _ = penalty.backward(retain_graph = True)
        self._discriminator.optimizer.step()
        return loss

