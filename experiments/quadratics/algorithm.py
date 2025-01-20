import torch
import torch.nn as nn
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm


class Quadratics(nn.Module):

    def __init__(self, dim: int):
        super(Quadratics, self).__init__()

        self.dim = dim

        size = 10
        size_out = 1
        self.update_layer = nn.Sequential(
            nn.Conv2d(3, size, kernel_size=1, bias=False),
            nn.Conv2d(size, size, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(size, size, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(size, size, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(size, size, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(size, size, kernel_size=1, bias=False),
            nn.Conv2d(size, size_out, kernel_size=1, bias=False),
        )

        h_size = 8
        self.coefficients = nn.Sequential(
            nn.Linear(4, h_size, bias=False),
            nn.ReLU(),
            nn.Linear(h_size, h_size, bias=False),
            nn.ReLU(),
            nn.Linear(h_size, h_size, bias=False),
            nn.ReLU(),
            nn.Linear(h_size, h_size, bias=False),
            nn.Linear(h_size, size_out, bias=False),
        )

        # For stability
        self.eps = torch.tensor(1e-24).float()

    def forward(self, opt_algo: OptimizationAlgorithm) -> torch.Tensor:

        # Compute and normalize gradient
        gradient = opt_algo.loss_function.compute_gradient(opt_algo.current_state[1])
        gradient_norm = torch.linalg.norm(gradient).reshape((1,))
        if gradient_norm > self.eps:
            gradient = gradient / gradient_norm

        # Compute and normalize momentum
        momentum = opt_algo.current_state[1] - opt_algo.current_state[0]
        momentum_norm = torch.linalg.norm(momentum).reshape((1,))
        if momentum_norm > self.eps:
            momentum = momentum / momentum_norm

        step_size = self.coefficients(
            torch.concat(
                (torch.log(1 + gradient_norm.reshape((1,))),
                 torch.log(1 + momentum_norm.reshape((1,))),
                 torch.log(1 + opt_algo.loss_function(opt_algo.current_state[1]).detach().reshape((1,))),
                 torch.log(1 + opt_algo.loss_function(opt_algo.current_state[0]).detach().reshape((1,)))
                 ))
        )
        direction = self.update_layer(torch.concat((
            gradient.reshape((1, 1, 1, -1)),
            momentum.reshape((1, 1, 1, -1)),
            (gradient * momentum).reshape((1, 1, 1, -1)),
        ), dim=1)).reshape((self.dim, -1))

        # old update: torch.matmul(direction, step_size).flatten()
        return opt_algo.current_state[-1] - step_size * direction.flatten()

    @staticmethod
    def update_state(opt_algo: OptimizationAlgorithm) -> None:
        opt_algo.current_state[0] = opt_algo.current_state[1].detach().clone()
        opt_algo.current_state[1] = opt_algo.current_iterate.detach().clone()
