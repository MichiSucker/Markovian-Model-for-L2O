import torch
import torch.nn as nn
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm


class Quadratics(nn.Module):

    def __init__(self, dim: int):
        super(Quadratics, self).__init__()

        self.dim = dim

        in_size = 3
        h_size = 10
        size_out = 1
        self.compute_update_direction = nn.Sequential(
            nn.Conv2d(in_size, h_size, kernel_size=1, bias=False),
            nn.Conv2d(h_size, h_size, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(h_size, h_size, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(h_size, h_size, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(h_size, h_size, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(h_size, h_size, kernel_size=1, bias=False),
            nn.Conv2d(h_size, size_out, kernel_size=1, bias=False),
        )

        in_size = 4
        out_size = 1
        h_size = 8
        self.compute_step_size = nn.Sequential(
            nn.Linear(in_size, h_size, bias=False),
            nn.Linear(h_size, h_size, bias=False),
            nn.ReLU(),
            nn.Linear(h_size, h_size, bias=False),
            nn.ReLU(),
            nn.Linear(h_size, h_size, bias=False),
            nn.ReLU(),
            nn.Linear(h_size, h_size, bias=False),
            nn.Linear(h_size, out_size, bias=False),
        )

        # For stability
        self.eps = torch.tensor(1e-20).float()

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

        loss = opt_algo.loss_function(opt_algo.current_state[1]).detach()
        old_loss = opt_algo.loss_function(opt_algo.current_state[0]).detach()
        step_size = self.compute_step_size(
            torch.concat(
                (torch.log(1 + gradient_norm.reshape((1,))),
                 torch.log(1 + momentum_norm.reshape((1,))),
                 torch.log(1 + loss.reshape((1,))),
                 torch.log(1 + old_loss.reshape((1,)))
                 ))
        )
        direction = self.compute_update_direction(torch.concat((
            gradient.reshape((1, 1, 1, -1)),
            momentum.reshape((1, 1, 1, -1)),
            (gradient * momentum).reshape((1, 1, 1, -1)),
        ), dim=1)).reshape((self.dim, -1))

        return opt_algo.current_state[-1] - step_size * direction.flatten()

    @staticmethod
    def update_state(opt_algo: OptimizationAlgorithm) -> None:
        opt_algo.current_state[0] = opt_algo.current_state[1].detach().clone()
        opt_algo.current_state[1] = opt_algo.current_iterate.detach().clone()
