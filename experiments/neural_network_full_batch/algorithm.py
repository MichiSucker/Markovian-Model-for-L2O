import torch
import torch.nn as nn
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm


class NnOptimizer(nn.Module):

    def __init__(self, dim: int):
        super(NnOptimizer, self).__init__()

        self.extrapolation = nn.Parameter(0.001 * torch.ones(dim))
        self.gradient = nn.Parameter(0.001 * torch.ones(dim))
        input_size, hidden_size, output_size = 6, 16, 1
        self.compute_update_step = nn.Sequential(
            nn.Conv2d(input_size, hidden_size, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_size, output_size, kernel_size=1),
        )

        input_size, hidden_size, output_size = 5, 8, 4
        self.compute_weighting_coefficients = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

        # For stability
        self.eps = torch.tensor(1e-10).float()

    def forward(self, algorithm: OptimizationAlgorithm) -> torch.Tensor:

        # Compute and normalize gradient
        grad = algorithm.loss_function.compute_gradient(algorithm.current_state[1])
        grad_norm = torch.linalg.norm(grad).reshape((1,))
        if grad_norm > self.eps:
            grad = grad / grad_norm

        # Compute and normalize momentum
        diff = algorithm.current_state[1] - algorithm.current_state[0]
        diff_norm = torch.linalg.norm(diff).reshape((1,))
        if diff_norm > self.eps:
            diff = diff / diff_norm

        coefficients = self.compute_weighting_coefficients(
            torch.concat(
                (torch.log(1 + grad_norm.reshape((1,))),
                 torch.log(1 + diff_norm.reshape((1,))),
                 torch.dot(grad, diff).reshape((1,)),
                 torch.log(algorithm.loss_function(algorithm.current_state[1]).detach().reshape((1,))),
                 torch.log(algorithm.loss_function(algorithm.current_state[0]).detach().reshape((1,)))))
        )
        update = self.compute_update_step(
            torch.concat((
                coefficients[0] * self.gradient * grad.reshape((1, 1, 1, -1)),
                coefficients[1] * self.extrapolation * diff.reshape((1, 1, 1, -1)),
                coefficients[2] * grad.reshape((1, 1, 1, -1)),
                coefficients[3] * diff.reshape((1, 1, 1, -1)),
                algorithm.current_state[0].reshape((1, 1, 1, -1)),
                algorithm.current_state[1].reshape((1, 1, 1, -1)),
                ), dim=1)).flatten()

        return algorithm.current_state[-1] + update

    @staticmethod
    def update_state(opt_algo: OptimizationAlgorithm) -> None:
        opt_algo.current_state[0] = opt_algo.current_state[1].detach().clone()
        opt_algo.current_state[1] = opt_algo.current_iterate.detach().clone()
