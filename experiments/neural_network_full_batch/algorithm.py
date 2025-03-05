import torch
import torch.nn as nn
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm


class NnOptimizer(nn.Module):

    def __init__(self, dim: int):
        super(NnOptimizer, self).__init__()

        self.extrapolation = nn.Parameter(0.001 * torch.ones(dim))
        self.gradient = nn.Parameter(0.001 * torch.ones(dim))
        in_channels = 6
        hidden_channels = 16
        out_channels = 1
        self.compute_update_direction = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

        in_features = 5
        hidden_features = 8
        out_features = 4
        self.compute_weights = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features),
        )

        # For stability
        self.eps = torch.tensor(1e-10).float()

    def forward(self, algorithm: OptimizationAlgorithm) -> torch.Tensor:

        # Normalize gradient
        gradient = algorithm.loss_function.compute_gradient(algorithm.current_state[1])
        gradient_norm = torch.linalg.norm(gradient).reshape((1,))
        if gradient_norm > self.eps:
            gradient = gradient / gradient_norm

        # Compute new hidden state
        difference = algorithm.current_state[1] - algorithm.current_state[0]
        difference_norm = torch.linalg.norm(difference).reshape((1,))
        if difference_norm > self.eps:
            difference = difference / difference_norm

        weight_vector = self.compute_weights(
            torch.concat(
                (torch.log(1 + gradient_norm.reshape((1,))),
                 torch.log(1 + difference_norm.reshape((1,))),
                 torch.dot(gradient, difference).reshape((1,)),
                 torch.log(algorithm.loss_function(algorithm.current_state[1]).detach().reshape((1,))),
                 torch.log(algorithm.loss_function(algorithm.current_state[0]).detach().reshape((1,)))))
        )
        update_step = self.compute_update_direction(
            torch.concat((
                weight_vector[0] * self.gradient * gradient.reshape((1, 1, 1, -1)),
                weight_vector[1] * self.extrapolation * difference.reshape((1, 1, 1, -1)),
                weight_vector[2] * gradient.reshape((1, 1, 1, -1)),
                weight_vector[3] * difference.reshape((1, 1, 1, -1)),
                algorithm.current_state[0].reshape((1, 1, 1, -1)),
                algorithm.current_state[1].reshape((1, 1, 1, -1)),
                ), dim=1)).flatten()

        return algorithm.current_state[-1] + update_step

    @staticmethod
    def update_state(opt_algo: OptimizationAlgorithm) -> None:
        opt_algo.current_state[0] = opt_algo.current_state[1].detach().clone()
        opt_algo.current_state[1] = opt_algo.current_iterate.detach().clone()
