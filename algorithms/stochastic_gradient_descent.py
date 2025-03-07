import torch
import torch.nn as nn


class SGD(nn.Module):

    def __init__(self, step_size: torch.tensor, batch_size: int):
        super(SGD, self).__init__()
        self.step_size = nn.Parameter(step_size)
        self.batch_size = batch_size

    def forward(self, optimization_algorithm):
        return (optimization_algorithm.current_state[-1]
                - self.step_size * optimization_algorithm.loss_function.compute_stochastic_gradient(
                    optimization_algorithm.current_state[-1], batch_size=self.batch_size))

    @staticmethod
    def update_state(opt_algo):
        opt_algo.current_state = opt_algo.current_iterate.detach().clone().reshape((1, -1))
