import torch
import torch.nn as nn


class ConvNet(nn.Module):

    def __init__(self,
                 img_height: torch.tensor,
                 img_width: torch.tensor,
                 num_channels: torch.tensor,
                 kernel_size: int, smoothness: float):
        super(ConvNet, self).__init__()

        # Store parameters
        self.num_channels = num_channels
        self.width = img_width
        self.height = img_height
        self.dim = img_width * img_height
        self.shape = (num_channels, 1, img_height, img_width)
        self.kernel_size = kernel_size
        self.smoothness = smoothness

        in_channels = 4
        out_channels = 2
        h_size = 15
        self.compute_update_directions = nn.Sequential(
            nn.Conv2d(in_channels, h_size, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(h_size, h_size, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(h_size, h_size, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(h_size, out_channels, kernel_size=1, bias=False),
        )

        size_in = 11
        size_out = in_channels
        h_size = 10
        self.compute_weights = nn.Sequential(
            nn.Linear(size_in, 3 * h_size, bias=False),
            nn.ReLU(),
            nn.Linear(3 * h_size, 2 * h_size, bias=False),
            nn.ReLU(),
            nn.Linear(2 * h_size, h_size, bias=False),
            nn.ReLU(),
            nn.Linear(h_size, size_out, bias=False),
        )

        # For stability
        self.eps = torch.tensor(1e-20).float()

    def forward(self, opt_algo):

        # Extract regularization parameter
        regularization_parameter = opt_algo.loss_function.get_parameter()['mu']

        # Compute loss
        loss = opt_algo.loss_function(opt_algo.current_state[1]).reshape((1,))
        old_loss = opt_algo.loss_function(opt_algo.current_state[0]).reshape((1,))
        data_fidelity_loss = opt_algo.loss_function.functional_part(opt_algo.current_state[1]).reshape((1,))
        old_data_fidelity_loss = opt_algo.loss_function.functional_part(opt_algo.current_state[0]).reshape((1,))
        regularization_loss = opt_algo.loss_function.regularizer(opt_algo.current_state[1]).reshape((1,))
        old_regularization_loss = opt_algo.loss_function.regularizer(opt_algo.current_state[0]).reshape((1,))

        # Compute and normalize gradient(s).
        data_fidelity_gradient = opt_algo.loss_function.func_grad(opt_algo.current_state[1])
        data_fidelity_gradient_norm = torch.linalg.norm(data_fidelity_gradient).reshape((1,))

        regularization_gradient = opt_algo.loss_function.reg_grad(opt_algo.current_state[1])
        regularization_gradient_norm = torch.linalg.norm(regularization_gradient).reshape((1,))
        gradient = data_fidelity_gradient + regularization_gradient
        gradient_norm = torch.linalg.norm(gradient).reshape((1,))

        if gradient_norm > self.eps:
            gradient = gradient / gradient_norm

        if data_fidelity_gradient_norm > self.eps:
            data_fidelity_gradient = data_fidelity_gradient / data_fidelity_gradient_norm

        if regularization_gradient_norm > self.eps:
            regularization_gradient = regularization_gradient / regularization_gradient_norm

        # Compute new hidden state
        difference = opt_algo.current_state[1] - opt_algo.current_state[0]
        difference_norm = torch.linalg.norm(difference).reshape((1,))

        if difference_norm > self.eps:
            difference = difference / difference_norm

        weight_vector = self.compute_weights(torch.concat((
            regularization_parameter.reshape((1,)),
            torch.max(torch.abs(gradient)).reshape((1,)),
            torch.log(1. + old_loss) - torch.log(1. + loss),
            torch.log(1. + old_data_fidelity_loss) - torch.log(1. + data_fidelity_loss),
            torch.log(1. + old_regularization_loss) - torch.log(1. + regularization_loss),
            torch.log(1. + gradient_norm.reshape((1,))),
            torch.dot(data_fidelity_gradient, regularization_gradient).reshape((1,)),
            torch.dot(difference, data_fidelity_gradient).reshape((1,)),
            torch.dot(difference, regularization_gradient).reshape((1,)),
            torch.dot(difference, gradient).reshape((1,)),
            torch.log(1. + difference_norm),
        )))

        update_directions = self.compute_update_directions(torch.concat((
            weight_vector[0] * regularization_gradient.reshape(self.shape),
            weight_vector[1] * data_fidelity_gradient.reshape(self.shape),
            weight_vector[2] * difference.reshape(self.shape),
            weight_vector[3] * gradient.reshape(self.shape),
        ), dim=1)).reshape((-1, self.dim))

        result = (opt_algo.current_state[-1]
                  + gradient_norm * update_directions[0] / self.smoothness
                  + difference_norm * update_directions[1] - gradient_norm * gradient / self.smoothness).flatten()

        return result

    @staticmethod
    def update_state(opt_algo):
        opt_algo.current_state[0] = opt_algo.current_state[1].detach().clone()
        opt_algo.current_state[1] = opt_algo.current_iterate.detach().clone()
