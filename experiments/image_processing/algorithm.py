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
        # They are actually not needed here; we just left such that we can easily reuse all the experimental setup
        # from before.
        self.num_channels = num_channels
        self.width = img_width
        self.height = img_height
        self.dim = img_width * img_height
        self.shape = (num_channels, 1, img_height, img_width)
        self.kernel_size = kernel_size
        self.smoothness = smoothness

        size_in = 3
        size_out = 2
        h_size = 9
        self.compute_weights = nn.Sequential(
            nn.Linear(size_in, h_size, bias=False),
            nn.Linear(h_size, h_size, bias=False),
            nn.ReLU(),
            nn.Linear(h_size, h_size, bias=False),
            nn.ReLU(),
            nn.Linear(h_size, h_size, bias=False),
            nn.ReLU(),
            nn.Linear(h_size, h_size, bias=False),
            nn.Linear(h_size, size_out, bias=False),
        )

        # For stability
        self.eps = torch.tensor(1e-20).float()

    def forward(self, opt_algo) -> torch.Tensor:

        # Compute loss
        loss = opt_algo.loss_function(opt_algo.current_state[1]).reshape((1,))
        old_loss = opt_algo.loss_function(opt_algo.current_state[0]).reshape((1,))

        # Compute gradient
        gradient = opt_algo.loss_function.compute_gradient(opt_algo.current_state[1])
        gradient_norm = torch.linalg.norm(gradient).reshape((1,))

        if gradient_norm > self.eps:
            gradient = gradient / gradient_norm

        # Compute new hidden state
        difference = opt_algo.current_state[1] - opt_algo.current_state[0]
        difference_norm = torch.linalg.norm(difference).reshape((1,))

        if difference_norm > self.eps:
            difference = difference / difference_norm

        weight_vector = self.compute_weights(torch.concat((
            torch.log(1. + loss) - torch.log(1. + old_loss),
            torch.log(1. + gradient_norm.reshape((1,))),
            torch.log(1. + difference_norm),
        )))
        result = opt_algo.current_state[1] - weight_vector[0] * gradient + weight_vector[1] * difference
        return result

    @staticmethod
    def update_state(opt_algo):
        opt_algo.current_state[0] = opt_algo.current_state[1].detach().clone()
        opt_algo.current_state[1] = opt_algo.current_iterate.detach().clone()


class ConvNetOld(nn.Module):

    def __init__(self, img_height: torch.tensor, img_width: torch.tensor, num_channels: torch.tensor,
                 kernel_size: int, smoothness: float):
        super(ConvNetOld, self).__init__()

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
        self.update_layer = nn.Sequential(
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
        self.weighting_layer = nn.Sequential(
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
        reg_param = opt_algo.loss_function.get_parameter()['mu']

        # Compute loss
        loss = opt_algo.loss_function(opt_algo.current_state[1]).reshape((1,))
        old_loss = opt_algo.loss_function(opt_algo.current_state[0]).reshape((1,))
        func_loss = opt_algo.loss_function.functional_part(opt_algo.current_state[1]).reshape((1,))
        old_func_loss = opt_algo.loss_function.functional_part(opt_algo.current_state[0]).reshape((1,))
        reg_loss = opt_algo.loss_function.regularizer(opt_algo.current_state[1]).reshape((1,))
        old_reg_loss = opt_algo.loss_function.regularizer(opt_algo.current_state[0]).reshape((1,))

        # Compute and normalize gradient(s).
        func_grad = opt_algo.loss_function.func_grad(opt_algo.current_state[1])
        func_grad_norm = torch.linalg.norm(func_grad).reshape((1,))
        reg_grad = opt_algo.loss_function.reg_grad(opt_algo.current_state[1])
        reg_grad_norm = torch.linalg.norm(reg_grad).reshape((1,))
        grad = func_grad + reg_grad
        grad_norm = torch.linalg.norm(grad).reshape((1,))

        if grad_norm > self.eps:
            grad = grad / grad_norm

        if func_grad_norm > self.eps:
            func_grad = func_grad/func_grad_norm

        if reg_grad_norm > self.eps:
            reg_grad = reg_grad/reg_grad_norm

        # Compute new hidden state
        diff = opt_algo.current_state[1] - opt_algo.current_state[0]
        diff_norm = torch.linalg.norm(diff).reshape((1,))

        if diff_norm > self.eps:
            diff = diff/diff_norm

        c = self.weighting_layer(torch.concat((
            reg_param.reshape((1,)),
            torch.max(torch.abs(grad)).reshape((1,)),
            torch.log(1. + old_loss) - torch.log(1. + loss),
            torch.log(1. + old_func_loss) - torch.log(1. + func_loss),
            torch.log(1. + old_reg_loss) - torch.log(1. + reg_loss),
            torch.log(1. + grad_norm.reshape((1,))),
            torch.dot(func_grad, reg_grad).reshape((1,)),
            torch.dot(diff, func_grad).reshape((1,)),
            torch.dot(diff, reg_grad).reshape((1,)),
            torch.dot(diff, grad).reshape((1,)),
            torch.log(1. + diff_norm),
        )))

        directions = self.update_layer(torch.concat((
            c[0] * reg_grad.reshape(self.shape),
            c[1] * func_grad.reshape(self.shape),
            c[2] * diff.reshape(self.shape),
            c[3] * grad.reshape(self.shape),
        ), dim=1)).reshape((-1, self.dim))

        result = (opt_algo.current_state[-1]
                  + grad_norm * directions[0] / self.smoothness
                  + diff_norm * directions[1] - grad_norm * grad / self.smoothness).flatten()

        return result

    @staticmethod
    def update_state(opt_algo):
        opt_algo.current_state[0] = opt_algo.current_state[1].detach().clone()
        opt_algo.current_state[1] = opt_algo.current_iterate.detach().clone()
