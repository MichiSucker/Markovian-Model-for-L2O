import torch
import torch.nn as nn
import torch.nn.functional as f
from typing import Callable, Tuple, List
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def polynomial_features(x: torch.Tensor, degree: int) -> torch.Tensor:
    x = x.unsqueeze(1)
    return torch.cat([x ** i for i in range(1, degree + 1)], 1).reshape((-1, degree))


class NeuralNetworkForStandardTraining(nn.Module):

    def __init__(self, degree: int):
        super().__init__()
        self.degree = degree
        self.fc1 = nn.Linear(self.degree, 10 * self.degree)
        self.fc2 = nn.Linear(10 * self.degree, 1)

    def get_shape_parameters(self) -> List:
        return [p.size() for p in self.parameters() if p.requires_grad]

    def get_dimension_of_weights(self) -> int:
        return sum([torch.prod(torch.tensor(s)).item() for s in self.get_shape_parameters()])

    def load_parameters_from_tensor(self, tensor: torch.Tensor) -> None:
        counter = 0
        # If the parameter is updated by the (learned) optimization algorithm, then they have corresponding entries in
        # the tensor, which should be loaded into the template.
        for param in self.parameters():
            if param.requires_grad:
                cur_size = torch.prod(torch.tensor(param.size()))
                cur_shape = param.shape
                param.data = tensor[counter:counter + cur_size].reshape(cur_shape)
                counter += cur_size

    def transform_parameters_to_tensor(self) -> torch.Tensor:
        all_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                all_params.append(param.flatten())
        return torch.concat(all_params, dim=0).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = f.relu(self.fc1(polynomial_features(x=x, degree=self.degree)))
        res = self.fc2(x)
        return res


def train_model(net: NeuralNetworkForStandardTraining,
                data: dict,
                criterion: Callable,
                batch_size: int,
                n_it: int,
                lr: float) -> Tuple[nn.Module, list, list]:

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    iterates, losses = [], []

    for i in range(n_it + 1):

        idx = torch.randint(low=0, high=len(data['x_values']), size=(batch_size,))  # Sample random functions
        iterates.append(net.transform_parameters_to_tensor())
        optimizer.zero_grad()
        loss = criterion(net(data['x_values'][idx]), data['y_values'][idx])
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    return net, losses, iterates


def grid_search(net, parameters, batch_size, criterion, lr_min, lr_max, num_lr_to_test, num_steps, path):

    num_runs_per_problem = 3

    # Setup learning rates that should be tested
    learning_rates_to_test = torch.linspace(start=lr_min, end=lr_max, steps=num_lr_to_test)

    all_losses = np.empty((len(learning_rates_to_test), num_runs_per_problem * len(parameters)))
    pbar = tqdm(enumerate(learning_rates_to_test))
    pbar.set_description('Gridsearch Adam')
    for i, lr in pbar:
        print(lr)
        losses_for_lr = []
        pbar_2 = tqdm(parameters)
        for p in pbar_2:
            pbar_3 = tqdm(range(num_runs_per_problem))
            for _ in pbar_3:

                # Compute losses of adam
                net, losses_adam, iterates_adam = train_model(net=net,
                                                              data=p,
                                                              batch_size=batch_size,
                                                              criterion=criterion,
                                                              n_it=num_steps,
                                                              lr=lr)

                losses_for_lr.append(losses_adam[-1])

        all_losses[i, :] = losses_for_lr

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    ax.plot(learning_rates_to_test, np.mean(all_losses, axis=1), linestyle='dashed', color='orange')
    ax.plot(learning_rates_to_test, np.median(all_losses, axis=1), linestyle='dotted', color='orange')
    ax.fill_between(learning_rates_to_test,
                    np.quantile(all_losses, q=0.025, axis=1),
                    np.quantile(all_losses, q=0.975, axis=1), alpha=0.3, color='orange')

    ax.grid('on')
    ax.set_title('Results Gridsearch')
    ax.set_ylabel('$\ell(x^{(n)}_{\mathrm{Adam}})$')
    ax.set_xlabel('$\\tau$')
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.tight_layout()
    fig.savefig(path + '/grid_search_adam.pdf', dpi=300, bbox_inches='tight')


class NeuralNetworkForLearning(nn.Module):

    def __init__(self, degree: int, shape_parameters: list) -> None:
        super().__init__()
        self.degree = degree
        self.shape_param = shape_parameters
        self.dim_param = [torch.prod(torch.tensor(p)) for p in shape_parameters]

    def forward(self, x: torch.Tensor, neural_net_parameters: torch.Tensor) -> torch.Tensor:
        x = polynomial_features(x=x, degree=self.degree)

        # From the neural_net_parameters (prediction of optimization algorithm), extract the weights of the neural
        # network into the corresponding torch.nn.functional-functions. Then, perform the prediction in the usual way,
        # that is, by calling them successively.
        c = 0
        for i in range(0, len(self.dim_param), 2):

            # Extract weights and biases and reshape them correctly using self.shape_param
            weights = neural_net_parameters[c:c+self.dim_param[i]]
            weights = weights.reshape(self.shape_param[i])
            bias = neural_net_parameters[c+self.dim_param[i]:c+self.dim_param[i]+self.dim_param[i+1]]
            bias = bias.reshape(self.shape_param[i+1])

            x = f.linear(input=x, weight=weights, bias=bias)
            c += self.dim_param[i] + self.dim_param[i+1]
            if len(self.shape_param) > 2 and (i+2 < len(self.dim_param)):
                x = f.relu(x)
        return x
