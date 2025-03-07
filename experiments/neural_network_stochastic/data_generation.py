import torch
import torch.nn as nn
from typing import Callable, Tuple
from experiments.neural_network_full_batch.neural_network import NeuralNetworkForLearning


def check_and_extract_number_of_datapoints(number_of_datapoints_per_dataset: dict) -> Tuple[int, int, int, int]:
    if (('prior' not in number_of_datapoints_per_dataset)
            or ('train' not in number_of_datapoints_per_dataset)
            or ('test' not in number_of_datapoints_per_dataset)
            or ('validation' not in number_of_datapoints_per_dataset)):
        raise ValueError("Missing number of datapoints.")
    else:
        return (number_of_datapoints_per_dataset['prior'],
                number_of_datapoints_per_dataset['train'],
                number_of_datapoints_per_dataset['test'],
                number_of_datapoints_per_dataset['validation'])


def get_loss_of_neural_network() -> Callable:
    return nn.MSELoss()


def get_distribution_of_datapoints() -> torch.distributions.Distribution:
    return torch.distributions.uniform.Uniform(-2., 2.)


def get_distribution_of_coefficients() -> torch.distributions.Distribution:
    return torch.distributions.uniform.Uniform(-5, 5)


def get_powers_of_polynomials() -> torch.Tensor:
    # Note that we return 6 here, because then torch.arange will return the numbers [0, ..., 5].
    return torch.arange(5+1)


def get_observations_for_x_values(number_of_samples: int,
                                  distribution_x_values: torch.distributions.Distribution) -> torch.Tensor:
    xes = distribution_x_values.sample(torch.Size((number_of_samples,)))
    xes, _ = torch.sort(xes)  # Sort them already now. This is needed, at least, for plotting.
    return xes.reshape((-1, 1))


def get_coefficients(distribution_of_coefficients: torch.distributions.Distribution,
                     maximal_degree: int) -> torch.Tensor:
    return distribution_of_coefficients.sample(torch.Size((maximal_degree + 1, )))


def get_ground_truth_values(x_values: torch.Tensor, coefficients: torch.Tensor, powers: torch.Tensor) -> torch.Tensor:
    return torch.sum(coefficients * (x_values ** powers), dim=1).reshape((-1, 1))


def get_y_values(ground_truth: torch.Tensor) -> torch.Tensor:
    return ground_truth + torch.randn(ground_truth.shape)


def get_loss_of_algorithm(neural_network: NeuralNetworkForLearning, loss_of_neural_network: Callable):

    def loss_function_of_the_algorithm(x: torch.Tensor, parameter: dict) -> torch.Tensor:
        return loss_of_neural_network(neural_network(x=parameter['x_values'], neural_net_parameters=x),
                                      parameter['y_values'])

    return loss_function_of_the_algorithm


def get_single_loss_of_algorithm(neural_network: NeuralNetworkForLearning, loss_of_neural_network: Callable):
    # TODO: Add Type-annotations
    def single_loss(x, d):
        return loss_of_neural_network(neural_network(x=d[0].reshape((1, 1)), neural_net_parameters=x),
                                      d[1].reshape((1, 1)))
    return single_loss


def create_parameter(x_values: torch.Tensor,
                     y_values: torch.Tensor,
                     ground_truth_values: torch.Tensor,
                     coefficients: torch.Tensor) -> dict:
    return {'x_values': x_values,
            'y_values': y_values,
            'dataset': torch.hstack((x_values, y_values)),
            'ground_truth_values': ground_truth_values,
            'coefficients': coefficients,
            'optimal_loss': torch.tensor(0.0)}


def get_parameters(number_of_datapoints_per_dataset: dict) -> dict:

    n_prior, n_train, n_test, n_val = check_and_extract_number_of_datapoints(number_of_datapoints_per_dataset)
    distribution_of_datapoints = get_distribution_of_datapoints()
    distribution_of_coefficients = get_distribution_of_coefficients()
    powers_of_polynomials = get_powers_of_polynomials()

    parameters = {'prior': [], 'train': [], 'test': [], 'validation': []}
    for number_of_functions, name in [(n_prior, 'prior'), (n_train, 'train'), (n_test, 'test'), (n_val, 'validation')]:
        for _ in range(number_of_functions):
            x_values = get_observations_for_x_values(number_of_samples=50,
                                                     distribution_x_values=distribution_of_datapoints)
            coefficients = get_coefficients(distribution_of_coefficients,
                                            maximal_degree=torch.max(powers_of_polynomials).item())
            ground_truth_values = get_ground_truth_values(x_values=x_values, coefficients=coefficients,
                                                          powers=powers_of_polynomials)
            y_values = get_y_values(ground_truth_values)
            parameters[name].append(create_parameter(x_values=x_values, y_values=y_values,
                                                     ground_truth_values=ground_truth_values,
                                                     coefficients=coefficients))
    return parameters


def get_data(neural_network: NeuralNetworkForLearning,
             number_of_datapoints_per_dataset: dict) -> Tuple[Callable, Callable, Callable, dict]:

    parameters = get_parameters(number_of_datapoints_per_dataset)
    loss_of_neural_network = get_loss_of_neural_network()
    loss_of_algorithm = get_loss_of_algorithm(neural_network, loss_of_neural_network)
    single_loss_of_algorithm = get_single_loss_of_algorithm(neural_network, loss_of_neural_network)

    return loss_of_algorithm, single_loss_of_algorithm, loss_of_neural_network, parameters
