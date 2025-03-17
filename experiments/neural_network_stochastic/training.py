import torch
from typing import List, Callable, Tuple
from pathlib import Path
from classes.LossFunction.class_LossFunction import LossFunction
from classes.StoppingCriterion.derived_classes.subclass_LossCriterion import LossCriterion
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm
from algorithms.stochastic_gradient_descent import StochasticGradientDescent
from experiments.neural_network_stochastic.data_generation import get_data, get_powers_of_polynomials
from experiments.neural_network_stochastic.neural_network import (NeuralNetworkForLearning,
                                                                  NeuralNetworkForStandardTraining)
from classes.LossFunction.derived_classes.derived_classes.StochasticParametricLossFunctions import (
    StochasticParametricLossFunction)
from classes.OptimizationAlgorithm.derived_classes.derived_classes.subclass_PacBayesOptimizationAlgorithm import (
    PacBayesOptimizationAlgorithm)
from experiments.neural_network_stochastic.algorithm import StochasticNnOptimizer
from exponential_family.describing_property.average_rate_property import get_rate_property

from classes.Constraint.class_ProbabilisticConstraint import ProbabilisticConstraint
from classes.Constraint.class_Constraint import create_list_of_constraints_from_functions, Constraint
import numpy as np
from numpy.typing import NDArray
import pickle


def get_number_of_datapoints() -> dict:
    return {'prior': 500, 'train': 1000, 'test': 250, 'validation': 250}


def create_folder_for_storing_data(path_of_experiment: str) -> str:
    savings_path = path_of_experiment + "/data/"
    Path(savings_path).mkdir(parents=True, exist_ok=True)
    return savings_path


def create_parametric_loss_functions_from_parameters(template_loss_function: Callable, template_single_loss: Callable,
                                                     parameters: dict) -> dict:
    loss_functions = {
        'prior': [StochasticParametricLossFunction(function=template_loss_function,
                                                   single_function=template_single_loss,
                                                   parameter=p)
                  for p in parameters['prior']],

        'train': [StochasticParametricLossFunction(function=template_loss_function,
                                                   single_function=template_single_loss,
                                                   parameter=p)
                  for p in parameters['train']],

        'test': [StochasticParametricLossFunction(function=template_loss_function,
                                                  single_function=template_single_loss,
                                                  parameter=p)
                 for p in parameters['test']],

        'validation': [StochasticParametricLossFunction(function=template_loss_function,
                                                        single_function=template_single_loss,
                                                        parameter=p)
                       for p in parameters['validation']],
    }
    return loss_functions


def get_initial_state(dim: int) -> torch.Tensor:
    # Note that it is important to keep the same initial point: Here, the algorithm only gets on a single starting
    # point, so it depends on this concrete initialization.
    size_state = 3
    initial_state = torch.zeros(size_state * dim).reshape((size_state, -1))
    initial_state[-1] = torch.randn(dim).reshape((1, dim))
    return initial_state


def get_parameters_of_estimation() -> dict:
    return {'quantile_distance': 0.075, 'quantiles': (0.01, 0.99), 'probabilities': (0.95, 1.0)}


def get_update_parameters() -> dict:
    return {'num_iter_print_update': 1000,
            'with_print': True,
            'bins': [1e5, 1e4, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2][::-1]}


def get_sampling_parameters(maximal_number_of_iterations: int) -> dict:
    length_trajectory = 1
    restart_probability = length_trajectory / maximal_number_of_iterations
    return {'lr': torch.tensor(1e-6),
            'length_trajectory': length_trajectory,
            'with_restarting': True,
            'restart_probability': restart_probability,
            'num_samples': 10,
            'num_iter_burnin': 0}


def get_fitting_parameters(maximal_number_of_iterations: int) -> dict:
    length_trajectory = 1
    restart_probability = length_trajectory / maximal_number_of_iterations
    return {'restart_probability': restart_probability,
            'length_trajectory': length_trajectory,
            # TODO: Rename n_max to number_of_training_iterations
            'n_max': int(500e3),
            'lr': 1e-4,
            'num_iter_update_stepsize': int(40e3),
            'factor_stepsize_update': 0.5}


def get_initialization_parameters() -> dict:
    return {'lr': 1e-3, 'num_iter_max': 1000, 'num_iter_print_update': 200, 'num_iter_update_stepsize': 200,
            'with_print': True}


def get_describing_property() -> Tuple[Callable, Callable]:
    pac_bayes_parameters = get_pac_bayes_parameters()
    rate_property, rate_constraint = get_rate_property(bound=pac_bayes_parameters['upper_bound'],
                                                       n_max=pac_bayes_parameters['n_max'])
    return rate_property, rate_constraint


def get_constraint_parameters(number_of_training_iterations: int) -> dict:
    pac_bayes_parameters = get_pac_bayes_parameters()
    describing_property, _ = get_describing_property()
    return {'describing_property': describing_property,
            'num_iter_update_constraint': int(number_of_training_iterations // 4),
            'upper_bound': pac_bayes_parameters['upper_bound']}


def get_pac_bayes_parameters() -> dict:
    return {'epsilon': torch.tensor(0.05),
            'upper_bound': 1.0,
            # TODO: Rename n_max to maximal_number_of_iterations
            'n_max': 2500}


def get_constraint(parameters_of_estimation: dict, loss_functions_for_constraint: List[LossFunction]) -> Constraint:
    describing_property, _ = get_describing_property()
    list_of_constraints = create_list_of_constraints_from_functions(describing_property=describing_property,
                                                                    list_of_functions=loss_functions_for_constraint)
    probabilistic_constraint = ProbabilisticConstraint(list_of_constraints=list_of_constraints,
                                                       parameters_of_estimation=parameters_of_estimation)
    return probabilistic_constraint.create_constraint()


def get_stopping_criterion():
    return LossCriterion(threshold=0.9)


def get_batch_size():
    return 5


def get_algorithm_for_learning(loss_functions: dict,
                               dimension_of_optimization_variable: int) -> PacBayesOptimizationAlgorithm:

    initial_state = get_initial_state(dim=dimension_of_optimization_variable)
    parameters_of_estimation = get_parameters_of_estimation()
    pac_bayes_parameters = get_pac_bayes_parameters()
    constraint = get_constraint(parameters_of_estimation=parameters_of_estimation,
                                loss_functions_for_constraint=loss_functions['validation'])
    stopping_criterion = get_stopping_criterion()
    algorithm_for_learning = PacBayesOptimizationAlgorithm(
        initial_state=initial_state,
        implementation=StochasticNnOptimizer(batch_size=get_batch_size()),
        stopping_criterion=stopping_criterion,
        loss_function=loss_functions['prior'][0],
        epsilon=pac_bayes_parameters['epsilon'],
        n_max=pac_bayes_parameters['n_max'],
        constraint=constraint
    )
    return algorithm_for_learning


def get_algorithm_for_initialization(initial_state_for_std_algorithm: torch.Tensor,
                                     loss_function: LossFunction) -> OptimizationAlgorithm:
    alpha = torch.tensor(1e-5)
    return OptimizationAlgorithm(initial_state=initial_state_for_std_algorithm,
                                 implementation=StochasticGradientDescent(step_size=alpha, batch_size=get_batch_size()),
                                 loss_function=loss_function)


def instantiate_neural_networks() -> Tuple[NeuralNetworkForStandardTraining, NeuralNetworkForLearning]:
    degree = torch.max(get_powers_of_polynomials()).item()
    neural_network_for_std_training = NeuralNetworkForStandardTraining(degree=degree)
    neural_network_for_learning = NeuralNetworkForLearning(
        degree=degree, shape_parameters=neural_network_for_std_training.get_shape_parameters())
    return neural_network_for_std_training, neural_network_for_learning


def set_up_and_train_algorithm(path_of_experiment: str) -> None:
    # This is pretty important! Without increased accuracy, the model will struggle to train, because at some point
    # (about loss of 1e-6) the incurred losses are subject to numerical instabilities, which do not provide meaningful
    # information for learning.
    torch.set_default_dtype(torch.double)

    neural_network_for_std_training, neural_network_for_learning = instantiate_neural_networks()
    loss_of_algorithm, single_loss_of_algorithm, loss_of_neural_network, parameters = get_data(
        neural_network=neural_network_for_learning, number_of_datapoints_per_dataset=get_number_of_datapoints())

    loss_functions = create_parametric_loss_functions_from_parameters(
        template_loss_function=loss_of_algorithm, template_single_loss=single_loss_of_algorithm, parameters=parameters)

    algorithm_for_learning = get_algorithm_for_learning(
        loss_functions=loss_functions,
        dimension_of_optimization_variable=neural_network_for_std_training.get_dimension_of_weights())
    algorithm_for_initialization = get_algorithm_for_initialization(
        initial_state_for_std_algorithm=algorithm_for_learning.initial_state[-1].reshape((1, -1)),
        loss_function=loss_functions['prior'][0]
    )
    algorithm_for_learning.initialize_with_other_algorithm(other_algorithm=algorithm_for_initialization,
                                                           loss_functions=loss_functions['prior'],
                                                           parameters_of_initialization=get_initialization_parameters())

    fitting_parameters = get_fitting_parameters(maximal_number_of_iterations=algorithm_for_learning.n_max)
    sampling_parameters = get_sampling_parameters(maximal_number_of_iterations=algorithm_for_learning.n_max)
    constraint_parameters = get_constraint_parameters(number_of_training_iterations=fitting_parameters['n_max'])
    update_parameters = get_update_parameters()
    (pac_bound_rate,
     pac_bound_conv_prob,
     pac_bound_time,
     state_dict_samples_prior) = algorithm_for_learning.pac_bayes_fit(
        loss_functions_prior=loss_functions['prior'],
        loss_functions_train=loss_functions['train'],
        fitting_parameters=fitting_parameters,
        sampling_parameters=sampling_parameters,
        constraint_parameters=constraint_parameters,
        update_parameters=update_parameters
    )

    savings_path = create_folder_for_storing_data(path_of_experiment)
    save_data(savings_path=savings_path,
              pac_bound_rate=pac_bound_rate.numpy(), pac_bound_conv_prob=pac_bound_conv_prob.numpy(),
              pac_bound_time=pac_bound_time.numpy(),
              upper_bound_rate=get_pac_bayes_parameters()['upper_bound'],
              upper_bound_time=algorithm_for_learning.n_max,
              initialization=algorithm_for_learning.initial_state.clone().numpy(),
              number_of_iterations=algorithm_for_learning.n_max, parameters=parameters,
              samples_prior=state_dict_samples_prior,
              best_sample=algorithm_for_learning.implementation.state_dict())


def save_data(savings_path: str,
              pac_bound_rate: NDArray,
              pac_bound_conv_prob: NDArray,
              pac_bound_time: NDArray,
              upper_bound_rate: float,
              upper_bound_time: int,
              initialization: NDArray,
              number_of_iterations: int,
              parameters: dict,
              samples_prior: List[dict],
              best_sample: dict) -> None:

    np.save(savings_path + 'pac_bound_rate', pac_bound_rate)
    np.save(savings_path + 'pac_bound_conv_prob', pac_bound_conv_prob)
    np.save(savings_path + 'pac_bound_time', pac_bound_time)
    np.save(savings_path + 'upper_bound_rate', upper_bound_rate)
    np.save(savings_path + 'upper_bound_time', upper_bound_time)
    np.save(savings_path + 'initialization', initialization)
    np.save(savings_path + 'number_of_iterations', number_of_iterations)
    with open(savings_path + 'parameters_problem', 'wb') as file:
        # noinspection PyTypeChecker
        pickle.dump(parameters, file)

    parameters_of_estimation = get_parameters_of_estimation()
    with open(savings_path + 'parameters_of_estimation', 'wb') as file:
        # noinspection PyTypeChecker
        pickle.dump(parameters_of_estimation, file)
    with open(savings_path + 'samples', 'wb') as file:
        # noinspection PyTypeChecker
        pickle.dump(samples_prior, file)
    with open(savings_path + 'best_sample', 'wb') as file:
        # noinspection PyTypeChecker
        pickle.dump(best_sample, file)
