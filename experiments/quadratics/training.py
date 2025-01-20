import torch
from typing import List, Callable, Tuple
from pathlib import Path
from classes.LossFunction.class_LossFunction import LossFunction
from classes.StoppingCriterion.derived_classes.subclass_LossCriterion import LossCriterion
from experiments.quadratics.data_generation import get_data
from classes.LossFunction.derived_classes.subclass_ParametricLossFunction import ParametricLossFunction
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm
from classes.OptimizationAlgorithm.derived_classes.derived_classes.subclass_PacBayesOptimizationAlgorithm import (
    PacBayesOptimizationAlgorithm)
from experiments.quadratics.algorithm import Quadratics
from exponential_family.describing_property.average_rate_property import get_rate_property
from algorithms.heavy_ball import HeavyBallWithFriction
from classes.Constraint.class_ProbabilisticConstraint import ProbabilisticConstraint
from classes.Constraint.class_Constraint import create_list_of_constraints_from_functions, Constraint
import numpy as np
from numpy.typing import NDArray
import pickle


def get_number_of_datapoints() -> dict:
    return {'prior': 250, 'train': 250, 'test': 250, 'validation': 250}


def create_folder_for_storing_data(path_of_experiment: str) -> str:
    savings_path = path_of_experiment + "/data/"
    Path(savings_path).mkdir(parents=True, exist_ok=True)
    return savings_path


def create_parametric_loss_functions_from_parameters(template_loss_function: Callable, parameters: dict) -> dict:
    loss_functions = {
        'prior': [ParametricLossFunction(function=template_loss_function, parameter=p) for p in parameters['prior']],
        'train': [ParametricLossFunction(function=template_loss_function, parameter=p) for p in parameters['train']],
        'test': [ParametricLossFunction(function=template_loss_function, parameter=p) for p in parameters['test']],
        'validation': [ParametricLossFunction(function=template_loss_function, parameter=p)
                       for p in parameters['validation']],
    }
    return loss_functions


def get_initial_state(dim: int) -> torch.Tensor:
    return torch.zeros((2, dim))


def get_parameters_of_estimation() -> dict:
    return {'quantile_distance': 0.075, 'quantiles': (0.01, 0.99), 'probabilities': (0.95, 1.0)}


def get_update_parameters() -> dict:
    return {'num_iter_print_update': 1000,
            'with_print': True,
            'bins': [1e6, 1e4, 1e2, 1e0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10][::-1]}


def get_sampling_parameters(maximal_number_of_iterations: int) -> dict:
    length_trajectory = 1
    restart_probability = length_trajectory / maximal_number_of_iterations
    return {'lr': torch.tensor(1e-6),
            'length_trajectory': length_trajectory,
            'with_restarting': True,
            'restart_probability': restart_probability,
            'num_samples': 1,
            'num_iter_burnin': 0}


def get_fitting_parameters(maximal_number_of_iterations: int) -> dict:
    length_trajectory = 1
    restart_probability = length_trajectory / maximal_number_of_iterations
    return {'restart_probability': restart_probability,
            'length_trajectory': length_trajectory,
            # TODO: Rename n_max to number_of_training_iterations
            'n_max': int(300e3),
            'lr': 1e-4,
            'num_iter_update_stepsize': int(30e3),
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
            # TODO: Rename n_max to maximal_number_of_iterations
            'upper_bound': 0.99,
            'n_max': 500}


def get_constraint(parameters_of_estimation: dict, loss_functions_for_constraint: List[LossFunction]) -> Constraint:
    describing_property, _ = get_describing_property()
    list_of_constraints = create_list_of_constraints_from_functions(describing_property=describing_property,
                                                                    list_of_functions=loss_functions_for_constraint)
    probabilistic_constraint = ProbabilisticConstraint(list_of_constraints=list_of_constraints,
                                                       parameters_of_estimation=parameters_of_estimation)
    return probabilistic_constraint.create_constraint()


def get_stopping_criterion():
    return LossCriterion(threshold=1e-8)


def get_algorithm_for_learning(loss_functions: dict,
                               dimension_of_optimization_variable: int) -> PacBayesOptimizationAlgorithm:

    initial_state = get_initial_state(dim=dimension_of_optimization_variable)
    parameters_of_estimation = get_parameters_of_estimation()
    pac_bayes_parameters = get_pac_bayes_parameters()
    constraint = get_constraint(
        parameters_of_estimation=parameters_of_estimation,
        loss_functions_for_constraint=loss_functions['validation']
    )
    stopping_criterion = get_stopping_criterion()
    algorithm_for_learning = PacBayesOptimizationAlgorithm(
        initial_state=initial_state,
        implementation=Quadratics(dim=dimension_of_optimization_variable),
        stopping_criterion=stopping_criterion,
        loss_function=loss_functions['prior'][0],
        epsilon=pac_bayes_parameters['epsilon'],
        n_max=pac_bayes_parameters['n_max'],
        constraint=constraint
    )
    return algorithm_for_learning


def get_baseline_algorithm(loss_function: LossFunction,
                           smoothness_constant: torch.Tensor,
                           strong_convexity_constant: torch.Tensor,
                           dim: int) -> OptimizationAlgorithm:

    initial_state = get_initial_state(dim=dim)
    alpha = 4/((torch.sqrt(smoothness_constant) + torch.sqrt(strong_convexity_constant)) ** 2)
    beta = ((torch.sqrt(smoothness_constant) - torch.sqrt(strong_convexity_constant)) /
            (torch.sqrt(smoothness_constant) + torch.sqrt(strong_convexity_constant))) ** 2
    stopping_criterion = get_stopping_criterion()
    std_algo = OptimizationAlgorithm(
        initial_state=initial_state,
        implementation=HeavyBallWithFriction(alpha=alpha, beta=beta),
        loss_function=loss_function,
        stopping_criterion=stopping_criterion)
    return std_algo


def set_up_and_train_algorithm(path_of_experiment: str) -> None:
    # This is pretty important! Without increased accuracy, the model will struggle to train, because at some point
    # (about loss of 1e-6) the incurred losses are subject to numerical instabilities, which do not provide meaningful
    # information for learning.
    torch.set_default_dtype(torch.double)

    parameters, loss_function_of_algorithm, mu_min, L_max, dim = get_data(get_number_of_datapoints())
    loss_functions = create_parametric_loss_functions_from_parameters(
        template_loss_function=loss_function_of_algorithm, parameters=parameters)
    baseline_algorithm = get_baseline_algorithm(
        loss_function=loss_functions['prior'][0], smoothness_constant=L_max, strong_convexity_constant=mu_min, dim=dim)
    algorithm_for_learning = get_algorithm_for_learning(loss_functions=loss_functions,
                                                        dimension_of_optimization_variable=dim)
    algorithm_for_learning.initialize_with_other_algorithm(other_algorithm=baseline_algorithm,
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
    save_data(savings_path=savings_path, strong_convexity_parameter=mu_min.numpy(), smoothness_parameter=L_max.numpy(),
              pac_bound_rate=pac_bound_rate.numpy(), pac_bound_conv_prob=pac_bound_conv_prob.numpy(),
              pac_bound_time=pac_bound_time.numpy(),
              upper_bound_rate=get_pac_bayes_parameters()['bound'],
              upper_bound_time=algorithm_for_learning.n_max,
              initialization=algorithm_for_learning.initial_state.clone().numpy(),
              number_of_iterations=algorithm_for_learning.n_max, parameters=parameters,
              samples_prior=state_dict_samples_prior,
              best_sample=algorithm_for_learning.implementation.state_dict())


def save_data(savings_path: str,
              strong_convexity_parameter: NDArray,
              smoothness_parameter: NDArray,
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

    np.save(savings_path + 'strong_convexity_parameter', strong_convexity_parameter)
    np.save(savings_path + 'smoothness_parameter', smoothness_parameter)
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
