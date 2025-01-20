import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Callable
import torch
import pickle
from pathlib import Path
from tqdm import tqdm

from classes.LossFunction.derived_classes.subclass_ParametricLossFunction import ParametricLossFunction
from experiments.quadratics.algorithm import Quadratics
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm
from experiments.quadratics.data_generation import get_loss_function_of_algorithm
from experiments.quadratics.training import get_describing_property, get_baseline_algorithm, get_stopping_criterion


class EvaluationAssistant:

    def __init__(self,
                 test_set: List,
                 loss_of_algorithm: Callable,
                 initial_state: torch.Tensor,
                 number_of_iterations_during_training: int,
                 optimal_hyperparameters: dict,
                 implementation_class: Callable):
        self.test_set = test_set
        self.initial_state = initial_state
        self.number_of_iterations_during_training = number_of_iterations_during_training
        self.number_of_iterations_for_testing = 2 * number_of_iterations_during_training
        self.loss_of_algorithm = loss_of_algorithm
        self.optimal_hyperparameters = optimal_hyperparameters
        self.implementation_class = implementation_class
        self.dim = initial_state.shape[1]
        self.implementation_arguments = None
        self.strong_convexity_parameter = None
        self.smoothness_parameter = None

    def set_up_learned_algorithm(self) -> OptimizationAlgorithm:
        stopping_criterion = get_stopping_criterion()
        learned_algorithm = OptimizationAlgorithm(
            implementation=Quadratics(dim=self.dim),
            initial_state=self.initial_state,
            loss_function=ParametricLossFunction(function=self.loss_of_algorithm,
                                                 parameter=self.test_set[0]),
            stopping_criterion=stopping_criterion
        )
        learned_algorithm.implementation.load_state_dict(self.optimal_hyperparameters)
        return learned_algorithm


def load_data(loading_path: str) -> Tuple:
    pac_bound_rate = np.load(loading_path + 'pac_bound_rate.npy')
    pac_bound_time = np.load(loading_path + 'pac_bound_time.npy')
    pac_bound_conv_prob = np.load(loading_path + 'pac_bound_conv_prob.npy')
    initial_state = torch.tensor(np.load(loading_path + 'initialization.npy'))
    n_train = np.load(loading_path + 'number_of_iterations.npy')
    smoothness_parameter = np.load(loading_path + 'smoothness_parameter.npy')
    strong_convexity_parameter = np.load(loading_path + 'strong_convexity_parameter.npy')
    with open(loading_path + 'parameters_problem', 'rb') as file:
        parameters = pickle.load(file)
    with open(loading_path + 'samples', 'rb') as file:
        samples = pickle.load(file)
    with open(loading_path + 'best_sample', 'rb') as file:
        best_sample = pickle.load(file)
    return (pac_bound_rate, pac_bound_time, pac_bound_conv_prob, initial_state, n_train, parameters, samples,
            best_sample, strong_convexity_parameter, smoothness_parameter)


def create_folder_for_storing_data(path_of_experiment: str) -> str:
    savings_path = path_of_experiment + "/data/"
    Path(savings_path).mkdir(parents=True, exist_ok=True)
    return savings_path


def save_data(savings_path: str,
              losses_of_learned_algorithm: NDArray,
              rates_of_learned_algorithm: NDArray,
              times_of_learned_algorithm: NDArray,
              losses_of_baseline_algorithm: NDArray,
              rates_of_baseline_algorithm: NDArray,
              times_of_baseline_algorithm: NDArray,
              ground_truth_losses: List,
              percentage_constrained_satisfied: float) -> None:

    np.save(savings_path + 'losses_of_baseline_algorithm', np.array(losses_of_baseline_algorithm))
    np.save(savings_path + 'rates_of_baseline_algorithm', np.array(rates_of_baseline_algorithm))
    np.save(savings_path + 'times_of_baseline_algorithm', np.array(times_of_baseline_algorithm))

    np.save(savings_path + 'losses_of_learned_algorithm', np.array(losses_of_learned_algorithm))
    np.save(savings_path + 'rates_of_learned_algorithm', np.array(rates_of_learned_algorithm))
    np.save(savings_path + 'times_of_learned_algorithm', np.array(times_of_learned_algorithm))

    np.save(savings_path + 'ground_truth_losses', np.array(ground_truth_losses))
    np.save(savings_path + 'empirical_probability', percentage_constrained_satisfied)


def set_up_evaluation_assistant(loading_path: str) -> EvaluationAssistant:
    (_, _, _, initial_state, n_train, parameters, samples, best_sample, strong_convexity_parameter,
     smoothness_parameter) = load_data(loading_path)
    loss_of_algorithm = get_loss_function_of_algorithm()

    evaluation_assistant = EvaluationAssistant(test_set=parameters['test'], loss_of_algorithm=loss_of_algorithm,
                                               initial_state=initial_state,
                                               number_of_iterations_during_training=n_train,
                                               optimal_hyperparameters=best_sample, implementation_class=Quadratics)
    evaluation_assistant.smoothness_parameter = torch.tensor(smoothness_parameter)
    evaluation_assistant.strong_convexity_parameter = torch.tensor(strong_convexity_parameter)
    return evaluation_assistant


def does_satisfy_constraint(convergence_risk_constraint: Callable,
                            loss_at_beginning: float,
                            loss_at_end: float,
                            convergence_time: int) -> bool:
    return convergence_risk_constraint(final_loss=loss_at_end, init_loss=loss_at_beginning, conv_time=convergence_time)


def compute_rate(loss_at_beginning: float, loss_at_end: float, stopping_time: int) -> float:
    return (loss_at_end/loss_at_beginning) ** (1/stopping_time)


def compute_losses_rates_and_convergence_time(evaluation_assistant: EvaluationAssistant,
                                              learned_algorithm: OptimizationAlgorithm,
                                              baseline_algorithm: OptimizationAlgorithm
                                              ) -> Tuple[NDArray, NDArray, NDArray, NDArray, NDArray, NDArray, float]:

    losses_of_baseline_algorithm, rates_of_baseline_algorithm, convergence_time_of_baseline_algorithm = [], [], []
    losses_of_learned_algorithm, rates_of_learned_algorithm, convergence_time_of_learned_algorithm = [], [], []
    number_of_times_constrained_satisfied = 0
    _, convergence_risk_constraint = get_describing_property()

    progress_bar = tqdm(evaluation_assistant.test_set)
    progress_bar.set_description('Compute losses')
    for test_parameter in progress_bar:

        loss_over_iterations, convergence_time = compute_losses_over_iterations_and_stopping_time(
            algorithm=learned_algorithm, evaluation_assistant=evaluation_assistant, parameter=test_parameter)

        if does_satisfy_constraint(
                convergence_risk_constraint=convergence_risk_constraint, loss_at_beginning=loss_over_iterations[0],
                loss_at_end=loss_over_iterations[evaluation_assistant.number_of_iterations_during_training],
                convergence_time=convergence_time):

            # Compute rate for learned algorithm and append everything
            rate = compute_rate(loss_at_beginning=loss_over_iterations[0],
                                loss_at_end=loss_over_iterations[convergence_time],
                                stopping_time=convergence_time)
            number_of_times_constrained_satisfied += 1
            losses_of_learned_algorithm.append(loss_over_iterations)
            rates_of_learned_algorithm.append(rate)
            convergence_time_of_learned_algorithm.append(convergence_time)

            # Compute losses, convergence time, and rate for baseline algorithm
            loss_over_iterations, convergence_time = compute_losses_over_iterations_and_stopping_time(
                algorithm=baseline_algorithm, evaluation_assistant=evaluation_assistant, parameter=test_parameter)
            losses_of_baseline_algorithm.append(loss_over_iterations)
            convergence_time_of_baseline_algorithm.append(convergence_time)
            rate = compute_rate(loss_at_beginning=loss_over_iterations[0],
                                loss_at_end=loss_over_iterations[convergence_time],
                                stopping_time=convergence_time)
            rates_of_baseline_algorithm.append(rate)

    return (np.array(losses_of_baseline_algorithm),
            np.array(rates_of_baseline_algorithm),
            np.array(convergence_time_of_baseline_algorithm),
            np.array(losses_of_learned_algorithm),
            np.array(rates_of_learned_algorithm),
            np.array(convergence_time_of_learned_algorithm),
            number_of_times_constrained_satisfied / len(evaluation_assistant.test_set))


def compute_losses_over_iterations_and_stopping_time(algorithm: OptimizationAlgorithm,
                                                     evaluation_assistant: EvaluationAssistant,
                                                     parameter: dict) -> Tuple[List[float], int]:
    algorithm.reset_state_and_iteration_counter()
    current_loss_function = ParametricLossFunction(function=evaluation_assistant.loss_of_algorithm, parameter=parameter)
    algorithm.set_loss_function(current_loss_function)
    loss_over_iterations = [algorithm.evaluate_loss_function_at_current_iterate().item()]
    stopping_time = evaluation_assistant.number_of_iterations_during_training
    number_of_iterations = evaluation_assistant.number_of_iterations_for_testing
    for i in range(number_of_iterations):

        if algorithm.stopping_criterion(algorithm):
            loss_over_iterations.extend([loss_over_iterations[-1]] * (number_of_iterations - i))
            stopping_time = np.minimum(i, evaluation_assistant.number_of_iterations_during_training)
            break

        algorithm.perform_step()
        loss_over_iterations.append(algorithm.evaluate_loss_function_at_current_iterate().item())

    return loss_over_iterations, stopping_time


def sanity_check_for_baseline(algorithm: OptimizationAlgorithm, savings_path: str) -> None:

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    iterations = torch.arange(2e4 + 1)
    losses = [algorithm.evaluate_loss_function_at_current_iterate().item()]
    for _ in range(len(iterations) - 1):
        algorithm.perform_step()
        losses.append(algorithm.evaluate_loss_function_at_current_iterate().item())

    losses = np.array(losses)
    ax.plot(iterations.numpy(), losses)
    ax.set_yscale('log')
    ax.set_ylabel('loss')
    ax.set_xlabel('iteration')
    ax.grid('on')

    fig.savefig(savings_path + 'sanity_check.pdf', dpi=300, bbox_inches='tight')


def evaluate_algorithm(loading_path: str, path_of_experiment: str) -> None:
    savings_path = create_folder_for_storing_data(path_of_experiment)
    evaluation_assistant = set_up_evaluation_assistant(loading_path)
    learned_algorithm = evaluation_assistant.set_up_learned_algorithm()
    baseline_algorithm = get_baseline_algorithm(
        loss_function=learned_algorithm.loss_function, smoothness_constant=evaluation_assistant.smoothness_parameter,
        strong_convexity_constant=evaluation_assistant.strong_convexity_parameter, dim=evaluation_assistant.dim)

    sanity_check_for_baseline(algorithm=baseline_algorithm, savings_path=savings_path)

    (losses_of_baseline,
     rates_of_baseline_algorithm,
     times_of_baseline_algorithm,
     losses_of_learned_algorithm,
     rates_of_learned_algorithm,
     times_of_learned_algorithm,
     percentage_constrained_satisfied) = compute_losses_rates_and_convergence_time(
        evaluation_assistant=evaluation_assistant, learned_algorithm=learned_algorithm,
        baseline_algorithm=baseline_algorithm
    )

    save_data(savings_path=savings_path,
              losses_of_learned_algorithm=losses_of_learned_algorithm,
              times_of_learned_algorithm=times_of_learned_algorithm,
              rates_of_learned_algorithm=rates_of_learned_algorithm,
              losses_of_baseline_algorithm=losses_of_baseline,
              rates_of_baseline_algorithm=rates_of_baseline_algorithm,
              times_of_baseline_algorithm=times_of_baseline_algorithm,
              ground_truth_losses=[0. for _ in range(len(evaluation_assistant.test_set))],
              percentage_constrained_satisfied=percentage_constrained_satisfied)
