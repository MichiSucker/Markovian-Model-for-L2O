from pathlib import Path
from typing import List, Tuple, Callable
from numpy.typing import NDArray
import numpy as np
import pickle
from tqdm import tqdm
from experiments.neural_network_full_batch.neural_network import train_model, NeuralNetworkForStandardTraining
from experiments.neural_network_full_batch.training import instantiate_neural_networks, get_stopping_criterion
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm
from experiments.neural_network_full_batch.algorithm import NnOptimizer
import torch
from experiments.neural_network_full_batch.data_generation import get_loss_of_algorithm, get_loss_of_neural_network
from classes.LossFunction.derived_classes.subclass_ParametricLossFunction import ParametricLossFunction
from experiments.neural_network_full_batch.training import get_describing_property


class EvaluationAssistant:

    def __init__(self,
                 test_set: List,
                 number_of_iterations_during_training: int,
                 number_of_iterations_for_testing: int,
                 loss_of_algorithm: Callable,
                 initial_state: torch.Tensor,
                 optimal_hyperparameters: dict,
                 implementation_class: Callable):
        self.test_set = test_set
        self.number_of_iterations_during_training = number_of_iterations_during_training
        self.number_of_iterations_for_testing = number_of_iterations_for_testing
        self.loss_of_algorithm = loss_of_algorithm
        self.initial_state = initial_state
        self.dim = initial_state.shape[1]
        self.optimal_hyperparameters = optimal_hyperparameters
        self.implementation_class = implementation_class
        self.loss_of_neural_network = None
        self.implementation_arguments = None
        self.lr_adam = None

    def set_up_learned_algorithm(self) -> OptimizationAlgorithm:
        stopping_criterion = get_stopping_criterion()
        learned_algorithm = OptimizationAlgorithm(
            implementation=NnOptimizer(dim=self.dim),
            initial_state=self.initial_state,
            loss_function=ParametricLossFunction(function=self.loss_of_algorithm,
                                                 parameter=self.test_set[0]),
            stopping_criterion=stopping_criterion
        )
        learned_algorithm.implementation.load_state_dict(self.optimal_hyperparameters)

        return learned_algorithm


def load_data(loading_path: str) -> Tuple:
    initial_state = np.load(loading_path + 'initialization.npy')
    n_train = np.load(loading_path + 'number_of_iterations.npy')
    with open(loading_path + 'parameters_problem', 'rb') as file:
        parameters = pickle.load(file)
    with open(loading_path + 'samples', 'rb') as file:
        samples = pickle.load(file)
    with open(loading_path + 'best_sample', 'rb') as file:
        best_sample = pickle.load(file)
    return initial_state, n_train, parameters, samples, best_sample


def create_folder_for_storing_data(path_of_experiment: str) -> str:
    savings_path = path_of_experiment + "/data/"
    Path(savings_path).mkdir(parents=True, exist_ok=True)
    return savings_path


def compute_ground_truth_loss(loss_of_neural_network: Callable,
                              parameter: dict) -> torch.Tensor:
    return loss_of_neural_network(parameter['ground_truth_values'], parameter['y_values'])


def compute_losses_over_iterations_and_stopping_time(learned_algorithm: OptimizationAlgorithm,
                                                     evaluation_assistant: EvaluationAssistant,
                                                     parameter: dict) -> Tuple[List[float], int]:

    learned_algorithm.reset_state_and_iteration_counter()
    current_loss_function = ParametricLossFunction(function=evaluation_assistant.loss_of_algorithm, parameter=parameter)
    learned_algorithm.set_loss_function(current_loss_function)
    loss_over_iterations = [learned_algorithm.evaluate_loss_function_at_current_iterate().item()]
    stopping_time = evaluation_assistant.number_of_iterations_during_training
    number_of_iterations = evaluation_assistant.number_of_iterations_for_testing

    for i in range(evaluation_assistant.number_of_iterations_for_testing):

        if learned_algorithm.stopping_criterion(learned_algorithm):
            loss_over_iterations.extend([loss_over_iterations[-1]] * (number_of_iterations - i))
            stopping_time = np.minimum(i, evaluation_assistant.number_of_iterations_during_training)
            break

        learned_algorithm.perform_step()
        loss_over_iterations.append(learned_algorithm.evaluate_loss_function_at_current_iterate().item())

    return loss_over_iterations, int(stopping_time)


def does_satisfy_constraint(convergence_risk_constraint: Callable,
                            loss_at_beginning: float,
                            loss_at_end: float,
                            convergence_time: int) -> bool:
    return convergence_risk_constraint(final_loss=loss_at_end, init_loss=loss_at_beginning, conv_time=convergence_time)


def compute_losses_over_iterations_for_adam(neural_network: NeuralNetworkForStandardTraining,
                                            evaluation_assistant: EvaluationAssistant,
                                            parameter: dict) -> List[float]:
    neural_network_for_standard_training, losses_over_iterations_of_adam, _ = train_model(
        net=neural_network, data=parameter, criterion=evaluation_assistant.loss_of_neural_network,
        n_it=evaluation_assistant.number_of_iterations_for_testing, lr=evaluation_assistant.lr_adam
    )
    return losses_over_iterations_of_adam


def compute_rate(loss_at_beginning: float, loss_at_end: float, stopping_time: int) -> float:
    return (loss_at_end/loss_at_beginning) ** (1/stopping_time)


def compute_losses(evaluation_assistant: EvaluationAssistant,
                   learned_algorithm: OptimizationAlgorithm,
                   neural_network_for_standard_training: NeuralNetworkForStandardTraining
                   ) -> Tuple[NDArray, NDArray, NDArray, NDArray, torch.Tensor, float]:

    ground_truth_losses = []
    losses_of_adam = []
    losses_of_learned_algorithm, stopping_times_learned_algorithm, rates_learned_algorithm = [], [], []
    number_of_times_constrained_satisfied = 0
    _, convergence_risk_constraint = get_describing_property()

    pbar = tqdm(evaluation_assistant.test_set)
    pbar.set_description("Compute losses")
    for test_parameter in pbar:

        loss_over_iterations, stopping_time = compute_losses_over_iterations_and_stopping_time(
            learned_algorithm=learned_algorithm, evaluation_assistant=evaluation_assistant, parameter=test_parameter)

        if does_satisfy_constraint(
                convergence_risk_constraint=convergence_risk_constraint, loss_at_beginning=loss_over_iterations[0],
                loss_at_end=loss_over_iterations[evaluation_assistant.number_of_iterations_during_training],
                convergence_time=stopping_time):

            number_of_times_constrained_satisfied += 1
            losses_of_learned_algorithm.append(loss_over_iterations)
            stopping_times_learned_algorithm.append(stopping_time)
            rate = compute_rate(loss_at_beginning=loss_over_iterations[0],
                                loss_at_end=loss_over_iterations[stopping_time],
                                stopping_time=stopping_time)
            rates_learned_algorithm.append(rate)
            ground_truth_losses.append(compute_ground_truth_loss(evaluation_assistant.loss_of_neural_network,
                                                                 test_parameter))

            neural_network_for_standard_training.load_parameters_from_tensor(
                learned_algorithm.initial_state[-1].clone())
            losses_of_adam.append(compute_losses_over_iterations_for_adam(
                neural_network=neural_network_for_standard_training, evaluation_assistant=evaluation_assistant,
                parameter=test_parameter))

    return (np.array(losses_of_adam),
            np.array(losses_of_learned_algorithm),
            np.array(rates_learned_algorithm),
            np.array(stopping_times_learned_algorithm),
            torch.tensor(ground_truth_losses),
            number_of_times_constrained_satisfied / len(evaluation_assistant.test_set))


def set_up_evaluation_assistant(loading_path: str) -> Tuple[EvaluationAssistant, NeuralNetworkForStandardTraining]:
    initial_state, n_train, parameters, samples, best_sample = load_data(loading_path)
    neural_network_for_standard_training, neural_network_for_learning = instantiate_neural_networks()
    loss_of_neural_network = get_loss_of_neural_network()
    loss_of_algorithm = get_loss_of_algorithm(neural_network=neural_network_for_learning,
                                              loss_of_neural_network=loss_of_neural_network)

    evaluation_assistant = EvaluationAssistant(
        test_set=parameters['test'], number_of_iterations_during_training=n_train,
        number_of_iterations_for_testing=2*n_train,
        loss_of_algorithm=loss_of_algorithm, initial_state=torch.tensor(initial_state),
        optimal_hyperparameters=best_sample, implementation_class=NnOptimizer)
    evaluation_assistant.loss_of_neural_network = loss_of_neural_network
    evaluation_assistant.implementation_arguments = (
        {'dim': neural_network_for_standard_training.get_dimension_of_weights()})
    evaluation_assistant.lr_adam = 0.008  # Originally, this was found by gridsearch.
    return evaluation_assistant, neural_network_for_standard_training


def evaluate_algorithm(loading_path: str, path_of_experiment: str) -> None:

    evaluation_assistant, neural_network_for_standard_training = set_up_evaluation_assistant(loading_path)
    learned_algorithm = evaluation_assistant.set_up_learned_algorithm()

    (losses_of_adam,
     losses_of_learned_algorithm,
     rates_of_learned_algorithm,
     stopping_times_of_learned_algorithm,
     ground_truth_losses,
     percentage_constrained_satisfied) = compute_losses(
        evaluation_assistant=evaluation_assistant, learned_algorithm=learned_algorithm,
        neural_network_for_standard_training=neural_network_for_standard_training
    )

    save_data(savings_path=create_folder_for_storing_data(path_of_experiment),
              losses_of_learned_algorithm=losses_of_learned_algorithm,
              rates_of_learned_algorithm=rates_of_learned_algorithm,
              stopping_times_of_learned_algorithm=stopping_times_of_learned_algorithm,
              losses_of_adam=losses_of_adam,
              ground_truth_losses=ground_truth_losses.numpy(),
              percentage_constrained_satisfied=percentage_constrained_satisfied)


def save_data(savings_path: str,
              losses_of_learned_algorithm: NDArray,
              rates_of_learned_algorithm: NDArray,
              stopping_times_of_learned_algorithm: NDArray,
              losses_of_adam: NDArray,
              ground_truth_losses: NDArray,
              percentage_constrained_satisfied: float):

    np.save(savings_path + 'losses_of_adam', losses_of_adam)
    np.save(savings_path + 'losses_of_learned_algorithm', losses_of_learned_algorithm)
    np.save(savings_path + 'rates_of_learned_algorithm', rates_of_learned_algorithm)
    np.save(savings_path + 'stopping_times_of_learned_algorithm', stopping_times_of_learned_algorithm)
    np.save(savings_path + 'ground_truth_losses', ground_truth_losses)
    np.save(savings_path + 'empirical_probability', percentage_constrained_satisfied)
