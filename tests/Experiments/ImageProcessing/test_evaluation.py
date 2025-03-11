import unittest
import torch
from typing import Callable
from experiments.image_processing.training import get_initial_states, get_describing_property, get_baseline_algorithm
from experiments.image_processing.evaluation import (set_up_evaluation_assistant,
                                                     EvaluationAssistant,
                                                     does_satisfy_constraint,
                                                     compute_rate,
                                                     compute_losses_rates_and_convergence_time,
                                                     compute_losses_over_iterations_and_stopping_time)


class TestEvaluationImageProcessing(unittest.TestCase):

    def setUp(self):
        # Make sure that all tensors are of the same type.
        torch.set_default_dtype(torch.double)
        self.path_to_experiment = '/home/michael/Desktop/JMLR_Markovian_Model/new_implementation/image_processing/'
        self.dummy_savings_path = self.path_to_experiment + 'dummy_data/'
        self.loading_path = self.path_to_experiment + 'data/'

    def test_set_up_evaluation_assistant(self):
        # Check that we get an EvaluationAssistant with the correct implementation.
        eval_assist = set_up_evaluation_assistant(loading_path=self.loading_path)
        self.assertIsInstance(eval_assist, EvaluationAssistant)
        self.assertIsInstance(eval_assist.loss_of_algorithm, Callable)
        self.assertIsInstance(eval_assist.data_fidelity, Callable)
        self.assertIsInstance(eval_assist.regularization, Callable)
        self.assertIsInstance(eval_assist.initial_state_learned, torch.Tensor)
        self.assertIsInstance(eval_assist.initial_state_baseline, torch.Tensor)
        _, init = get_initial_states()
        # The first one gets randomly generated, so we cannot check for equality
        self.assertEqual(init.shape, eval_assist.initial_state_learned.shape)
        self.assertTrue(
            2 * eval_assist.number_of_iterations_during_training == eval_assist.number_of_iterations_for_testing)
        self.assertIsInstance(eval_assist.optimal_hyperparameters, dict)

    def test_does_satisfy_constraint(self):

        _, convergence_risk_constraint = get_describing_property()

        self.assertTrue(does_satisfy_constraint(convergence_risk_constraint=convergence_risk_constraint,
                                                loss_at_beginning=10, loss_at_end=1, convergence_time=2))
        self.assertFalse(does_satisfy_constraint(convergence_risk_constraint=convergence_risk_constraint,
                                                 loss_at_beginning=1, loss_at_end=10, convergence_time=2))

    def test_compute_rate(self):

        dist = torch.distributions.uniform.Uniform(0, 10)
        loss_at_beginning = dist.sample((1,)).item()
        loss_at_end = dist.sample((1,)).item()
        stopping_time = torch.randint(low=1, high=20, size=(1,)).item()
        rate = compute_rate(loss_at_beginning=loss_at_beginning, loss_at_end=loss_at_end, stopping_time=stopping_time)
        self.assertEqual(rate, (loss_at_end/loss_at_beginning) ** (1/stopping_time))

    def test_compute_losses(self):
        # Check that we get the same amount of values for both algorithms.
        # Also check that percentage lies in [0,1].
        eval_assist = set_up_evaluation_assistant(loading_path=self.loading_path)
        eval_assist.number_of_iterations_during_training = 10
        eval_assist.number_of_iterations_for_testing = 20
        eval_assist.test_set = eval_assist.test_set[0:2]
        learned_algorithm = eval_assist.set_up_learned_algorithm()
        init, _ = get_initial_states()
        baseline_algorithm = get_baseline_algorithm(loss_function_of_algorithm=eval_assist.loss_of_algorithm,
                                                    smoothness_parameter=eval_assist.smoothness_parameter)
        (losses_of_baseline_algorithm,
         rates_of_baseline_algorithm,
         times_of_baseline_algorithm,
         losses_of_learned_algorithm,
         rates_of_learned_algorithm,
         times_of_learned_algorithm,
         percentage) = (
            compute_losses_rates_and_convergence_time(
                evaluation_assistant=eval_assist, learned_algorithm=learned_algorithm,
                baseline_algorithm=baseline_algorithm))

        self.assertEqual(losses_of_baseline_algorithm.shape, losses_of_learned_algorithm.shape)
        self.assertEqual(rates_of_baseline_algorithm.shape, rates_of_learned_algorithm.shape)
        self.assertEqual(times_of_baseline_algorithm.shape, times_of_learned_algorithm.shape)

        self.assertEqual(losses_of_baseline_algorithm.shape[0], len(eval_assist.test_set[0:2]))
        self.assertEqual(losses_of_baseline_algorithm.shape[1], eval_assist.number_of_iterations_for_testing + 1)

        self.assertEqual(len(rates_of_learned_algorithm), len(eval_assist.test_set[0:2]))
        self.assertEqual(len(times_of_learned_algorithm), len(eval_assist.test_set[0:2]))

        self.assertTrue(0 <= percentage <= 1)

    def test_compute_losses_over_iterations(self):
        # Check that we get N+1 values, where N is the number of iterations for testing.
        # +1 because we also want the initial loss => Check that.
        eval_assist = set_up_evaluation_assistant(loading_path=self.loading_path)
        eval_assist.test_set = eval_assist.test_set[0:2]
        algorithm = eval_assist.set_up_learned_algorithm()
        losses, conv_time = compute_losses_over_iterations_and_stopping_time(algorithm=algorithm,
                                                                             evaluation_assistant=eval_assist,
                                                                             parameter=eval_assist.test_set[0])
        cur_optimal_loss = eval_assist.test_set[0]['optimal_loss']
        self.assertTrue(len(losses), eval_assist.number_of_iterations_for_testing + 1)
        self.assertEqual(losses[0],
                         eval_assist.loss_of_algorithm(eval_assist.initial_state_learned[-1],
                                                       eval_assist.test_set[0]) - cur_optimal_loss)
        self.assertTrue(0 <= conv_time <= eval_assist.number_of_iterations_during_training)
