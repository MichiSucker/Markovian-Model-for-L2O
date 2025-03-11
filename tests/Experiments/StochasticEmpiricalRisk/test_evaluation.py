import unittest
import torch
import torch.nn as nn
from experiments.neural_network_stochastic.training import get_describing_property
from experiments.neural_network_stochastic.evaluation import (compute_ground_truth_loss,
                                                              compute_losses_over_iterations_and_stopping_time,
                                                              set_up_evaluation_assistant,
                                                              does_satisfy_constraint,
                                                              compute_losses_over_iterations_for_adam,
                                                              compute_rate,
                                                              compute_losses,
                                                              EvaluationAssistant)


class TestEvaluationStochasticERM(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.double)
        self.path_to_experiment = ('/home/michael/Desktop/JMLR_Markovian_Model/'
                                   'new_implementation/stochastic_neural_network_training/')
        self.dummy_savings_path = self.path_to_experiment + 'dummy_data/'
        self.loading_path = self.path_to_experiment + 'data/'

    def test_compute_ground_truth_loss(self):
        # Check that the ground-truth loss is computed with MSE.
        criterion = nn.MSELoss()
        parameter = {'ground_truth_values': torch.rand((10,)), 'y_values': torch.rand((10,))}
        gt_loss = compute_ground_truth_loss(loss_of_neural_network=criterion, parameter=parameter)
        self.assertEqual(gt_loss, criterion(parameter['ground_truth_values'], parameter['y_values']))

    def test_compute_losses_of_learned_algorithm(self):
        # Check that we get N+1 values, where N is the number of iterations for testing.
        # +1 because we want the initial loss.
        eval_assist, neural_network = set_up_evaluation_assistant(loading_path=self.loading_path)
        eval_assist.number_of_iterations_for_testing = 100
        learned_algorithm = eval_assist.set_up_learned_algorithm()
        losses, stopping_time = compute_losses_over_iterations_and_stopping_time(learned_algorithm=learned_algorithm,
                                                                                 evaluation_assistant=eval_assist,
                                                                                 parameter=eval_assist.test_set[0])
        self.assertTrue(len(losses), eval_assist.number_of_iterations_for_testing + 1)
        self.assertEqual(losses[0], eval_assist.loss_of_algorithm(eval_assist.initial_state[-1],
                                                                  eval_assist.test_set[0]))
        self.assertIsInstance(stopping_time, int)
        self.assertTrue(0 <= stopping_time <= eval_assist.number_of_iterations_during_training)

    def test_does_satisfy_constraint(self):

        _, convergence_risk_constraint = get_describing_property()

        self.assertTrue(does_satisfy_constraint(convergence_risk_constraint=convergence_risk_constraint,
                                                loss_at_beginning=10, loss_at_end=1, convergence_time=2))
        self.assertFalse(does_satisfy_constraint(convergence_risk_constraint=convergence_risk_constraint,
                                                 loss_at_beginning=1, loss_at_end=10, convergence_time=2))

    def test_compute_losses_of_adam(self):
        # Basically the same test as for the learned algorithm.
        eval_assist, neural_network = set_up_evaluation_assistant(loading_path=self.loading_path)
        eval_assist.number_of_iterations_for_testing = 100
        neural_network.load_parameters_from_tensor(eval_assist.initial_state[-1].clone())
        losses_adam = compute_losses_over_iterations_for_adam(neural_network, evaluation_assistant=eval_assist,
                                                              parameter=eval_assist.test_set[0])
        self.assertEqual(len(losses_adam), eval_assist.number_of_iterations_for_testing + 1)

    def test_compute_rate(self):

        dist = torch.distributions.uniform.Uniform(0, 10)
        loss_at_beginning = dist.sample((1,)).item()
        loss_at_end = dist.sample((1,)).item()
        stopping_time = torch.randint(low=1, high=20, size=(1,)).item()
        rate = compute_rate(loss_at_beginning=loss_at_beginning, loss_at_end=loss_at_end, stopping_time=stopping_time)
        self.assertEqual(rate, (loss_at_end/loss_at_beginning) ** (1/stopping_time))

    def test_compute_losses(self):
        # Check that we compute the same amount of values for both algorithms, that we have the correct number of
        # ground-truth losses, and that percentage lies in [0,1].
        eval_assist, neural_network = set_up_evaluation_assistant(loading_path=self.loading_path)
        eval_assist.test_set = eval_assist.test_set[0:2]
        learned_algorithm = eval_assist.set_up_learned_algorithm()

        (losses_adam,
         losses_of_learned_algorithm,
         rates_of_learned_algorithm,
         stopping_times_of_learned_algorithm,
         ground_truth_losses, percentage) = compute_losses(evaluation_assistant=eval_assist,
                                                           learned_algorithm=learned_algorithm,
                                                           neural_network_for_standard_training=neural_network)
        self.assertEqual(losses_adam.shape, losses_of_learned_algorithm.shape)
        self.assertEqual(len(losses_adam), len(ground_truth_losses))
        self.assertTrue(0 <= percentage <= 1)

    def test_set_up_evaluation_assistant(self):
        eval_assist, _ = set_up_evaluation_assistant(loading_path=self.loading_path)
        self.assertIsInstance(eval_assist, EvaluationAssistant)
