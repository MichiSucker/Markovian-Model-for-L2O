import unittest
import torch
import torch.nn as nn
import numpy as np
from experiments.neural_network_full_batch.training import get_describing_property
from experiments.neural_network_full_batch.evaluation import (compute_losses,
                                                              compute_ground_truth_loss,
                                                              compute_losses_over_iterations_and_stopping_time,
                                                              does_satisfy_constraint,
                                                              compute_losses_over_iterations_for_adam,
                                                              set_up_evaluation_assistant,
                                                              load_data, EvaluationAssistant,
                                                              evaluate_algorithm,
                                                              save_data,
                                                              create_folder_for_storing_data)


class TestEvaluation(unittest.TestCase):

    def setUp(self):
        torch.set_default_dtype(torch.double)
        self.path_to_experiment = ('/home/michael/Desktop/JMLR_Markovian_Model/'
                                   'new_implementation/neural_network_training/')
        self.dummy_savings_path = self.path_to_experiment + 'dummy_data/'
        self.loading_path = self.path_to_experiment + 'data/'

    def test_load_data(self):
        # Just check that it does not throw an error => all variables are found.
        load_data(self.loading_path)

    def test_create_folder(self):
        # Just check that it does not throw an error.
        create_folder_for_storing_data(self.loading_path)

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
        neural_network.load_parameters_from_tensor(eval_assist.initial_state[-1].clone())
        losses_adam = compute_losses_over_iterations_for_adam(neural_network, evaluation_assistant=eval_assist,
                                                              parameter=eval_assist.test_set[0])
        self.assertEqual(len(losses_adam), eval_assist.number_of_iterations_for_testing + 1)
        self.assertEqual(losses_adam[0], eval_assist.loss_of_algorithm(eval_assist.initial_state[-1],
                                                                       eval_assist.test_set[0]))

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

    def test_save_data(self):
        # This is a weak test.
        save_data(savings_path=self.dummy_savings_path,
                  losses_of_learned_algorithm=np.empty(1),
                  rates_of_learned_algorithm=np.empty(1),
                  stopping_times_of_learned_algorithm=np.empty(1),
                  losses_of_adam=np.empty(1),
                  ground_truth_losses=np.empty(1),
                  percentage_constrained_satisfied=1.)

    @unittest.skip('Too expensive to test all the time.')
    def test_evaluate_algorithm(self):
        evaluate_algorithm(loading_path=self.loading_path, path_of_experiment=self.path_to_experiment)
