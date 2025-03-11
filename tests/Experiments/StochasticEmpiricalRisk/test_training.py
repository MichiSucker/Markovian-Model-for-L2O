import unittest
import torch
from typing import Callable
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm
from algorithms.stochastic_gradient_descent import StochasticGradientDescent
from classes.LossFunction.class_LossFunction import LossFunction
from classes.OptimizationAlgorithm.derived_classes.derived_classes.subclass_PacBayesOptimizationAlgorithm import (
    PacBayesOptimizationAlgorithm)
from classes.StoppingCriterion.derived_classes.subclass_LossCriterion import LossCriterion
from classes.Constraint.class_Constraint import Constraint
from classes.LossFunction.derived_classes.derived_classes.StochasticParametricLossFunctions import (
    StochasticParametricLossFunction)
from experiments.neural_network_stochastic.neural_network import (NeuralNetworkForLearning,
                                                                  NeuralNetworkForStandardTraining)
from experiments.neural_network_stochastic.training import (get_number_of_datapoints,
                                                            create_parametric_loss_functions_from_parameters,
                                                            get_initial_state,
                                                            get_parameters_of_estimation,
                                                            get_update_parameters,
                                                            get_sampling_parameters,
                                                            get_fitting_parameters,
                                                            get_initialization_parameters,
                                                            get_describing_property,
                                                            get_constraint_parameters,
                                                            get_pac_bayes_parameters,
                                                            get_constraint,
                                                            get_stopping_criterion,
                                                            get_batch_size,
                                                            get_algorithm_for_learning,
                                                            get_algorithm_for_initialization,
                                                            instantiate_neural_networks)


class TestTrainingStochasticERM(unittest.TestCase):

    def test_get_number_of_datapoints(self):
        # Check that each data set is specified
        number_of_datapoints = get_number_of_datapoints()
        self.assertTrue('prior' in number_of_datapoints.keys())
        self.assertTrue('train' in number_of_datapoints.keys())
        self.assertTrue('test' in number_of_datapoints.keys())
        self.assertTrue('validation' in number_of_datapoints.keys())
        self.assertTrue(len(number_of_datapoints.keys()) == 4)

    def test_create_parametric_loss_functions(self):

        # Initialize setting.
        def dummy_function(x, parameter):
            return parameter['scale'] * torch.linalg.norm(x)

        n_prior = torch.randint(low=1, high=10, size=(1,)).item()
        n_train = torch.randint(low=1, high=10, size=(1,)).item()
        n_test = torch.randint(low=1, high=10, size=(1,)).item()
        n_val = torch.randint(low=1, high=10, size=(1,)).item()
        parameters = {'prior': [{'dataset': torch.rand(size=(1,))} for _ in range(n_prior)],
                      'train': [{'dataset': torch.rand(size=(1,))} for _ in range(n_train)],
                      'test': [{'dataset': torch.rand(size=(1,))} for _ in range(n_test)],
                      'validation': [{'dataset': torch.rand(size=(1,))} for _ in range(n_val)]}

        # Check that we get a dictionary with four entries, where each entry corresponds to one data set and is a list
        # of ParametricLossFunctions.
        loss_functions = create_parametric_loss_functions_from_parameters(template_loss_function=dummy_function,
                                                                          template_single_loss=dummy_function,
                                                                          parameters=parameters)
        self.assertIsInstance(loss_functions, dict)
        self.assertEqual(len(loss_functions['prior']), n_prior)
        self.assertEqual(len(loss_functions['train']), n_train)
        self.assertEqual(len(loss_functions['test']), n_test)
        self.assertEqual(len(loss_functions['validation']), n_val)
        self.assertTrue(len(loss_functions.keys()) == 4)

        for name in ['prior', 'train', 'test', 'validation']:
            for function in loss_functions[name]:
                self.assertIsInstance(function, StochasticParametricLossFunction)

    def test_get_initial_state(self):
        dim = torch.randint(low=1, high=100, size=(1,)).item()
        x_0 = get_initial_state(dim=dim)
        self.assertEqual(x_0.shape, torch.Size((3, dim)))

    def test_get_parameters_of_estimation(self):
        # Check that each field is specified, and only those.
        estimation_parameters = get_parameters_of_estimation()
        self.assertTrue('quantile_distance' in estimation_parameters.keys())
        self.assertTrue('quantiles' in estimation_parameters.keys())
        self.assertTrue('probabilities' in estimation_parameters.keys())
        self.assertTrue(len(estimation_parameters.keys()) == 3)

    def test_get_update_parameters(self):
        # Check that each field is specified, and only those.
        update_parameters = get_update_parameters()
        self.assertTrue('num_iter_print_update' in update_parameters.keys())
        self.assertTrue('with_print' in update_parameters.keys())
        self.assertTrue('bins' in update_parameters.keys())
        self.assertTrue(len(update_parameters.keys()) == 3)

    def test_get_sampling_parameters(self):
        # Check that each field is specified, and only those.
        # Also check that restart_probability is specified correctly.
        maximal_number_of_iterations = torch.randint(low=1, high=100, size=(1,)).item()
        sampling_parameters = get_sampling_parameters(maximal_number_of_iterations)
        self.assertTrue('restart_probability' in sampling_parameters.keys())
        self.assertTrue('length_trajectory' in sampling_parameters.keys())
        self.assertTrue('num_samples' in sampling_parameters.keys())
        self.assertTrue('lr' in sampling_parameters.keys())
        self.assertTrue('with_restarting' in sampling_parameters.keys())
        self.assertTrue('num_iter_burnin' in sampling_parameters.keys())
        self.assertTrue(len(sampling_parameters.keys()) == 6)
        self.assertTrue(sampling_parameters['restart_probability']
                        == sampling_parameters['length_trajectory']/maximal_number_of_iterations)

    def test_get_fitting_parameters(self):
        # Check that each field is specified, and only those.
        # Also check that restart_probability is specified correctly.
        maximal_number_of_iterations = torch.randint(low=1, high=100, size=(1,)).item()
        fitting_parameters = get_fitting_parameters(maximal_number_of_iterations=maximal_number_of_iterations)
        self.assertTrue('restart_probability' in fitting_parameters.keys())
        self.assertTrue('length_trajectory' in fitting_parameters.keys())
        self.assertTrue('n_max' in fitting_parameters.keys())
        self.assertTrue('lr' in fitting_parameters.keys())
        self.assertTrue('num_iter_update_stepsize' in fitting_parameters.keys())
        self.assertTrue('factor_stepsize_update' in fitting_parameters.keys())
        self.assertTrue(len(fitting_parameters.keys()) == 6)
        self.assertTrue(fitting_parameters['restart_probability']
                        == fitting_parameters['length_trajectory']/maximal_number_of_iterations)

    def test_get_initialization_parameters(self):
        # Check that each field is specified, and only those.
        initialization_parameters = get_initialization_parameters()
        self.assertTrue('lr' in initialization_parameters.keys())
        self.assertTrue('num_iter_max' in initialization_parameters.keys())
        self.assertTrue('num_iter_print_update' in initialization_parameters.keys())
        self.assertTrue('num_iter_update_stepsize' in initialization_parameters.keys())
        self.assertTrue('with_print' in initialization_parameters.keys())
        self.assertTrue(len(initialization_parameters.keys()) == 5)

    def test_get_describing_property(self):
        # Check that you get three functions.
        reduction_property, convergence_risk_constraint = get_describing_property()
        self.assertIsInstance(reduction_property, Callable)
        self.assertIsInstance(convergence_risk_constraint, Callable)

    def test_get_constraint_parameters(self):
        # Check that each field is specified, and only those.
        maximal_number_of_iterations = torch.randint(low=1, high=100, size=(1,)).item()
        constraint_parameters = get_constraint_parameters(maximal_number_of_iterations)
        self.assertTrue('describing_property' in constraint_parameters.keys())
        self.assertTrue('num_iter_update_constraint' in constraint_parameters.keys())
        self.assertTrue('upper_bound' in constraint_parameters.keys())
        self.assertTrue(len(constraint_parameters.keys()) == 3)

    def test_get_pac_bayes_parameters(self):

        # Check that each field is specified, and only those.
        pac_bayes_parameters = get_pac_bayes_parameters()
        self.assertTrue('epsilon' in pac_bayes_parameters.keys())
        self.assertTrue('n_max' in pac_bayes_parameters.keys())
        self.assertTrue('upper_bound' in pac_bayes_parameters.keys())
        self.assertTrue(len(pac_bayes_parameters.keys()) == 3)

    def test_get_constraint(self):

        def dummy_function(x, parameter):
            return parameter['scale'] * torch.linalg.norm(x)

        parameters = {'prior': [{'dataset': torch.rand(size=(1,))} for _ in range(0)],
                      'train': [{'dataset': torch.rand(size=(1,))} for _ in range(0)],
                      'test': [{'dataset': torch.rand(size=(1,))} for _ in range(0)],
                      'validation': [{'dataset': torch.rand(size=(1,))} for _ in range(3)]}

        # Check that you get a Constraint-object.
        loss_functions = create_parametric_loss_functions_from_parameters(template_loss_function=dummy_function,
                                                                          template_single_loss=dummy_function,
                                                                          parameters=parameters)
        constraint = get_constraint(parameters_of_estimation={'quantile_distance': 0.1,
                                                              'quantiles': (0.05, 0.95),
                                                              'probabilities': (0.9, 1.0)},
                                    loss_functions_for_constraint=loss_functions['validation'])
        self.assertIsInstance(constraint, Constraint)

    def test_get_stopping_criterion(self):
        stop_crit = get_stopping_criterion()
        self.assertIsInstance(stop_crit, LossCriterion)

    def test_get_batch_size(self):
        batch_size = get_batch_size()
        self.assertIsInstance(batch_size, int)
        self.assertTrue(batch_size > 0)

    def test_get_algorithm_for_learning(self):
        def dummy_function(x, parameter):
            return parameter['scale'] * torch.linalg.norm(x)

        n_prior = torch.randint(low=1, high=10, size=(1,)).item()
        n_train = torch.randint(low=1, high=10, size=(1,)).item()
        n_test = torch.randint(low=1, high=10, size=(1,)).item()
        n_val = torch.randint(low=1, high=10, size=(1,)).item()
        parameters = {'prior': [{'dataset': torch.rand(size=(1,))} for _ in range(n_prior)],
                      'train': [{'dataset': torch.rand(size=(1,))} for _ in range(n_train)],
                      'test': [{'dataset': torch.rand(size=(1,))} for _ in range(n_test)],
                      'validation': [{'dataset': torch.rand(size=(1,))} for _ in range(n_val)]}

        loss_functions = create_parametric_loss_functions_from_parameters(template_loss_function=dummy_function,
                                                                          template_single_loss=dummy_function,
                                                                          parameters=parameters)
        algo = get_algorithm_for_learning(loss_functions=loss_functions, dimension_of_optimization_variable=10)
        # Check that we get a PacBayesOptimizationAlgorithm, and the needed parameters are set.
        self.assertIsInstance(algo, PacBayesOptimizationAlgorithm)
        self.assertIsInstance(algo.n_max, int)
        self.assertIsInstance(algo.epsilon.item(), float)
        self.assertIsInstance(algo.constraint, Constraint)

    def test_get_algorithm_for_initialization(self):

        def dummy_function(x):
            return torch.linalg.norm(x)

        # Check that we initialize with GradientDescent.
        dim = torch.randint(low=1, high=100, size=(1,)).item()
        x_0 = get_initial_state(dim=dim)
        algo = get_algorithm_for_initialization(initial_state_for_std_algorithm=x_0[-1].reshape((1, -1)),
                                                loss_function=LossFunction(function=dummy_function))
        self.assertIsInstance(algo, OptimizationAlgorithm)
        self.assertIsInstance(algo.implementation, StochasticGradientDescent)
        self.assertEqual(algo.initial_state.shape, torch.Size((1, dim)))

    def test_instantiate_neural_networks(self):
        # Check that we get both version of the neural network.
        nn_std, nn_learn = instantiate_neural_networks()
        self.assertIsInstance(nn_std, NeuralNetworkForStandardTraining)
        self.assertIsInstance(nn_learn, NeuralNetworkForLearning)
        self.assertEqual(nn_learn.degree, 5)
