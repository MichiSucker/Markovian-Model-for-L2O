import unittest
import torch
from typing import Callable
from classes.LossFunction.derived_classes.subclass_ParametricLossFunction import ParametricLossFunction
from classes.StoppingCriterion.class_StoppingCriterion import StoppingCriterion
from classes.Constraint.class_Constraint import Constraint
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm
from classes.OptimizationAlgorithm.derived_classes.derived_classes.subclass_PacBayesOptimizationAlgorithm import (
    PacBayesOptimizationAlgorithm)
from algorithms.nesterov_accelerated_gradient_descent import NesterovAcceleratedGradient
from experiments.image_processing.data_generation import get_image_height_and_width, get_loss_function_of_algorithm
from experiments.image_processing.training import (get_number_of_datapoints,
                                                   get_parameters_of_estimation,
                                                   get_update_parameters,
                                                   get_sampling_parameters,
                                                   get_fitting_parameters,
                                                   get_initialization_parameters,
                                                   get_constraint_parameters,
                                                   get_pac_bayes_parameters,
                                                   get_describing_property,
                                                   get_dimension_of_optimization_variable,
                                                   get_initial_states,
                                                   get_baseline_algorithm,
                                                   get_constraint,
                                                   create_parametric_loss_functions_from_parameters,
                                                   get_stopping_criterion,
                                                   get_algorithm_for_learning)


class TestTrainingImageProcessing(unittest.TestCase):

    def test_get_number_of_datapoints(self):
        # Check that each data set is specified
        number_of_datapoints = get_number_of_datapoints()
        self.assertTrue('prior' in number_of_datapoints.keys())
        self.assertTrue('train' in number_of_datapoints.keys())
        self.assertTrue('test' in number_of_datapoints.keys())
        self.assertTrue('validation' in number_of_datapoints.keys())
        self.assertTrue(len(number_of_datapoints.keys()) == 4)

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

    def test_get_describing_property(self):
        # Check that you get three functions.
        reduction_property, convergence_risk_constraint = get_describing_property()
        self.assertIsInstance(reduction_property, Callable)
        self.assertIsInstance(convergence_risk_constraint, Callable)

    def test_get_dimension_of_optimization_variable(self):
        dim = get_dimension_of_optimization_variable()
        h, w = get_image_height_and_width()
        self.assertTrue(dim == w*h)

    def test_get_initial_states(self):
        # We need two initial states, as Nesterov needs one dimension more.
        init_base, init_lear = get_initial_states()
        self.assertIsInstance(init_base, torch.Tensor)
        self.assertIsInstance(init_lear, torch.Tensor)
        self.assertTrue(torch.equal(init_base[1:], init_lear))

    def test_get_baseline_algorithm(self):

        def dummy_function(x):
            return torch.linalg.norm(x) ** 2

        baseline = get_baseline_algorithm(loss_function_of_algorithm=dummy_function, smoothness_parameter=1)
        self.assertIsInstance(baseline, OptimizationAlgorithm)
        self.assertIsInstance(baseline.implementation, NesterovAcceleratedGradient)

    def test_get_constraint(self):

        def dummy_function(x, parameter):
            return parameter['scale'] * torch.linalg.norm(x)

        parameters = {'prior': [{'scale': torch.rand(size=(1,)).item()} for _ in range(0)],
                      'train': [{'scale': torch.rand(size=(1,)).item()} for _ in range(0)],
                      'test': [{'scale': torch.rand(size=(1,)).item()} for _ in range(0)],
                      'validation': [{'scale': torch.rand(size=(1,)).item()} for _ in range(3)]}

        # Check that you get a Constraint-object.
        loss_function, data_fidelity, regularization, blur_tensor = get_loss_function_of_algorithm()
        loss_functions = create_parametric_loss_functions_from_parameters(template_loss_function=dummy_function,
                                                                          template_data_fidelity=data_fidelity,
                                                                          template_regularizer=regularization,
                                                                          parameters=parameters)
        constraint = get_constraint(loss_functions_for_constraint=loss_functions['validation'])
        self.assertIsInstance(constraint, Constraint)

    def test_get_stopping_criterion(self):
        stopping_crit = get_stopping_criterion()
        self.assertIsInstance(stopping_crit, StoppingCriterion)

    def test_get_algorithm_for_learning(self):

        def dummy_function(x, parameter):
            return parameter['scale'] * torch.linalg.norm(x)

        n_prior = torch.randint(low=1, high=10, size=(1,)).item()
        n_train = torch.randint(low=1, high=10, size=(1,)).item()
        n_test = torch.randint(low=1, high=10, size=(1,)).item()
        n_val = torch.randint(low=1, high=10, size=(1,)).item()
        parameters = {'prior': [{'scale': torch.rand(size=(1,)).item()} for _ in range(n_prior)],
                      'train': [{'scale': torch.rand(size=(1,)).item()} for _ in range(n_train)],
                      'test': [{'scale': torch.rand(size=(1,)).item()} for _ in range(n_test)],
                      'validation': [{'scale': torch.rand(size=(1,)).item()} for _ in range(n_val)]}

        loss_function, data_fidelity, regularization, blur_tensor = get_loss_function_of_algorithm()
        loss_functions = create_parametric_loss_functions_from_parameters(template_loss_function=dummy_function,
                                                                          template_data_fidelity=data_fidelity,
                                                                          template_regularizer=regularization,
                                                                          parameters=parameters)
        algo = get_algorithm_for_learning(loss_functions=loss_functions, smoothness_parameter=2.)
        # Check that we get a PacBayesOptimizationAlgorithm, and the needed parameters are set.
        self.assertIsInstance(algo, PacBayesOptimizationAlgorithm)
        self.assertIsInstance(algo.n_max, int)
        self.assertIsInstance(algo.epsilon.item(), float)
        self.assertIsInstance(algo.constraint, Constraint)

    def test_create_parametric_loss_functions_from_parameters(self):

        # Initialize setting.
        def dummy_function(x, parameter):
            return parameter['scale'] * torch.linalg.norm(x)

        n_prior = torch.randint(low=1, high=10, size=(1,)).item()
        n_train = torch.randint(low=1, high=10, size=(1,)).item()
        n_test = torch.randint(low=1, high=10, size=(1,)).item()
        n_val = torch.randint(low=1, high=10, size=(1,)).item()
        parameters = {'prior': [{'scale': torch.rand(size=(1,)).item()} for _ in range(n_prior)],
                      'train': [{'scale': torch.rand(size=(1,)).item()} for _ in range(n_train)],
                      'test': [{'scale': torch.rand(size=(1,)).item()} for _ in range(n_test)],
                      'validation': [{'scale': torch.rand(size=(1,)).item()} for _ in range(n_val)]}

        # Check that we get a dictionary with four entries, where each entry corresponds to one data set and is a list
        # of ParametricLossFunction.
        loss_function, data_fidelity, regularization, blur_tensor = get_loss_function_of_algorithm()
        loss_functions = create_parametric_loss_functions_from_parameters(template_loss_function=dummy_function,
                                                                          template_data_fidelity=data_fidelity,
                                                                          template_regularizer=regularization,
                                                                          parameters=parameters)
        self.assertIsInstance(loss_functions, dict)
        self.assertEqual(len(loss_functions['prior']), n_prior)
        self.assertEqual(len(loss_functions['train']), n_train)
        self.assertEqual(len(loss_functions['test']), n_test)
        self.assertEqual(len(loss_functions['validation']), n_val)
        self.assertTrue(len(loss_functions.keys()) == 4)

        for name in ['prior', 'train', 'test', 'validation']:
            for function in loss_functions[name]:
                self.assertIsInstance(function, ParametricLossFunction)