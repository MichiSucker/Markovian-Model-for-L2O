import unittest
from typing import Callable

import torch.nn
import torch
from experiments.lasso.data_generation import (get_dimensions,
                                               get_distribution_of_right_hand_side,
                                               get_distribution_of_regularization_parameter,
                                               get_matrix_for_smooth_part,
                                               calculate_smoothness_parameter,
                                               get_loss_function_of_algorithm,
                                               check_and_extract_number_of_datapoints,
                                               create_parameter,
                                               get_parameters,
                                               get_data)


class TestDataGeneration(unittest.TestCase):

    def test_get_dimensions(self):
        rhs, opt_var = get_dimensions()
        self.assertIsInstance(rhs, int)
        self.assertIsInstance(opt_var, int)

    def test_get_distribution_of_right_hand_side(self):
        dist = get_distribution_of_right_hand_side()
        self.assertIsInstance(dist, torch.distributions.multivariate_normal.MultivariateNormal)
        mean = dist.mean
        dim_rhs, _ = get_dimensions()
        self.assertTrue(len(mean), dim_rhs)

    def test_get_distribution_of_regularization_parameter(self):
        dist = get_distribution_of_regularization_parameter()
        self.assertIsInstance(dist, torch.distributions.uniform.Uniform)

    def test_get_matrix_for_smooth_part(self):
        matrix = get_matrix_for_smooth_part()
        self.assertIsInstance(matrix, torch.Tensor)
        rhs, opt_var = get_dimensions()
        self.assertEqual(matrix.shape, (rhs, opt_var))

    def test_calculate_smoothness_parameter(self):
        matrix = torch.randn((5, 5))
        sdp_matrix = matrix.T @ matrix
        eigenvalues = torch.linalg.eigvalsh(sdp_matrix)
        self.assertTrue(eigenvalues[-1] == torch.max(eigenvalues))
        self.assertTrue(eigenvalues[-1] == calculate_smoothness_parameter(matrix))

    def test_get_loss_function_of_algorithm(self):

        loss_function, smooth_part, nonsmooth_part = get_loss_function_of_algorithm()
        self.assertIsInstance(loss_function, Callable)
        self.assertIsInstance(smooth_part, Callable)
        self.assertIsInstance(nonsmooth_part, Callable)

        matrix = torch.randn((5, 5))
        sdp_matrix = matrix.T @ matrix
        rhs = torch.randn((5,))
        regularization = torch.randn((1,)) ** 2
        parameter = {'A': sdp_matrix, 'b': rhs, 'mu': regularization}
        x = torch.randn((5,))

        self.assertTrue(smooth_part(x, parameter) + nonsmooth_part(x, parameter) == loss_function(x, parameter))

    def test_check_and_extract_number_of_datapoints(self):
        # Check that it raises an error if at least one of the data sets is not specified.
        # And check that the extracted numbers are correct.
        with self.assertRaises(ValueError):
            check_and_extract_number_of_datapoints({})
        with self.assertRaises(ValueError):
            check_and_extract_number_of_datapoints({'prior': 1})
        with self.assertRaises(ValueError):
            check_and_extract_number_of_datapoints({'prior': 1, 'train': 1})
        with self.assertRaises(ValueError):
            check_and_extract_number_of_datapoints({'prior': 1, 'train': 1, 'test': 1})
        number_data = {'prior': torch.randint(low=1, high=100, size=(1,)).item(),
                       'train': torch.randint(low=1, high=100, size=(1,)).item(),
                       'test': torch.randint(low=1, high=100, size=(1,)).item(),
                       'validation': torch.randint(low=1, high=100, size=(1,)).item()}
        n_prior, n_train, n_test, n_val = check_and_extract_number_of_datapoints(number_data)
        self.assertEqual(n_prior, number_data['prior'])
        self.assertEqual(n_train, number_data['train'])
        self.assertEqual(n_test, number_data['test'])
        self.assertEqual(n_val, number_data['validation'])

    def test_create_parameter(self):
        matrix = torch.randn((5, 5))
        sdp_matrix = matrix.T @ matrix
        rhs = torch.randn((5,))
        regularization = torch.randn((1,)) ** 2
        test_parameter = {'A': sdp_matrix, 'b': rhs, 'mu': regularization}
        parameter = create_parameter(matrix=sdp_matrix, right_hand_side=rhs, regularization_parameter=regularization)
        self.assertIsInstance(parameter, dict)
        self.assertTrue(test_parameter.items() == parameter.items())

    def test_get_parameters(self):
        matrix = torch.randn((5, 5))
        sdp_matrix = matrix.T @ matrix
        number_of_datapoints = {'prior': 2, 'train': 1, 'test': 4, 'validation': 7}

        parameters = get_parameters(matrix=sdp_matrix, number_of_datapoints_per_dataset=number_of_datapoints)
        self.assertIsInstance(parameters, dict)
        self.assertTrue(parameters.keys() == number_of_datapoints.keys())
        for dataset in parameters.keys():
            self.assertIsInstance(parameters[dataset], list)
            self.assertTrue(len(parameters[dataset]), number_of_datapoints[dataset])

    def test_get_data(self):
        number_of_datapoints = {'prior': 1, 'train': 0, 'test': 0, 'validation': 0}
        (parameters, loss_function_of_algorithm,
         smooth_part, nonsmooth_part, smoothness_parameter) = get_data(number_of_datapoints)

        self.assertIsInstance(parameters, dict)
        self.assertTrue(parameters.keys() == number_of_datapoints.keys())
        for dataset in parameters.keys():
            self.assertTrue(len(parameters[dataset]) == number_of_datapoints[dataset])

        self.assertIsInstance(loss_function_of_algorithm, Callable)
        self.assertIsInstance(smooth_part, Callable)
        self.assertIsInstance(nonsmooth_part, Callable)
        self.assertIsInstance(smoothness_parameter, torch.Tensor)
        self.assertIsInstance(smoothness_parameter.item(), float)
        self.assertTrue(smoothness_parameter.item() > 1)
