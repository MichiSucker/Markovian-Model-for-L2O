import unittest
import torch
import copy
from classes.Constraint.class_Constraint import Constraint
from classes.OptimizationAlgorithm.derived_classes.derived_classes.subclass_PacBayesOptimizationAlgorithm import (
    PacBayesOptimizationAlgorithm)
from classes.LossFunction.derived_classes.subclass_ParametricLossFunction import ParametricLossFunction
from classes.StoppingCriterion.derived_classes.subclass_LossCriterion import LossCriterion
from algorithms.gradient_descent import GradientDescent
from exponential_family.describing_property.average_rate_property import get_rate_property


def f(x, parameter):
    return torch.linalg.norm(x) ** 2


class TestPacBayesOptimizationAlgorithm(unittest.TestCase):

    def setUp(self):
        self.dim = torch.randint(low=2, high=10, size=(1,)).item()
        self.initial_state = torch.randn(size=(1, self.dim))
        self.eps = torch.tensor(0.05)
        self.n_max = 10
        self.stopping_criterion = LossCriterion(threshold=0.5)
        self.loss_function = ParametricLossFunction(function=f, parameter={'optimal_loss': torch.tensor(0.0)})
        self.algorithm = PacBayesOptimizationAlgorithm(initial_state=self.initial_state,
                                                       epsilon=self.eps,
                                                       n_max=self.n_max,
                                                       loss_function=self.loss_function,
                                                       implementation=GradientDescent(alpha=torch.tensor(0.1)),
                                                       stopping_criterion=self.stopping_criterion)

    def test_compute_convergence_time_and_contraction_rate(self):

        # Check normal behavior
        convergence_time, contraction_factor = self.algorithm.compute_convergence_time_and_contraction_rate()
        self.assertIsInstance(convergence_time, torch.Tensor)
        self.assertIsInstance(convergence_time.item(), float)
        self.assertIsInstance(contraction_factor, torch.Tensor)
        self.assertIsInstance(contraction_factor.item(), float)
        self.assertTrue(0 <= convergence_time <= self.n_max)
        self.assertTrue(0 <= contraction_factor < 1.0)

        # Check for the case that we are already at the solution
        self.algorithm.initial_state = torch.zeros((1, self.dim))
        convergence_time, contraction_factor = self.algorithm.compute_convergence_time_and_contraction_rate()
        self.assertEqual(convergence_time, 0)
        self.assertEqual(contraction_factor, torch.tensor(0.0))

        # Check for the case that the optimal value was not specified
        with self.assertRaises(RuntimeError):
            del self.algorithm.loss_function.get_parameter()['optimal_loss']
            self.algorithm.compute_convergence_time_and_contraction_rate()

    def test_evaluate_convergence_risk(self):
        loss_functions = [ParametricLossFunction(function=f, parameter={'optimal_loss': torch.tensor(0.0)})
                          for _ in range(10)]
        constraint_functions = [Constraint(function=lambda x: True) for _ in range(10)]

        # Check normal behavior
        rates, probabilities, times = self.algorithm.evaluate_convergence_risk(
            loss_functions=loss_functions, constraint_functions=constraint_functions)
        self.assertIsInstance(rates, torch.Tensor)
        self.assertIsInstance(rates.item(), float)
        self.assertIsInstance(probabilities, torch.Tensor)
        self.assertIsInstance(probabilities.item(), float)
        self.assertIsInstance(times, torch.Tensor)
        self.assertIsInstance(times.item(), float)

        # Check that zero (or t_max) is set, if constraint is not satisfied
        constraint_functions = [Constraint(function=lambda x: False) for _ in range(10)]
        rates, probabilities, times = self.algorithm.evaluate_convergence_risk(
            loss_functions=loss_functions, constraint_functions=constraint_functions)
        self.assertTrue(rates == torch.tensor(0.0))
        self.assertTrue(probabilities == torch.tensor(0.0))
        self.assertTrue(times == self.n_max)

        # Check that zero (or t_max) is set, if constraint is not satisfied
        constraint_functions = ([Constraint(function=lambda x: False) for _ in range(5)]
                                + [Constraint(function=lambda x: True) for _ in range(5)])
        rates, probabilities, times = self.algorithm.evaluate_convergence_risk(
            loss_functions=loss_functions, constraint_functions=constraint_functions)
        self.assertTrue(probabilities == torch.tensor(0.5))

    def test_evaluate_potentials(self):

        loss_functions = [ParametricLossFunction(function=f, parameter={'optimal_loss': torch.tensor(0.0)})
                          for _ in range(10)]
        state_dict_samples = [copy.deepcopy(self.algorithm.implementation.state_dict()) for _ in range(3)]
        constraint_functions = [Constraint(function=lambda x: True) for _ in range(10)]

        rates, probabilities, stopping_times = self.algorithm.evaluate_potentials(
            loss_functions=loss_functions, constraint_functions=constraint_functions,
            state_dict_samples=state_dict_samples)

        # Only check for length, because concrete computation is already checked above.
        for x in [rates, probabilities, stopping_times]:
            self.assertIsInstance(x, torch.Tensor)
            self.assertTrue(len(x) == len(state_dict_samples))

    @unittest.skip("Too expensive.")
    def test_estimate_lambda_for_convergence_rate(self):
        length = torch.randint(2, 10, (1,)).item()
        prior = torch.softmax(torch.randn((length,)), dim=0)
        posterior = torch.softmax(torch.randn((length,)), dim=0)
        potentials = 3 * torch.randn((length,))
        n = torch.randint(1, 100, (1,)).item()
        constraint_parameters = {"bound": 1.0}
        lambda_rate = self.algorithm.estimate_lambda_for_convergence_rate(
            prior=prior, posterior=posterior, potentials_independent_from_prior=potentials, size_of_training_data=n,
            constraint_parameters=constraint_parameters)
        self.assertIsInstance(lambda_rate, torch.Tensor)
        self.assertTrue(lambda_rate > 0.)

    @unittest.skip("Too expensive.")
    def test_estimate_lambda_for_convergence_time(self):
        length = torch.randint(2, 10, (1,)).item()
        prior = torch.softmax(torch.randn((length,)), dim=0)
        posterior = torch.softmax(torch.randn((length,)), dim=0)
        potentials = 3 * torch.randn((length,))
        n = torch.randint(1, 100, (1,)).item()
        lambda_time = self.algorithm.estimate_lambda_for_convergence_time(
            prior=prior, posterior=posterior, potentials_independent_from_prior=potentials, size_of_training_data=n)
        self.assertIsInstance(lambda_time, torch.Tensor)
        self.assertTrue(lambda_time > 0.)

    @unittest.skip("Too expensive.")
    def test_estimate_lambda_for_convergence_probability(self):
        length = torch.randint(2, 10, (1,)).item()
        prior = torch.softmax(torch.randn((length,)), dim=0)
        posterior = torch.softmax(torch.randn((length,)), dim=0)
        potentials = 3 * torch.randn((length,))
        n = torch.randint(1, 100, (1,)).item()
        lambda_conv_prob = self.algorithm.estimate_lambda_for_convergence_probability(
            prior=prior, posterior=posterior, potentials_independent_from_prior=potentials, size_of_training_data=n)
        self.assertIsInstance(lambda_conv_prob, torch.Tensor)
        self.assertTrue(lambda_conv_prob > 0.)

    @unittest.skip("Too expensive.")
    def test_get_preliminary_prior_distribution(self):
        loss_functions = [ParametricLossFunction(function=f, parameter={'optimal_loss': torch.tensor(0.0)})
                          for _ in range(10)]
        state_dict_samples = [copy.deepcopy(self.algorithm.implementation.state_dict()) for _ in range(3)]
        constraint_functions = [Constraint(function=lambda x: True) for _ in range(10)]

        prior, rates, conv_probs, stopping_times = self.algorithm.get_preliminary_prior_distribution(
            loss_functions=loss_functions, state_dict_samples=state_dict_samples, constraints=constraint_functions)

        self.assertIsInstance(prior, torch.Tensor)
        self.assertTrue(torch.all(prior >= 0))
        self.assertTrue(torch.allclose(torch.sum(prior), torch.tensor(1.0)))

        self.assertIsInstance(rates, torch.Tensor)
        self.assertTrue(torch.all(rates < 0))   # Stored as negative numbers.
        self.assertTrue(len(rates) == len(state_dict_samples))

        self.assertIsInstance(conv_probs, torch.Tensor)
        self.assertTrue(torch.all(conv_probs <= 1.) and torch.all(conv_probs >= 0.))
        self.assertTrue(len(conv_probs) == len(state_dict_samples))

        self.assertIsInstance(stopping_times, torch.Tensor)
        self.assertTrue(torch.all(stopping_times >= 0.) and torch.all(stopping_times <= self.n_max))
        self.assertTrue(len(stopping_times) == len(state_dict_samples))

    @unittest.skip("Too expensive.")
    def test_get_preliminary_posterior_distribution(self):

        # Basically the same test as above.

        loss_functions = [ParametricLossFunction(function=f, parameter={'optimal_loss': torch.tensor(0.0)})
                          for _ in range(10)]
        state_dict_samples = [copy.deepcopy(self.algorithm.implementation.state_dict()) for _ in range(3)]
        constraint_functions = [Constraint(function=lambda x: True) for _ in range(10)]
        prior_rates = -torch.abs(torch.randn((len(state_dict_samples,))))

        prior, rates, conv_probs, stopping_times = self.algorithm.get_preliminary_posterior_distribution(
            loss_functions=loss_functions, prior_rates=prior_rates,
            state_dict_samples=state_dict_samples, constraints=constraint_functions)

        self.assertIsInstance(prior, torch.Tensor)
        self.assertTrue(torch.all(prior >= 0))
        self.assertTrue(torch.allclose(torch.sum(prior), torch.tensor(1.0)))

        self.assertIsInstance(rates, torch.Tensor)
        self.assertTrue(torch.all(rates < 0))  # Stored as negative numbers.
        self.assertTrue(len(rates) == len(state_dict_samples))

        self.assertIsInstance(conv_probs, torch.Tensor)
        self.assertTrue(torch.all(conv_probs <= 1.) and torch.all(conv_probs >= 0.))
        self.assertTrue(len(conv_probs) == len(state_dict_samples))

        self.assertIsInstance(stopping_times, torch.Tensor)
        self.assertTrue(torch.all(stopping_times >= 0.) and torch.all(stopping_times <= self.n_max))
        self.assertTrue(len(stopping_times) == len(state_dict_samples))

    @unittest.skip("Too expensive.")
    def test_get_estimates_for_lambdas_and_build_prior(self):
        loss_functions = [ParametricLossFunction(function=f, parameter={'optimal_loss': torch.tensor(0.0)})
                          for _ in range(10)]
        state_dict_samples = [copy.deepcopy(self.algorithm.implementation.state_dict()) for _ in range(3)]

        rate_property, rate_constraint = get_rate_property(bound=1.0, n_max=self.n_max)
        constraint_parameters = {'describing_property': rate_property, 'optimal_loss': torch.tensor(0.0), 'bound': 1.}

        (final_prior,
         final_prior_potentials,
         lambda_rate,
         lambda_time,
         lambda_prob) = self.algorithm.get_estimates_for_lambdas_and_build_prior(
            loss_functions_prior=loss_functions, state_dict_samples_prior=state_dict_samples,
            constraint_parameters=constraint_parameters)

        self.assertIsInstance(final_prior, torch.Tensor)
        self.assertTrue(torch.all(final_prior) >= 0)
        self.assertTrue(torch.allclose(torch.sum(final_prior), torch.tensor(1.0)))
        self.assertTrue(len(final_prior), len(state_dict_samples))

        self.assertIsInstance(final_prior_potentials, torch.Tensor)
        self.assertTrue(len(final_prior_potentials), len(state_dict_samples))

        self.assertIsInstance(lambda_rate, torch.Tensor)
        self.assertTrue(lambda_rate >= 0.)
        self.assertIsInstance(lambda_time, torch.Tensor)
        self.assertTrue(lambda_time >= 0.)
        self.assertIsInstance(lambda_prob, torch.Tensor)
        self.assertTrue(lambda_prob >= 0.)

    @unittest.skip("Too expensive.")
    def test_build_posterior(self):

        loss_functions = [ParametricLossFunction(function=f, parameter={'optimal_loss': torch.tensor(0.0)})
                          for _ in range(10)]
        state_dict_samples = [copy.deepcopy(self.algorithm.implementation.state_dict()) for _ in range(3)]
        prior_rates = -torch.abs(torch.randn((len(state_dict_samples,))))
        rate_property, rate_constraint = get_rate_property(bound=1.0, n_max=self.n_max)
        constraint_parameters = {'describing_property': rate_property, 'optimal_loss': torch.tensor(0.0), 'bound': 1.}

        (posterior,
         posterior_potentials,
         stopping_times,
         convergence_probabilities) = self.algorithm.build_posterior(loss_functions_train=loss_functions,
                                                                     state_dict_samples_prior=state_dict_samples,
                                                                     prior_potentials=prior_rates,
                                                                     constraint_parameters=constraint_parameters)

        self.assertIsInstance(posterior, torch.Tensor)
        self.assertTrue(torch.all(posterior) >= 0)
        self.assertTrue(torch.allclose(torch.sum(posterior), torch.tensor(1.0)))
        self.assertTrue(len(posterior), len(state_dict_samples))

        self.assertIsInstance(posterior_potentials, torch.Tensor)
        self.assertTrue(len(posterior_potentials), len(state_dict_samples))

        self.assertIsInstance(stopping_times, torch.Tensor)
        self.assertTrue(torch.all(stopping_times >= 0.))
        self.assertTrue(torch.all(stopping_times <= self.n_max))
        self.assertTrue(len(stopping_times), len(state_dict_samples))

        self.assertIsInstance(convergence_probabilities, torch.Tensor)
        self.assertTrue(torch.all(convergence_probabilities >= 0.))
        self.assertTrue(torch.all(convergence_probabilities <= 1.))

    def test_select_optimal_hyperparameters(self):

        posterior = torch.tensor([0.2, 0.5, 0.3])
        state_dict_samples = [copy.deepcopy(GradientDescent(alpha=torch.tensor(i).float()).state_dict())
                              for i in range(3)]
        self.algorithm.select_optimal_hyperparameters(posterior=posterior, state_dict_samples_prior=state_dict_samples)
        self.assertTrue(torch.equal(self.algorithm.implementation.state_dict()['alpha'], torch.tensor(1.)))

    def test_compute_pac_bound_for_convergence_rate(self):

        num_samples = 3
        prior = torch.tensor([0.2, 0.5, 0.3])
        posterior = torch.tensor([0.3, 0.6, 0.1])
        state_dict_samples = [copy.deepcopy(self.algorithm.implementation.state_dict()) for _ in range(num_samples)]
        potentials = -torch.distributions.uniform.Uniform(0.0, 1.0).sample((len(state_dict_samples), ))
        lambda_rate = torch.tensor(2.5)
        rate_property, rate_constraint = get_rate_property(bound=1.0, n_max=self.n_max)
        constraint_parameters = {'describing_property': rate_property, 'optimal_loss': torch.tensor(0.0), 'bound': 1.}
        pac_bound_rate = self.algorithm.compute_pac_bound_for_convergence_rate(
            prior=prior, posterior=posterior, potentials_that_are_independent_from_prior=potentials,
            lambda_rate=lambda_rate, size_of_training_data=10, constraint_parameters=constraint_parameters)
        self.assertIsInstance(pac_bound_rate, torch.Tensor)
        self.assertIsInstance(pac_bound_rate.item(), float)

    def test_compute_pac_bound_for_convergence_time(self):

        num_samples = 3
        prior = torch.tensor([0.2, 0.5, 0.3])
        posterior = torch.tensor([0.3, 0.6, 0.1])
        state_dict_samples = [copy.deepcopy(self.algorithm.implementation.state_dict()) for _ in range(num_samples)]
        potentials = torch.randint(low=0, high=self.n_max, size=(len(state_dict_samples), ))
        lambda_time = torch.tensor(3.7)
        pac_bound_time = self.algorithm.compute_pac_bound_for_convergence_time(
            prior=prior, posterior=posterior, potentials_that_are_independent_from_prior=potentials,
            lambda_time=lambda_time, size_of_training_data=10)
        self.assertIsInstance(pac_bound_time, torch.Tensor)
        self.assertIsInstance(pac_bound_time.item(), float)

    def test_compute_pac_bound_for_convergence_probability(self):

        num_samples = 3
        prior = torch.tensor([0.2, 0.5, 0.3])
        posterior = torch.tensor([0.3, 0.6, 0.1])
        state_dict_samples = [copy.deepcopy(self.algorithm.implementation.state_dict()) for _ in range(num_samples)]
        potentials = torch.distributions.uniform.Uniform(0.0, 1.0).sample((len(state_dict_samples), ))
        lambda_prob = torch.tensor(1.7)
        pac_bound_prob = self.algorithm.compute_pac_bound_for_convergence_probability(
            prior=prior, posterior=posterior, potentials_that_are_independent_from_prior=potentials,
            lambda_prob=lambda_prob, size_of_training_data=10)
        self.assertIsInstance(pac_bound_prob, torch.Tensor)
        self.assertIsInstance(pac_bound_prob.item(), float)

    def test_pac_bayes_fit(self):

        # Most things already have been tested, so we mainly check that the algorithm runs through and produces the
        # correct type of outputs.

        loss_functions_prior = [ParametricLossFunction(function=f, parameter={'optimal_loss': torch.tensor(0.0)})
                                for _ in range(10)]
        loss_functions_train = [ParametricLossFunction(function=f, parameter={'optimal_loss': torch.tensor(0.0)})
                                for _ in range(10)]
        rate_property, rate_constraint = get_rate_property(bound=1.0, n_max=self.n_max)
        constraint_parameters = {'describing_property': rate_property, 'optimal_loss': torch.tensor(0.0), 'bound': 1.,
                                 'num_iter_update_constraint': 100}
        fitting_parameters = {'restart_probability': 0.5, 'length_trajectory': 1, 'n_max': 100,
                              'num_iter_update_stepsize': 5, 'factor_stepsize_update': 0.5, 'lr': 1e-4}
        update_parameters = {'with_print': True, 'num_iter_print_update': 10, 'bins': []}
        sampling_parameters = {'restart_probability': 0.9, 'length_trajectory': 1, 'lr': 1e-6, 'num_samples': 5,
                               'num_iter_burnin': 5}

        (pac_bound_rate,
         pac_bound_convergence_probability,
         pac_bound_time,
         state_dict_samples) = self.algorithm.pac_bayes_fit(loss_functions_prior=loss_functions_prior,
                                                            loss_functions_train=loss_functions_train,
                                                            fitting_parameters=fitting_parameters,
                                                            sampling_parameters=sampling_parameters,
                                                            constraint_parameters=constraint_parameters,
                                                            update_parameters=update_parameters)

        for b in [pac_bound_rate, pac_bound_convergence_probability, pac_bound_time]:
            self.assertIsInstance(b, torch.Tensor)
            self.assertIsInstance(b.item(), float)
            self.assertTrue(b > 0.)

        self.assertTrue(len(state_dict_samples) == sampling_parameters['num_samples'])
