import unittest
from collections.abc import Callable
import torch
from classes.LossFunction.class_LossFunction import LossFunction
from classes.OptimizationAlgorithm.derived_classes.derived_classes.subclass_PacBayesOptimizationAlgorithm import (
    kl, get_pac_bound_as_function_of_lambda, phi_inv, specify_test_points, minimize_upper_bound_in_lambda,
    get_splitting_index, compute_pac_bound, build_final_prior)


class TestHelpers(unittest.TestCase):

    def test_kl(self):
        prior = torch.tensor([0.2, 0.5, 0.3])

        # Length does not match.
        wrong_posterior = torch.tensor([1.])
        with self.assertRaises(RuntimeError):
            kl(prior=prior, posterior=wrong_posterior)

        # Negative values
        wrong_posterior = torch.tensor([0.5, -0.5, 1.])
        with self.assertRaises(RuntimeError):
            kl(prior=prior, posterior=wrong_posterior)

        # Not probability distributions
        wrong_posterior = torch.tensor([0.5, 0.5, 1.])
        with self.assertRaises(RuntimeError):
            kl(prior=prior, posterior=wrong_posterior)

        posterior = torch.tensor([0.2, 0.5, 0.3])
        self.assertTrue(kl(prior, posterior) == 0)

        posterior = torch.tensor([0.1, 0.5, 0.4])
        self.assertTrue(kl(prior, posterior) > 0)

    def test_get_pac_bound_as_function(self):
        posterior_risk = torch.tensor(2.3)
        prior = torch.tensor([0.2, 0.5, 0.3])
        posterior = torch.tensor([0.2, 0.5, 0.3])
        n = 10
        upper_bound = 4

        with self.assertRaises(RuntimeError):
            get_pac_bound_as_function_of_lambda(posterior_risk=posterior_risk, prior=prior, posterior=posterior,
                                                eps=torch.tensor(1.), n=n, upper_bound=upper_bound)

        with self.assertRaises(RuntimeError):
            get_pac_bound_as_function_of_lambda(posterior_risk=posterior_risk, prior=prior, posterior=posterior,
                                                eps=torch.tensor(1.), n=n, upper_bound=upper_bound/2)

        eps = torch.tensor(0.95)
        f = get_pac_bound_as_function_of_lambda(posterior_risk=posterior_risk, prior=prior, posterior=posterior,
                                                eps=eps, n=n, upper_bound=upper_bound)
        self.assertIsInstance(f, Callable)
        self.assertIsInstance(f(torch.tensor(0.5)), torch.Tensor)
        test_points = torch.linspace(0.1, 100., steps=10)
        for t in test_points:
            self.assertTrue(f(t) >= posterior_risk)

    def test_phi_inv(self):

        def phi(q, b):
            return -torch.log(1 - (1 - torch.exp(-b)) * q) / b

        test_values_for_a = torch.linspace(-2.0, 2.0, 5)
        test_points = torch.linspace(0.0, 1.0, 5)
        for a in test_values_for_a:
            if a == 0:
                continue
            for t in test_points:
                self.assertTrue(torch.allclose(phi(q=phi_inv(q=t, a=a), b=a), t))

    def test_specify_test_points(self):
        test_points = specify_test_points()
        self.assertIsInstance(test_points, torch.Tensor)
        self.assertTrue(len(test_points) > 1)

    def test_minimize_upper_bound_in_lambda(self):

        def dummy_function(x: torch.Tensor) -> torch.Tensor:
            return 0.5*(x - 1)**2

        test_points = torch.linspace(0, 5, steps=11)
        best_upper_bound, lambda_opt = minimize_upper_bound_in_lambda(pac_bound_function=dummy_function,
                                                                      test_points=test_points)
        self.assertTrue(best_upper_bound == 0.0)
        self.assertTrue(lambda_opt == 1.0)

        def dummy_function(x: torch.Tensor) -> torch.Tensor:
            return 3 * x + 1

        test_points = torch.linspace(0, 5, steps=11)
        best_upper_bound, lambda_opt = minimize_upper_bound_in_lambda(pac_bound_function=dummy_function,
                                                                      test_points=test_points)
        self.assertTrue(best_upper_bound == 1.0)
        self.assertTrue(lambda_opt == 0.0)

        def dummy_function(x: torch.Tensor) -> torch.Tensor:
            return -3 * x + 1

        test_points = torch.linspace(0, 5, steps=11)
        best_upper_bound, lambda_opt = minimize_upper_bound_in_lambda(pac_bound_function=dummy_function,
                                                                      test_points=test_points)
        self.assertTrue(best_upper_bound == -14.0)
        self.assertTrue(lambda_opt == 5.0)

    def test_get_splitting_index(self):

        loss_functions = [LossFunction(lambda x: x) for _ in range(10)]
        n_half, N_1, N_2 = get_splitting_index(loss_functions)
        self.assertTrue(n_half == 5)
        self.assertTrue(N_1 == 5)
        self.assertTrue(N_2 == 5)

        loss_functions = [LossFunction(lambda x: x) for _ in range(11)]
        n_half, N_1, N_2 = get_splitting_index(loss_functions)
        self.assertTrue(n_half == 5)
        self.assertTrue(N_1 == 5)
        self.assertTrue(N_2 == 6)

    @unittest.skip("Too expensive.")
    def test_compute_pac_bound(self):
        posterior_risk = torch.tensor(10.)
        prior = torch.tensor([0.5, 0.3, 0.2])
        posterior = torch.tensor([0.3, 0.5, 0.2])
        eps = torch.tensor(0.05)
        n = 100
        upper_bound = torch.tensor(20.)

        best_upper_bound, lambda_opt = compute_pac_bound(posterior_risk=posterior_risk,
                                                         prior=prior, posterior=posterior,
                                                         eps=eps, n=n, upper_bound=upper_bound)
        self.assertIsInstance(best_upper_bound, torch.Tensor)
        self.assertTrue(best_upper_bound > posterior_risk)
        self.assertIsInstance(lambda_opt, torch.Tensor)
        self.assertTrue(lambda_opt > 0)

    def test_build_final_prior(self):
        potentials_1 = torch.tensor([1.0])
        potentials_2 = torch.tensor([1.0, 2.0])
        n_1 = 10
        n_2 = 20
        with self.assertRaises(RuntimeError):
            build_final_prior(potentials_1=potentials_1, potentials_2=potentials_2, n_1=n_1, n_2=n_2)

        potentials_1 = torch.tensor([1.0, 2.0])
        potentials_2 = torch.tensor([1.0, 2.0])
        n_1 = -1
        n_2 = 20
        with self.assertRaises(RuntimeError):
            build_final_prior(potentials_1=potentials_1, potentials_2=potentials_2, n_1=n_1, n_2=n_2)

        potentials_1 = torch.tensor([1.0, 4.0])
        potentials_2 = torch.tensor([1.0, 2.0])
        n_1 = 10
        n_2 = 20
        prior, prior_potentials = build_final_prior(potentials_1=potentials_1,
                                                    potentials_2=potentials_2,
                                                    n_1=n_1, n_2=n_2)
        self.assertIsInstance(prior, torch.Tensor)
        self.assertTrue(len(prior) == len(potentials_1))
        self.assertTrue(torch.all(prior) >= 0)
        self.assertTrue(torch.allclose(torch.sum(prior), torch.tensor(1.0)))

        self.assertIsInstance(prior_potentials, torch.Tensor)
        self.assertTrue(torch.all(prior_potentials == (n_1 * potentials_1 + n_2 * potentials_2) / (n_1 + n_2)))
