import unittest
from collections.abc import Callable
import torch
from classes.LossFunction.class_LossFunction import LossFunction
from classes.OptimizationAlgorithm.derived_classes.derived_classes.subclass_PacBayesOptimizationAlgorithm import (
    kl, get_pac_bound_as_function_of_lambda, phi_inv, specify_test_points, minimize_upper_bound_in_lambda,
    get_splitting_index, compute_pac_bound, build_final_prior)
from classes.OptimizationAlgorithm.derived_classes.derived_classes.subclass_PacBayesOptimizationAlgorithm import (
    PacBayesOptimizationAlgorithm)
from classes.LossFunction.class_LossFunction import LossFunction
from classes.StoppingCriterion.derived_classes.subclass_LossCriterion import LossCriterion
from algorithms.gradient_descent import GradientDescent


def f(x):
    return torch.linalg.norm(x) ** 2


class TestPacBayesOptimizationAlgorithm(unittest.TestCase):

    def setUp(self):
        self.dim = torch.randint(low=2, high=10, size=(10,)).item()
        self.initial_state = torch.randn(size=(1, dim))
        self.eps = torch.tensor(0.05)
        self.n_max = 10
        self.stopping_criterion = LossCriterion(threshold=0.5)
        self.loss_function = LossFunction(function=f)
        self.algorithm = PacBayesOptimizationAlgorithm(initial_state=self.initial_state,
                                                       epsilon=self.eps,
                                                       n_max=self.n_max,
                                                       loss_function=self.loss_function,
                                                       implementation=GradientDescent(alpha=torch.tensor(0.1)),
                                                       stopping_criterion=self.stopping_criterion)


