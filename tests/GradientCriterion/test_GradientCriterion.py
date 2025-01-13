import unittest
import torch
from algorithms.dummy import Dummy
from classes.LossFunction.class_LossFunction import LossFunction
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm
from classes.StoppingCriterion.derived_classes.subclass_GradientCriterion import GradientCriterion


def loss_function(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.linalg.norm(x) ** 2


class TestGradientCriterion(unittest.TestCase):

    def setUp(self):

        dim = torch.randint(low=1, high=1000, size=(1,)).item()
        length_state = 1
        self.initial_state = 0.05 * torch.randn(size=(length_state, dim))
        self.loss_function = LossFunction(loss_function)
        self.optimization_algorithm = OptimizationAlgorithm(
            implementation=Dummy(),
            initial_state=self.initial_state,
            loss_function=self.loss_function
        )

    def test_call_gradient_criterion(self):

        thresholds = 2 * torch.randn((25,))

        for t in thresholds:
            self.optimization_algorithm.set_stopping_criterion(GradientCriterion(threshold=t))
            self.assertIsInstance(self.optimization_algorithm.evaluate_stopping_criterion(), bool)
            self.assertEqual(
                (torch.linalg.norm(self.loss_function.compute_gradient(self.initial_state[-1])) < t).item(),
                self.optimization_algorithm.evaluate_stopping_criterion())
