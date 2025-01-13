import unittest
import torch
from classes.StoppingCriterion.derived_classes.subclass_LossCriterion import LossCriterion
from algorithms.dummy import Dummy
from classes.LossFunction.class_LossFunction import LossFunction
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm


def loss_function(x):
    return 0.5 * torch.linalg.norm(x) ** 2


class TestLossCriterion(unittest.TestCase):

    def setUp(self):

        dim = torch.randint(low=1, high=1000, size=(1,)).item()
        length_state = 1
        self.initial_state = 0.05 * torch.randn(size=(length_state, dim))

        self.optimization_algorithm = OptimizationAlgorithm(
            implementation=Dummy(),
            initial_state=self.initial_state,
            loss_function=LossFunction(loss_function)
        )

    def test_call_loss_criterion(self):

        thresholds = 2 * torch.randn((25,))

        for t in thresholds:
            self.optimization_algorithm.set_stopping_criterion(LossCriterion(threshold=t))
            self.assertEqual((loss_function(self.initial_state[-1]) < t).item(),
                             self.optimization_algorithm.evaluate_stopping_criterion())
