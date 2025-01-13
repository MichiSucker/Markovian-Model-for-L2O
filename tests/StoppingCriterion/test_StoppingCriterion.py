import unittest
import torch
from classes.StoppingCriterion.class_StoppingCriterion import StoppingCriterion


class TestStoppingCriterion(unittest.TestCase):

    # Note that StoppingCriterion is just a wrapper around the function, and calling it just evaluates the function.
    def test_call_stopping_criterion(self):

        def f_1(x):
            return x

        def f_2(x: torch.Tensor, y: torch.Tensor):
            return x + y

        def f_3(x: torch.Tensor, y: torch.Tensor):
            return torch.all(x < y)

        stopping_criterion_1 = StoppingCriterion(f_1)
        stopping_criterion_2 = StoppingCriterion(f_2)
        stopping_criterion_3 = StoppingCriterion(f_3)

        for i in range(10):
            dim = torch.randint(low=1, high=10, size=(1,)).item()
            x_1 = torch.randn((dim,))
            x_2 = torch.randn((dim,))

            self.assertTrue(torch.equal(f_1(x_1), stopping_criterion_1(x_1)))
            self.assertTrue(torch.equal(f_2(x_1, x_2), stopping_criterion_2(x_1, x_2)))
            self.assertTrue(torch.equal(f_3(x_1, x_2), stopping_criterion_3(x_1, x_2)))
