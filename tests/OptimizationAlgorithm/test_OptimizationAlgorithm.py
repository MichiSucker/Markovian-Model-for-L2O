import unittest

import algorithms.dummy
from algorithms.gradient_descent import GradientDescent
from classes.Constraint.class_Constraint import Constraint
from classes.StoppingCriterion.class_StoppingCriterion import StoppingCriterion
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm
from classes.LossFunction.class_LossFunction import LossFunction
from algorithms.dummy import Dummy
import torch

from classes.StoppingCriterion.derived_classes.subclass_LossCriterion import LossCriterion


def dummy_function(x):
    return 0.5 * torch.linalg.norm(x) ** 2


def dummy_constraint(optimization_algorithm):
    if torch.all(optimization_algorithm.current_state >= 0):
        return True
    else:
        return False


class TestClassOptimizationAlgorithm(unittest.TestCase):

    def setUp(self):
        self.dim = torch.randint(low=1, high=1000, size=(1,)).item()
        self.length_state = 1  # Take one, because it has to be compatible with Dummy()
        self.initial_state = torch.randn(size=(self.length_state, self.dim))
        self.current_state = self.initial_state.clone()
        self.loss_function = LossFunction(function=dummy_function)
        self.optimization_algorithm = OptimizationAlgorithm(implementation=Dummy(),
                                                            initial_state=self.initial_state,
                                                            loss_function=self.loss_function)

    def test_creation(self):
        self.assertIsInstance(self.optimization_algorithm, OptimizationAlgorithm)

    def test_attributes(self):
        self.assertTrue(hasattr(self.optimization_algorithm, 'implementation'))
        self.assertTrue(hasattr(self.optimization_algorithm, 'loss_function'))
        self.assertTrue(hasattr(self.optimization_algorithm, 'initial_state'))
        self.assertTrue(hasattr(self.optimization_algorithm, 'current_state'))
        self.assertTrue(hasattr(self.optimization_algorithm, 'current_iterate'))
        self.assertTrue(hasattr(self.optimization_algorithm, 'iteration_counter'))
        self.assertTrue(hasattr(self.optimization_algorithm, 'constraint'))
        self.assertTrue(hasattr(self.optimization_algorithm, 'n_max'))

    def test_get_initial_state(self):
        self.assertIsInstance(self.optimization_algorithm.get_initial_state(), torch.Tensor)
        self.assertTrue(torch.equal(self.optimization_algorithm.get_initial_state(), self.initial_state))

    def test_get_implementation(self):
        self.assertIsInstance(self.optimization_algorithm.get_implementation(), algorithms.dummy.Dummy)

    def test_get_current_state(self):
        self.assertIsInstance(self.optimization_algorithm.get_current_state(), torch.Tensor)
        self.assertTrue(torch.equal(self.optimization_algorithm.get_current_state(), self.current_state))

    def test_get_current_iterate(self):
        self.assertIsInstance(self.optimization_algorithm.get_current_iterate(), torch.Tensor)
        self.assertTrue(torch.equal(self.optimization_algorithm.get_current_iterate(),
                                    self.optimization_algorithm.get_current_state()[-1]))

    def test_get_iteration_counter(self):
        self.assertIsInstance(self.optimization_algorithm.get_iteration_counter(), int)
        self.assertTrue(self.optimization_algorithm.get_iteration_counter() >= 0)

    def test_set_iteration_counter(self):
        # Set iteration counter. Raise an error if it is not a non-negative integer.
        random_number = torch.randint(low=0, high=100, size=(1,)).item()
        self.optimization_algorithm.set_iteration_counter(n=random_number)
        self.assertEqual(self.optimization_algorithm.get_iteration_counter(), random_number)
        with self.assertRaises(TypeError):
            self.optimization_algorithm.set_iteration_counter(n=torch.randn(1).item())

    def test_reset_iteration_counter_to_zero(self):
        self.optimization_algorithm.set_iteration_counter(10)
        self.assertNotEqual(self.optimization_algorithm.get_iteration_counter(), 0)
        self.optimization_algorithm.reset_iteration_counter_to_zero()
        self.assertEqual(self.optimization_algorithm.get_iteration_counter(), 0)

    def test_reset_to_initial_state(self):
        self.optimization_algorithm.reset_to_initial_state()
        self.assertTrue(torch.equal(self.optimization_algorithm.current_state,
                                    self.optimization_algorithm.initial_state))
        self.assertTrue(torch.equal(self.optimization_algorithm.initial_state, self.initial_state))

    def test_reset_state_and_iteration_counter(self):
        # First, set state and iteration counter to something that is not the same as in the beginning.
        # Then, reset everything to beginning.
        current_state = self.optimization_algorithm.get_current_state()
        self.optimization_algorithm.set_current_state(torch.randn(current_state.shape))
        self.optimization_algorithm.set_iteration_counter(10)
        self.assertFalse(torch.equal(self.optimization_algorithm.get_current_state(),
                                     self.optimization_algorithm.get_initial_state()))
        self.assertNotEqual(self.optimization_algorithm.get_iteration_counter(), 0)
        self.optimization_algorithm.reset_state_and_iteration_counter()
        self.assertTrue(torch.equal(self.optimization_algorithm.get_current_state(),
                                    self.optimization_algorithm.get_initial_state()))
        self.assertEqual(self.optimization_algorithm.get_iteration_counter(), 0)

    def test_set_current_state(self):
        # Check that state gets changed. Make sure that current iterate gets changed, too.
        old_state = self.optimization_algorithm.get_current_state()
        new_state = torch.randn(size=(self.length_state, self.dim))
        self.optimization_algorithm.set_current_state(new_state)
        self.assertFalse(torch.equal(old_state, self.optimization_algorithm.get_current_state()))
        self.assertTrue(torch.equal(new_state, self.optimization_algorithm.get_current_state()))
        self.assertTrue(torch.equal(new_state[-1], self.optimization_algorithm.get_current_iterate()))
        with self.assertRaises(ValueError):
            state_with_wrong_shape = old_state[-1]
            self.optimization_algorithm.set_current_state(state_with_wrong_shape)

    def test_set_constraint(self):
        old_constraint = self.optimization_algorithm.constraint
        new_constraint = Constraint(dummy_constraint)
        self.optimization_algorithm.set_constraint(new_constraint)
        self.assertFalse(self.optimization_algorithm.constraint is None)
        self.optimization_algorithm.set_constraint(old_constraint)
        self.assertTrue(self.optimization_algorithm.constraint is None)

    def test_set_stopping_criterion(self):
        old_stopping_criterion = self.optimization_algorithm.stopping_criterion
        new_stopping_criterion = StoppingCriterion(
            lambda optimization_algorithm: optimization_algorithm.evaluate_loss_function_at_current_iterate() < 1.)
        self.optimization_algorithm.set_stopping_criterion(new_stopping_criterion)
        self.assertFalse(self.optimization_algorithm.stopping_criterion is None)
        self.optimization_algorithm.set_stopping_criterion(old_stopping_criterion)
        self.assertTrue(self.optimization_algorithm.stopping_criterion is None)

    def test_perform_step(self):
        # Perform one step with the algorithm and make sure that the state and the iteration counter got updated.
        self.assertTrue(hasattr(self.optimization_algorithm.implementation, 'forward'))
        self.assertTrue(hasattr(self.optimization_algorithm.implementation, 'update_state'))
        current_state = self.optimization_algorithm.get_current_state().clone()
        current_iteration_counter = self.optimization_algorithm.get_iteration_counter()
        self.optimization_algorithm.perform_step()
        self.assertEqual(self.optimization_algorithm.iteration_counter, current_iteration_counter + 1)
        self.assertFalse(torch.equal(self.optimization_algorithm.get_current_state(), current_state))

        # Check for return-value.
        self.assertTrue(self.optimization_algorithm.perform_step() is None)
        self.assertTrue(isinstance(self.optimization_algorithm.perform_step(return_iterate=True), torch.Tensor))
        self.assertEqual(self.optimization_algorithm.current_state.shape,
                         self.optimization_algorithm.initial_state.shape)

    def test_compute_trajectory(self):
        # Check that by computing the trajectory we also count the current state.
        self.assertEqual(len(self.optimization_algorithm.compute_partial_trajectory(number_of_steps=10)), 11)

        # Check that we get an error, if we forgot to specify a stopping criterion, but we want to check for convergence
        with self.assertRaises(RuntimeError):
            self.optimization_algorithm.compute_partial_trajectory(number_of_steps=10, check_convergence=True)

        # Check that, if we have a constraint, the algorithm also checks for convergence
        self.optimization_algorithm.set_stopping_criterion(lambda x: True)
        traj, conv = self.optimization_algorithm.compute_partial_trajectory(number_of_steps=10, check_convergence=True)
        self.assertEqual(len(traj), len(conv))
        for i, c in zip(traj, conv):
            self.assertIsInstance(i, torch.Tensor)
            self.assertIsInstance(c, bool)

    def test_compute_convergence_time(self):
        with self.assertRaises(RuntimeError):
            self.optimization_algorithm.compute_convergence_time(num_steps_max=10)

        self.optimization_algorithm.set_stopping_criterion(LossCriterion(threshold=0.5))

        # Convergence in one step
        self.optimization_algorithm.implementation = GradientDescent(alpha=torch.tensor(1.0))
        self.assertEqual(self.optimization_algorithm.compute_convergence_time(num_steps_max=25), 1)

        # Convergence in 0 steps, since already there
        self.assertEqual(self.optimization_algorithm.compute_convergence_time(num_steps_max=25), 0)

        # Convergence in a few steps
        self.optimization_algorithm.reset_state_and_iteration_counter()
        self.optimization_algorithm.implementation = GradientDescent(alpha=torch.tensor(0.5))
        self.assertTrue(self.optimization_algorithm.compute_convergence_time(num_steps_max=25) > 0)
        self.assertTrue(self.optimization_algorithm.compute_convergence_time(num_steps_max=25) < 25)

        # No convergence
        self.optimization_algorithm.reset_state_and_iteration_counter()
        self.optimization_algorithm.implementation = GradientDescent(alpha=torch.tensor(0.0))
        self.assertEqual(self.optimization_algorithm.compute_convergence_time(num_steps_max=25), 25)
        print(self.optimization_algorithm.compute_convergence_time(num_steps_max=25))

    def test_set_loss_function(self):
        current_loss_function = self.optimization_algorithm.loss_function
        new_loss_function = LossFunction(function=lambda x: torch.linalg.norm(x))
        self.optimization_algorithm.set_loss_function(new_loss_function)
        self.assertNotEqual(new_loss_function, current_loss_function)
        self.assertEqual(new_loss_function, self.optimization_algorithm.loss_function)
        self.optimization_algorithm.set_loss_function(current_loss_function)
        self.assertEqual(current_loss_function, self.optimization_algorithm.loss_function)

    def test_evaluate_loss_function_at_current_iterate(self):
        self.assertEqual(self.loss_function(self.optimization_algorithm.current_iterate),
                         self.optimization_algorithm.evaluate_loss_function_at_current_iterate())

    def test_evaluate_gradient_norm_at_current_iterate(self):
        self.assertEqual(
            torch.linalg.norm(self.loss_function.compute_gradient(self.optimization_algorithm.current_iterate)),
            self.optimization_algorithm.evaluate_gradient_norm_at_current_iterate())

    def test_evaluate_constraint(self):
        self.optimization_algorithm.set_constraint(Constraint(dummy_constraint))
        self.assertIsInstance(self.optimization_algorithm.evaluate_constraint(), bool)
        self.optimization_algorithm.set_current_state(torch.ones(size=self.initial_state.shape))
        self.assertTrue(self.optimization_algorithm.evaluate_constraint())

    def test_evaluate_stopping_criterion(self):

        def dummy_stopping_criterion(optimization_algorithm):
            return (optimization_algorithm.evaluate_loss_function_at_current_iterate() < 1.).item()

        self.optimization_algorithm.set_stopping_criterion(StoppingCriterion(dummy_stopping_criterion))
        self.assertEqual(self.optimization_algorithm.evaluate_stopping_criterion(),
                         dummy_stopping_criterion(self.optimization_algorithm))

        def dummy_stopping_criterion_2(optimization_algorithm):
            return (optimization_algorithm.evaluate_loss_function_at_current_iterate() > 1.).item()

        self.optimization_algorithm.set_stopping_criterion(StoppingCriterion(dummy_stopping_criterion_2))
        self.assertEqual(self.optimization_algorithm.evaluate_stopping_criterion(),
                         dummy_stopping_criterion_2(self.optimization_algorithm))

