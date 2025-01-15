import torch
from classes.LossFunction.class_LossFunction import LossFunction
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm
from typing import Tuple


def get_rate_property(bound: float, n_max: int) -> Tuple:

    def rate_constraint(final_loss: torch.Tensor, init_loss: torch.Tensor, conv_time: int) -> bool:
        avg_ratio = (final_loss / init_loss) ** (1 / conv_time)
        return avg_ratio <= bound

    def rate_property(f: LossFunction, opt_algo: OptimizationAlgorithm) -> bool:

        # Store current state, loss function, etc.
        cur_state, cur_loss = opt_algo.current_state, opt_algo.loss_function
        cur_iteration_count = opt_algo.iteration_counter
        opt_algo.reset_state_and_iteration_counter()

        # Set new loss function and compute corresponding losses
        opt_algo.set_loss_function(f)
        init_loss = opt_algo.evaluate_loss_function_at_current_iterate()
        conv_time = opt_algo.compute_convergence_time(num_steps_max=n_max)
        final_loss = opt_algo.evaluate_loss_function_at_current_iterate()

        # Subtract optimal loss
        init_loss = torch.abs(init_loss - opt_algo.loss_function.get_parameter()['opt_val'])
        final_loss = torch.abs(final_loss - opt_algo.loss_function.get_parameter()['opt_val'])

        # Reset current state, loss function, etc.
        opt_algo.set_current_state(cur_state)
        opt_algo.set_loss_function(cur_loss)
        opt_algo.set_iteration_counter(cur_iteration_count)

        return rate_constraint(final_loss, init_loss, conv_time)

    return rate_property, rate_constraint

