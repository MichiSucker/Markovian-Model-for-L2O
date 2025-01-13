import torch
from classes.OptimizationAlgorithm.class_OptimizationAlgorithm import OptimizationAlgorithm
from classes.StoppingCriterion.class_StoppingCriterion import StoppingCriterion


def criterion(optimization_algorithm: OptimizationAlgorithm, threshold: torch.Tensor) -> bool:
    return (optimization_algorithm.evaluate_gradient_norm_at_current_iterate() < threshold).item()


class GradientCriterion(StoppingCriterion):

    def __init__(self, threshold):
        self.threshold = threshold
        super().__init__(lambda opt_algo: criterion(optimization_algorithm=opt_algo, threshold=self.threshold))
