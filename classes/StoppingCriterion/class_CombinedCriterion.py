from typing import List
from classes.StoppingCriterion.class_StoppingCriterion import StoppingCriterion


class CombinedCriterion:
    def __init__(self, list_of_criteria: List[StoppingCriterion]):
        self.criteria = list_of_criteria

    def __call__(self, *args, **kwargs) -> bool:
        return any([c(*args) for c in self.criteria])
