class StoppingCriterion:

    def __init__(self, stopping_criterion):
        self.stopping_criterion = stopping_criterion

    def __call__(self, *args, **kwargs):
        return self.stopping_criterion(*args)
