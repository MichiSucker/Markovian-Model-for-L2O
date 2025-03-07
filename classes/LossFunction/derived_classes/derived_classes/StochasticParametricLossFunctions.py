from classes.LossFunction.derived_classes.subclass_ParametricLossFunction import ParametricLossFunction
import torch


class StochasticParametricLossFunction(ParametricLossFunction):

    def __init__(self, function, single_function, parameter):
        self.N = len(parameter['dataset'])
        self.single_function = single_function
        self.function = function
        self.parameter = parameter
        self.data = parameter['dataset']
        self.empirical_risk = [ParametricLossFunction(function=single_function, parameter=d)
                               for d in parameter['dataset']]
        super().__init__(function=function, parameter=parameter)

    def compute_stochastic_gradient(self, x, batch_size=1, return_loss=False, *args, **kwargs):

        # Get first batch_size elements of random permutation of numbers 0 to N-1
        idx = torch.randperm(self.N)[:batch_size]

        # Select corresponding datapoints
        cur_xes, cur_yes = self.parameter['xes'][idx], self.parameter['yes'][idx]
        cur_parameter = self.parameter.copy()
        cur_parameter['xes'], cur_parameter['yes'] = cur_xes, cur_yes

        # Setup function and compute grad (as usual)
        current_f = ParametricLossFunction(function=self.function, parameter=cur_parameter)
        if return_loss:
            return current_f.compute_gradient(x), current_f(x)
        else:
            return current_f.compute_gradient(x)
