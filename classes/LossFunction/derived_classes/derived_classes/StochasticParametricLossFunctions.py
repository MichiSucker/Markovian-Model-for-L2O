from classes.LossFunction.derived_classes.subclass_ParametricLossFunction import ParametricLossFunction
import torch


class StochasticParametricLossFunction(ParametricLossFunction):

    def __init__(self, function, single_function, parameter):
        self.N = len(parameter['dataset'])
        # Here, we need to be careful with the names, as they could be overwritten by the constructor below.
        self.single_function = single_function
        self.whole_function = function
        self.parameter = parameter
        self.data = parameter['dataset']
        self.empirical_risk = [ParametricLossFunction(function=single_function, parameter=d)
                               for d in parameter['dataset']]
        super().__init__(function=function, parameter=parameter)

    def compute_stochastic_gradient(self, x, batch_size=1, return_loss=False, *args, **kwargs):

        # Get first batch_size elements of random permutation of numbers 0 to N-1
        idx = torch.randperm(self.N)[:batch_size]

        # Select corresponding datapoints
        cur_xes, cur_yes = self.parameter['x_values'][idx], self.parameter['y_values'][idx]
        cur_parameter = self.parameter.copy()
        cur_parameter['x_values'], cur_parameter['y_values'] = cur_xes, cur_yes
        cur_parameter['dataset'] = torch.hstack((cur_xes, cur_yes))

        # Setup function and compute grad (as usual)
        current_f = ParametricLossFunction(function=self.whole_function, parameter=cur_parameter)
        if return_loss:
            return current_f.compute_gradient(x), current_f(x)
        else:
            return current_f.compute_gradient(x)
