from classes.LossFunction.derived_classes.subclass_ParametricLossFunction import ParametricLossFunction


class RegularizedParametricLossFunction(ParametricLossFunction):

    def __init__(self, function, data_fidelity, regularizer, parameter):
        self.parameter = parameter
        self.function = function
        self.data_fidelity = ParametricLossFunction(function=data_fidelity, parameter=self.parameter)
        self.regularizer = ParametricLossFunction(function=regularizer, parameter=self.parameter)
        super().__init__(function=self.function, parameter=parameter)

    def compute_gradient_of_data_fidelity(self, x, *args, **kwargs):
        return self.data_fidelity.compute_gradient(x, *args, **kwargs)

    def compute_gradient_of_regularizer(self, x, *args, **kwargs):
        return self.regularizer.compute_gradient(x, *args, **kwargs)
