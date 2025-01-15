from typing import Callable, List, Dict, Tuple
from classes.StoppingCriterion.class_StoppingCriterion import StoppingCriterion
from classes.Constraint.class_Constraint import Constraint
from classes.OptimizationAlgorithm.derived_classes.subclass_ParametricOptimizationAlgorithm import (
    ParametricOptimizationAlgorithm)
from classes.LossFunction.derived_classes.subclass_ParametricLossFunction import LossFunction, ParametricLossFunction
from classes.Constraint.class_Constraint import create_list_of_constraints_from_functions
import torch
import torch.nn as nn
from tqdm import tqdm


def kl(prior: torch.Tensor, posterior: torch.Tensor) -> torch.Tensor:
    if len(prior) != len(posterior):
        raise RuntimeError("Posterior and prior have differing support.")

    if (torch.any(prior < 0)) or (torch.any(posterior < 0)):
        raise RuntimeError("Distributions are miss-specified.")

    if ((not torch.allclose(torch.sum(prior), torch.tensor(1.0)))
            or (not torch.allclose(torch.sum(posterior), torch.tensor(1.0)))):
        raise RuntimeError("Distributions are no probability distributions.")

    return torch.sum(posterior * torch.log(posterior / prior))


# To generalize this to \capital_lambda with more than one point, which is avoided here by first estimating a
# sufficiently good lambda, one needs to add + torch.log(covering_number) in the numerator.
# Here, it is torch.log(1.0) = 0.
def get_pac_bound_as_function_of_lambda(
        posterior_risk: torch.Tensor,
        prior: torch.Tensor,
        posterior: torch.Tensor,
        eps: torch.Tensor,
        n: int,
        upper_bound: torch.Tensor | int) -> Callable[[torch.Tensor], torch.Tensor]:

    if (eps >= 1.0) or (eps < 0):
        raise RuntimeError("Parameter eps does not lie in [0,1].")

    if posterior_risk > upper_bound:
        raise RuntimeError("Upper bound is smaller than risk.")

    return lambda lamb: (posterior_risk + (kl(posterior, prior) - torch.log(eps)) / lamb
                         + 0.5 * lamb * upper_bound ** 2 / n)


def phi_inv(q: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    return (1 - torch.exp(-a*q)) / (1 - torch.exp(-a))


def specify_test_points() -> torch.Tensor:
    # This could be adjusted based on computational constraints. Here, it was just fixed for simplicity.
    return torch.linspace(start=1e-3, end=1e2, steps=75000)


def minimize_upper_bound_in_lambda(pac_bound_function: Callable[[torch.Tensor], torch.Tensor],
                                   test_points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

    values_upper_bound = torch.stack([pac_bound_function(lamb) for lamb in test_points])
    best_upper_bound = torch.min(values_upper_bound)
    lambda_opt = test_points[torch.argmin(values_upper_bound)]

    if lambda_opt == test_points[0]:
        print("Note: Optimal lambda found at left boundary!")
    if lambda_opt == test_points[-1]:
        print("Note: Optimal lambda found at right boundary!")

    return best_upper_bound, lambda_opt


def get_splitting_index(loss_functions: List[LossFunction]) -> Tuple[int, int, int]:
    n_half = int(len(loss_functions) / 2)
    N_1, N_2 = len(loss_functions[:n_half]), len(loss_functions[n_half:])
    return n_half, N_1, N_2


def compute_pac_bound(posterior_risk: torch.Tensor,
                      prior: torch.Tensor,
                      posterior: torch.Tensor,
                      eps: torch.Tensor,
                      n: int,
                      upper_bound: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

    if posterior_risk > upper_bound:
        raise RuntimeError("Upper is smaller than risk.")

    if len(posterior) != len(prior):
        raise RuntimeError("Distributions do not match in length.")

    if (eps >= 1.0) or (eps < 0):
        raise RuntimeError("Parameter eps does not lie in [0,1].")

    if n < 1:
        raise RuntimeError("Parameter n appears to be too small.")

    test_points = specify_test_points()
    pac_bound_function = get_pac_bound_as_function_of_lambda(posterior_risk=posterior_risk, prior=prior,
                                                             posterior=posterior, eps=eps, n=n,
                                                             upper_bound=upper_bound)
    best_upper_bound, lambda_opt = minimize_upper_bound_in_lambda(pac_bound_function=pac_bound_function,
                                                                  test_points=test_points)

    return best_upper_bound, lambda_opt


def build_final_prior(potentials_1: torch.Tensor,
                      potentials_2: torch.Tensor,
                      n_1: int,
                      n_2: int) -> Tuple[torch.Tensor, torch.Tensor]:

    if len(potentials_1) != len(potentials_2):
        raise RuntimeError("Potentials do not have the same length.")

    if (n_1 <= 0) or (n_2 <= 0):
        raise RuntimeError("Number of datapoints seems to be wrong.")

    # Note that, to get the empirical risk here, one cannot just add the prior_potentials, but one has to reweight
    # them accordingly.
    prior_potentials = (n_1 * potentials_1 + n_2 * potentials_2) / (n_1 + n_2)
    prior = torch.softmax(prior_potentials, dim=0)
    return prior, prior_potentials


class PacBayesOptimizationAlgorithm(ParametricOptimizationAlgorithm):

    def __init__(self,
                 initial_state: torch.Tensor,
                 implementation: nn.Module,
                 stopping_criterion: StoppingCriterion,
                 loss_function: ParametricLossFunction,
                 # sufficient_statistics: Callable,
                 # natural_parameters: Callable,
                 # covering_number: torch.Tensor,
                 epsilon: torch.Tensor,
                 n_max: int,
                 constraint: Constraint = None):

        super().__init__(initial_state=initial_state, implementation=implementation, loss_function=loss_function,
                         constraint=constraint)
        self.set_stopping_criterion(stopping_criterion)
        self.n_max = n_max
        # self.sufficient_statistics = sufficient_statistics
        # self.natural_parameters = natural_parameters
        # self.covering_number = covering_number
        self.epsilon = epsilon
        self.pac_bound = None
        self.optimal_lambda = None

    def compute_convergence_time_and_contraction_rate(self) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.loss_function.get_parameter().get('opt_val') is None:
            raise RuntimeError("Optimal value not given.")

        # Compute loss over iterates and append final loss to list
        init_loss = self.evaluate_loss_function_at_current_iterate()
        convergence_time = self.compute_convergence_time(num_steps_max=self.n_max)
        final_loss = self.evaluate_loss_function_at_current_iterate()

        if convergence_time == 0:
            return torch.tensor(0.0), torch.tensor(0.0)

        # Subtract optimal loss (use absolute value, as optimal loss is also approximated; to avoid negative values)
        init_loss = torch.abs(init_loss - self.loss_function.get_parameter()['opt_val'])
        final_loss = torch.abs(final_loss - self.loss_function.get_parameter()['opt_val'])

        contraction_factor = (final_loss.detach() / init_loss.detach()) ** (1 / convergence_time)

        # For further computations, transform convergence time to float-tensor
        return torch.tensor(convergence_time).float(), contraction_factor

    def evaluate_convergence_risk(self,
                                  loss_functions: List[ParametricLossFunction],
                                  constraint_functions: List[Constraint]
                                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        rates, probabilities, stopping_times = [], [], []

        for loss_func, constraint_func in zip(loss_functions, constraint_functions):

            self.reset_state_and_iteration_counter()
            self.set_loss_function(loss_func)

            # If the constraint is not satisfied, one does not have to compute the losses
            # as they do only occur as a 0 in the convergence risk. Note that one has to append 0 here,
            # as later on, we take the mean, which takes the NUMBER OF LOSSES into account,
            # i.e. the final output would be too large, if one does not include 0.
            if not constraint_func(self):
                rates.append(torch.tensor(0.0))  # Take zero, because it involves the indicator function.
                probabilities.append(torch.tensor(0.))
                # If the algorithm does not converge, it gets stopped after n_max iterations.
                stopping_times.append(torch.tensor(self.n_max).float())
                continue

            probabilities.append(torch.tensor(1.))
            convergence_time, contraction_factor = self.compute_convergence_time_and_contraction_rate()
            stopping_times.append(convergence_time)
            rates.append(contraction_factor)

        return (torch.mean(torch.stack(rates)),
                torch.mean(torch.stack(probabilities)),
                torch.mean(torch.stack(stopping_times)))

    def evaluate_potentials(self,
                            loss_functions: List[ParametricLossFunction],
                            constraint_functions: List[Constraint],
                            state_dict_samples: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        convergence_rates, convergence_probabilities, stopping_times = [], [], []
        pbar = tqdm(state_dict_samples)
        pbar.set_description('Computing prior potentials')
        for hp in pbar:

            self.set_hyperparameters_to(hp)
            convergence_rate, convergence_probability, stopping_time = self.evaluate_convergence_risk(
                loss_functions=loss_functions, constraint_functions=constraint_functions)
            convergence_rates.append(-convergence_rate)  # Take -1, because we use it as potential in torch.exp(...)
            convergence_probabilities.append(convergence_probability)
            stopping_times.append(stopping_time)

        return torch.tensor(convergence_rates), torch.tensor(convergence_probabilities), torch.tensor(stopping_times)

    def estimate_lambda_for_convergence_rate(self,
                                             prior: torch.Tensor,
                                             posterior: torch.Tensor,
                                             potentials_independent_from_prior: torch.Tensor,
                                             size_of_training_data: int,
                                             constraint_parameters: Dict) -> torch.Tensor:

        # Note that we use epsilon/3 here, as we want three Pac-bounds holding simultaneously, that is,
        # we use a union-bound argument with epsilon/3 to get 3 * (epsilon/3) = epsilon.
        # We have to take *-1 here, because the rates are stored as negative values (for potentials).
        prelim_rate = torch.sum(posterior * (-potentials_independent_from_prior))
        _, lambda_rate = compute_pac_bound(
            posterior_risk=prelim_rate,
            prior=prior, posterior=posterior,
            eps=self.epsilon / 3,
            n=size_of_training_data, upper_bound=constraint_parameters['bound'])

        return lambda_rate

    def estimate_lambda_for_convergence_time(self,
                                             prior: torch.Tensor,
                                             posterior: torch.Tensor,
                                             potentials_independent_from_prior: torch.Tensor,
                                             size_of_training_data: int) -> torch.Tensor:

        # Note that we use epsilon/3 here, as we want three Pac-bounds holding simultaneously, that is,
        # we use a union-bound argument with epsilon/3 to get 3 * (epsilon/3) = epsilon.
        posterior_time = torch.sum(posterior * potentials_independent_from_prior)
        _, lambda_time = compute_pac_bound(
            posterior_risk=posterior_time, prior=prior, posterior=posterior, eps=self.epsilon / 3,
            n=size_of_training_data, upper_bound=torch.tensor(self.n_max))

        return lambda_time

    def estimate_lambda_for_convergence_probability(self,
                                                    prior: torch.Tensor,
                                                    posterior: torch.Tensor,
                                                    potentials_independent_from_prior: torch.Tensor,
                                                    size_of_training_data: int) -> torch.Tensor:

        # Note that we use epsilon/3 here, as we want three Pac-bounds holding simultaneously, that is,
        # we use a union-bound argument with epsilon/3 to get 3 * (epsilon/3) = epsilon.
        posterior_prob = torch.sum(posterior * (1 - potentials_independent_from_prior))
        kl_eps = kl(prior=prior, posterior=posterior) - torch.log(self.epsilon / 3)
        lambdas_to_test = torch.linspace(1, 1000, 100000)
        values = torch.tensor([phi_inv(q=posterior_prob + kl_eps / lamb, a=lamb / size_of_training_data)
                               for lamb in lambdas_to_test])
        lambda_prob = lambdas_to_test[torch.argmin(values)]
        return lambda_prob

    def get_preliminary_prior_distribution(self,
                                           loss_functions: List[ParametricLossFunction],
                                           constraints: List[Constraint],
                                           state_dict_samples: List[Dict]
                                           ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rates, conv_probs, stopping_times = self.evaluate_potentials(
            loss_functions=loss_functions, constraint_functions=constraints, state_dict_samples=state_dict_samples)
        prior = torch.softmax(rates, dim=0)
        return prior, rates, conv_probs, stopping_times

    def get_preliminary_posterior_distribution(self,
                                               loss_functions: List[ParametricLossFunction],
                                               prior_rates: torch.Tensor,
                                               constraints: List[Constraint],
                                               state_dict_samples: List[Dict]
                                               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rates, conv_probs, stopping_times = self.evaluate_potentials(
            loss_functions=loss_functions, constraint_functions=constraints, state_dict_samples=state_dict_samples)
        prelim_posterior = torch.softmax(prior_rates + rates, dim=0)
        return prelim_posterior, rates, conv_probs, stopping_times

    def get_estimates_for_lambdas_and_build_prior(
            self,
            loss_functions_prior: List[ParametricLossFunction],
            state_dict_samples_prior: List[Dict],
            constraint_parameters: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        constraint_functions_prior = create_list_of_constraints_from_functions(
            describing_property=constraint_parameters['describing_property'],
            list_of_functions=loss_functions_prior)

        # Construct a preliminary prior and a preliminary posterior, such that we can use these to estimate good values
        # for the parameters \lambda
        n_half, N_1, N_2 = get_splitting_index(loss_functions=loss_functions_prior)
        (prelim_prior,
         prior_rates_dependent,
         prior_conv_probs_dependent,
         prior_stopping_times_dependent) = self.get_preliminary_prior_distribution(
            loss_functions=loss_functions_prior[:n_half], constraints=constraint_functions_prior[:n_half],
            state_dict_samples=state_dict_samples_prior)
        (prelim_posterior,
         prior_rates_independent,
         prior_conv_probs_independent,
         prior_stopping_times_independent) = self.get_preliminary_posterior_distribution(
            loss_functions=loss_functions_prior[:n_half], constraints=constraint_functions_prior[:n_half],
            prior_rates=prior_rates_dependent, state_dict_samples=state_dict_samples_prior)

        # Get an estimate for all three lambdas by using this preliminary prior and posterior
        lambda_rate = self.estimate_lambda_for_convergence_rate(
            prior=prelim_prior, posterior=prelim_posterior,
            potentials_independent_from_prior=prior_rates_independent,
            size_of_training_data=n_half, constraint_parameters=constraint_parameters)
        lambda_time = self.estimate_lambda_for_convergence_time(
            prior=prelim_prior, posterior=prelim_posterior,
            potentials_independent_from_prior=prior_stopping_times_independent,
            size_of_training_data=n_half)
        lambda_prob = self.estimate_lambda_for_convergence_probability(
            prior=prelim_prior, posterior=prelim_posterior,
            potentials_independent_from_prior=prior_conv_probs_independent,
            size_of_training_data=n_half)

        final_prior, final_prior_potentials = build_final_prior(potentials_1=prior_rates_dependent,
                                                                potentials_2=prior_rates_independent,
                                                                n_1=N_1, n_2=N_2)

        return final_prior, final_prior_potentials, lambda_rate, lambda_time, lambda_prob

    def build_posterior(self,
                        loss_functions_train: List[ParametricLossFunction],
                        state_dict_samples_prior: List[Dict],
                        prior_potentials: torch.Tensor,
                        constraint_parameters: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        constraint_functions_posterior = create_list_of_constraints_from_functions(
            describing_property=constraint_parameters['describing_property'],
            list_of_functions=loss_functions_train)

        posterior_potentials, convergence_probabilities, stopping_times = self.evaluate_potentials(
            loss_functions=loss_functions_train,
            constraint_functions=constraint_functions_posterior,
            state_dict_samples=state_dict_samples_prior)

        posterior = torch.softmax(posterior_potentials + prior_potentials, dim=0)

        return posterior, posterior_potentials, stopping_times, convergence_probabilities

    def select_optimal_hyperparameters(self,
                                       state_dict_samples_prior: List[Dict],
                                       posterior: torch.Tensor) -> None:
        alpha_opt = state_dict_samples_prior[torch.argmax(posterior)]
        self.implementation.load_state_dict(alpha_opt)

    def compute_pac_bound_for_convergence_rate(self,
                                               prior: torch.Tensor,
                                               posterior: torch.Tensor,
                                               potentials_that_are_independent_from_prior: torch.Tensor,
                                               lambda_rate: torch.Tensor,
                                               size_of_training_data: int,
                                               constraint_parameters: Dict) -> torch.Tensor:
        posterior_risk_convergence_rate = torch.sum(posterior * (-potentials_that_are_independent_from_prior))
        pac_bound_function_for_rate = get_pac_bound_as_function_of_lambda(
            posterior_risk=posterior_risk_convergence_rate, prior=prior, posterior=posterior, eps=self.epsilon / 3,
            n=size_of_training_data, upper_bound=constraint_parameters['bound'])
        pac_bound_rate = pac_bound_function_for_rate(lambda_rate)
        return pac_bound_rate

    def compute_pac_bound_for_convergence_time(self,
                                               prior: torch.Tensor,
                                               posterior: torch.Tensor,
                                               potentials_that_are_independent_from_prior: torch.Tensor,
                                               lambda_time: torch.Tensor,
                                               size_of_training_data: int):
        posterior_risk_convergence_time = torch.sum(posterior * potentials_that_are_independent_from_prior)
        pac_bound_function_for_time = get_pac_bound_as_function_of_lambda(
            posterior_risk=posterior_risk_convergence_time, prior=prior, posterior=posterior, eps=self.epsilon / 3,
            n=size_of_training_data, upper_bound=self.n_max)
        pac_bound_time = pac_bound_function_for_time(lambda_time)
        return pac_bound_time

    def compute_pac_bound_for_convergence_probability(self,
                                                      prior: torch.Tensor,
                                                      posterior: torch.Tensor,
                                                      potentials_that_are_independent_from_prior: torch.Tensor,
                                                      lambda_prob: torch.Tensor,
                                                      size_of_training_data: int):
        posterior_risk_convergence_probability = torch.sum(posterior * (1 - potentials_that_are_independent_from_prior))
        kl_eps = kl(prior=prior, posterior=posterior) - torch.log(self.epsilon)
        pac_bound_convergence_probability = phi_inv(q=posterior_risk_convergence_probability + kl_eps / lambda_prob,
                                                    a=lambda_prob / size_of_training_data)
        return pac_bound_convergence_probability

    def pac_bayes_fit(self,
                      loss_functions_prior: List[ParametricLossFunction],
                      loss_functions_train: List[ParametricLossFunction],
                      fitting_parameters: Dict,
                      sampling_parameters_prior: Dict,
                      constraint_parameters: Dict,
                      update_parameters: Dict
                      ) -> (torch.Tensor, List, List, List, List):

        # Step 1: Find a point inside the constraint set that has a good performance.
        self.fit(loss_functions=loss_functions_prior,
                 fitting_parameters=fitting_parameters,
                 constraint_parameters=constraint_parameters,
                 update_parameters=update_parameters)

        # Step 2: Create the support of the prior distribution by:
        #   - sampling (with the same loss-function) with SGLD in a probabilistically constrained way
        #   - estimating \lambda
        _, state_dict_samples_prior, _ = self.sample_with_sgld(
            loss_functions=loss_functions_prior, parameters=sampling_parameters_prior)
        (prior,
         prior_potentials_rate,
         lambda_rate, lambda_time, lambda_prob) = self.get_estimates_for_lambdas_and_build_prior(
            loss_functions_prior=loss_functions_prior, state_dict_samples_prior=state_dict_samples_prior,
            constraint_parameters=constraint_parameters)

        # Step 3: Build posterior potentials and evaluate them on the given samples
        (posterior,
         potentials_rate,
         potentials_stopping_time,
         potentials_convergence_probability) = self.build_posterior(
            loss_functions_train=loss_functions_train, state_dict_samples_prior=state_dict_samples_prior,
            prior_potentials=prior_potentials_rate, constraint_parameters=constraint_parameters)

        self.select_optimal_hyperparameters(state_dict_samples_prior, posterior)

        # 4: Compute Guarantees
        pac_bound_rate = self.compute_pac_bound_for_convergence_rate(
            prior=prior, posterior=posterior, potentials_that_are_independent_from_prior=potentials_rate,
            lambda_rate=lambda_rate, size_of_training_data=len(loss_functions_train),
            constraint_parameters=constraint_parameters)
        pac_bound_time = self.compute_pac_bound_for_convergence_time(
            prior=prior, posterior=posterior, potentials_that_are_independent_from_prior=potentials_stopping_time,
            lambda_time=lambda_time, size_of_training_data=len(loss_functions_train))
        pac_bound_convergence_probability = self.compute_pac_bound_for_convergence_probability(
            prior=prior, posterior=posterior,
            potentials_that_are_independent_from_prior=potentials_convergence_probability,
            lambda_prob=lambda_prob, size_of_training_data=len(loss_functions_train))

        return pac_bound_rate, 1 - pac_bound_convergence_probability, pac_bound_time, state_dict_samples_prior
