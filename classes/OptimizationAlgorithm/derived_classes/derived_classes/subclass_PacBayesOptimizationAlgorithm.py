from typing import Callable, List, Dict
from classes.StoppingCriterion.class_StoppingCriterion import StoppingCriterion
from classes.Constraint.class_Constraint import Constraint
from classes.OptimizationAlgorithm.derived_classes.subclass_ParametricOptimizationAlgorithm import (
    ParametricOptimizationAlgorithm)
from classes.LossFunction.derived_classes.subclass_ParametricLossFunction import ParametricLossFunction
from classes.Constraint.class_Constraint import create_list_of_constraints_from_functions
import torch
import torch.nn as nn
from tqdm import tqdm


def kl(prior, posterior):
    return torch.sum(posterior * torch.log(posterior / prior))


# To generalize this to \capital_lambda with more than one point, which is avoided here by first estimating a
# sufficiently good lambda, one needs to add + torch.log(covering_number) in the numerator.
# Here, it is torch.log(1.0) = 0.
def get_pac_bound_as_function_of_lambda(posterior_risk, prior, posterior, eps, n, upper_bound) -> Callable:
    return lambda lamb: (posterior_risk + (kl(posterior, prior) - torch.log(eps)) / lamb
                         + 0.5 * lamb * upper_bound ** 2 / n)


def phi_inv(q, a):
    return (1 - torch.exp(-a*q)) / (1 - torch.exp(-a))


def specify_test_points():
    # This could be adjusted based on computational constraints. Here, it was just fixed for simplicity.
    return torch.linspace(start=1e-3, end=1e2, steps=75000)


def minimize_upper_bound_in_lambda(pac_bound_function, test_points):

    values_upper_bound = torch.stack([pac_bound_function(lamb) for lamb in test_points])
    best_upper_bound = torch.min(values_upper_bound)
    lambda_opt = test_points[torch.argmin(values_upper_bound)]

    if lambda_opt == test_points[0]:
        print("Note: Optimal lambda found at left boundary!")
    if lambda_opt == test_points[-1]:
        print("Note: Optimal lambda found at right boundary!")

    return best_upper_bound, lambda_opt


def compute_pac_bound(posterior_risk, prior, posterior, eps, n, upper_bound):

    test_points = specify_test_points()
    pac_bound_function = get_pac_bound_as_function_of_lambda(posterior_risk=posterior_risk, prior=prior,
                                                             posterior=posterior, eps=eps, n=n,
                                                             upper_bound=upper_bound)
    best_upper_bound, lambda_opt = minimize_upper_bound_in_lambda(pac_bound_function=pac_bound_function,
                                                                  test_points=test_points)

    return best_upper_bound, lambda_opt


class PacBayesOptimizationAlgorithm(ParametricOptimizationAlgorithm):

    def __init__(self,
                 initial_state: torch.Tensor,
                 implementation: nn.Module,
                 stopping_criterion: StoppingCriterion,
                 loss_function: ParametricLossFunction,
                 sufficient_statistics: Callable,
                 natural_parameters: Callable,
                 covering_number: torch.Tensor,
                 epsilon: torch.Tensor,
                 n_max: int,
                 constraint: Constraint = None):

        super().__init__(initial_state=initial_state, implementation=implementation, loss_function=loss_function,
                         constraint=constraint)
        self.set_stopping_criterion(stopping_criterion)
        self.n_max = n_max
        self.sufficient_statistics = sufficient_statistics
        self.natural_parameters = natural_parameters
        self.covering_number = covering_number
        self.epsilon = epsilon
        self.pac_bound = None
        self.optimal_lambda = None

    def compute_convergence_time_and_contraction_rate(self):

        # Compute loss over iterates and append final loss to list
        init_loss = self.evaluate_loss_function_at_current_iterate()
        convergence_time = self.compute_convergence_time(num_steps_max=self.n_max).detach()
        final_loss = self.evaluate_loss_function_at_current_iterate()

        # Subtract optimal loss (use absolute value, as optimal loss is also approximated; to avoid negative values)
        init_loss = torch.abs(init_loss - self.loss_function.get_parameter()['opt_val'])
        final_loss = torch.abs(final_loss - self.loss_function.get_parameter()['opt_val'])

        contraction_factor = (final_loss.detach() / init_loss.detach()) ** (1 / convergence_time)

        return convergence_time.float(), contraction_factor

    def evaluate_convergence_risk(self,
                                  loss_functions: List[ParametricLossFunction],
                                  constraint_functions: List[Constraint]
                                  ) -> torch.tensor:

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
                torch.mean(torch.stack(stopping_times).float()))

    def evaluate_potentials(self,
                            loss_functions,
                            constraint_functions,
                            state_dict_samples):

        potentials, convergence_probabilities, stopping_times = [], [], []
        pbar = tqdm(state_dict_samples)
        pbar.set_description('Computing prior potentials')
        for hp in pbar:

            self.set_hyperparameters_to(hp)
            convergence_risk, convergence_probability, stopping_time = self.evaluate_convergence_risk(
                loss_functions=loss_functions, constraint_functions=constraint_functions)
            potentials.append(-convergence_risk)
            convergence_probabilities.append(convergence_probability)
            stopping_times.append(stopping_time)

        return torch.tensor(potentials), torch.tensor(convergence_probabilities), torch.tensor(stopping_times)

    def get_estimates_for_lambdas_and_build_prior(self,
                                                  loss_functions_prior,
                                                  state_dict_samples_prior,
                                                  constraint_parameters):

        constraint_functions_prior = create_list_of_constraints_from_functions(
            describing_property=constraint_parameters['describing_property'],
            list_of_functions=loss_functions_prior)

        n_half = int(len(loss_functions_prior) / 2)
        N_1, N_2 = len(loss_functions_prior[:n_half]), len(loss_functions_prior[n_half:])
        prior_potentials_1, prior_conv_probs_1, prior_stopping_times_1 = self.evaluate_potentials(
            loss_functions=loss_functions_prior[:n_half],
            constraint_functions=constraint_functions_prior[:n_half],
            state_dict_samples=state_dict_samples_prior)

        prior_potentials_2, prior_conv_probs_2, prior_stopping_times_2 = self.evaluate_potentials(
            loss_functions=loss_functions_prior[n_half:],
            constraint_functions=constraint_functions_prior[n_half:],
            state_dict_samples=state_dict_samples_prior)

        prelim_prior = torch.softmax(prior_potentials_1, dim=0)
        prelim_posterior = torch.softmax(prior_potentials_1 + prior_potentials_2, dim=0)

        # Estimate all three lambdas
        prelim_rate = torch.sum(prelim_posterior * (-prior_potentials_2))
        print(f"Prelim Rate = {prelim_rate}")
        _, lambda_rate = compute_pac_bound(
            posterior_risk=prelim_rate,
            prior=prelim_prior, posterior=prelim_posterior,
            eps=self.epsilon / 3,
            n=n_half, upper_bound=constraint_parameters['bound'])

        print(f"Prelim Lambda Rate = {lambda_rate}")

        posterior_time = torch.sum(prelim_posterior * prior_stopping_times_2)
        _, lambda_time = compute_pac_bound(
            posterior_risk=posterior_time, prior=prelim_prior, posterior=prelim_posterior, eps=self.epsilon / 3,
            n=n_half, upper_bound=self.n_max)

        posterior_prob = torch.sum(prelim_posterior * (1 - prior_conv_probs_2))

        # Note that we use epsilon/3 here, as we want three Pac-bounds holding simultaneously, that is,
        # we use a union-bound argument with epsilon/3 to get 3 * (epsilon/3) = epsilon. Similarly below.
        kl_eps = kl(prior=prelim_prior, posterior=prelim_posterior) - torch.log(self.epsilon / 3)
        lambdas_to_test = torch.linspace(1, 1000, 100000)
        values = torch.tensor([phi_inv(q=posterior_prob + kl_eps / lamb, a=lamb / n_half)
                               for lamb in lambdas_to_test])
        lambda_prob = lambdas_to_test[torch.argmin(values)]

        # Note that, to get the empirical risk here, one cannot just add the prior_potentials, but one has to reweight
        # them accordingly.
        prior_potentials = (N_1 * prior_potentials_1 + N_2 * prior_potentials_2) / (N_1 + N_2)
        prior = torch.softmax(prior_potentials, dim=0)

        return prior, prior_potentials, lambda_rate, lambda_time, lambda_prob

    def build_posterior(self, loss_functions_train, state_dict_samples_prior, prior_potentials, constraint_parameters):
        constraint_functions_posterior = create_list_of_constraints_from_functions(
            describing_property=constraint_parameters['describing_property'],
            list_of_functions=loss_functions_train)

        posterior_potentials, convergence_probabilities, stopping_times = self.evaluate_potentials(
            loss_functions=loss_functions_train,
            constraint_functions=constraint_functions_posterior,
            state_dict_samples=state_dict_samples_prior)

        posterior = torch.softmax(posterior_potentials + prior_potentials, dim=0)

        return posterior, posterior_potentials, stopping_times, convergence_probabilities

    def select_optimal_hyperparameters(self, state_dict_samples_prior, posterior):
        alpha_opt = state_dict_samples_prior[torch.argmax(posterior)]
        self.implementation.load_state_dict(alpha_opt)

    def pac_bayes_fit(self,
                      loss_functions_prior: List[ParametricLossFunction],
                      loss_functions_train: List[ParametricLossFunction],
                      fitting_parameters: Dict,
                      sampling_parameters_prior: Dict,
                      constraint_parameters: Dict,
                      update_parameters: Dict
                      ) -> (torch.tensor, List, List, List, List):

        # Step 1: Find a point inside the constraint set that has a good performance.
        # For this, perform empirical risk minimization on prior data.
        self.fit(loss_functions=loss_functions_prior,
                 fitting_parameters=fitting_parameters,
                 constraint_parameters=constraint_parameters,
                 update_parameters=update_parameters)

        # Step 2: Create the support of the prior distribution by:
        # 2.1: sampling (with the same loss-function) with SGLD in a probabilistically constrained way.
        _, state_dict_samples_prior, _ = self.sample_with_sgld(
            loss_functions=loss_functions_prior, parameters=sampling_parameters_prior)

        # 2.2 Estimate good values for each \lambda and build prior
        prior, prior_potentials, lambda_rate, lambda_time, lambda_prob = self.get_estimates_for_lambdas_and_build_prior(
            loss_functions_prior=loss_functions_prior, state_dict_samples_prior=state_dict_samples_prior,
            constraint_parameters=constraint_parameters)

        # 3: Build posterior potentials and evaluate them on the given samples
        posterior, posterior_potentials, stopping_times, convergence_probabilities = self.build_posterior(
            loss_functions_train=loss_functions_train, state_dict_samples_prior=state_dict_samples_prior,
            prior_potentials=prior_potentials, constraint_parameters=constraint_parameters)

        self.select_optimal_hyperparameters(state_dict_samples_prior, posterior)

        # 4: Compute Guarantees
        # 4.1: PAC-Bound for Rate
        posterior_rate = torch.sum(posterior * (-posterior_potentials))
        pac_bound_function_for_rate = get_pac_bound_as_function_of_lambda(
            posterior_risk=posterior_rate, prior=prior, posterior=posterior, eps=self.epsilon / 3,
            n=len(loss_functions_train), upper_bound=constraint_parameters['bound'])
        pac_bound_rate = pac_bound_function_for_rate(lambda_rate)

        # 4.2: PAC-Bound for Probability
        # 1 - ... because of the UPPER bound, i.e. take the complementary event
        posterior_prob = torch.sum(posterior * (1 - convergence_probabilities))
        kl_eps = kl(prior=prior, posterior=posterior) - torch.log(self.epsilon)
        pac_bound_convergence_probability = phi_inv(q=posterior_prob + kl_eps / lambda_prob,
                                                    a=lambda_prob / len(loss_functions_train))

        # 4.3: PAC-Bound for Convergence Time
        posterior_time = torch.sum(posterior * stopping_times)
        pac_bound_function_for_time = get_pac_bound_as_function_of_lambda(
            posterior_risk=posterior_time, prior=prior, posterior=posterior, eps=self.epsilon / 3,
            n=len(loss_functions_train), upper_bound=self.n_max)
        pac_bound_time = pac_bound_function_for_time(lambda_time)

        return pac_bound_rate, 1 - pac_bound_convergence_probability, pac_bound_time, state_dict_samples_prior
