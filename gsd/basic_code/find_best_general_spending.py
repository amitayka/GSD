import numpy as np
import numpy.typing as npt
from scipy.optimize import differential_evolution

from gsd.basic_code.gsd_statistics_calculator import get_statistics_given_thresholds
from gsd.basic_code.gsd_threshold_finder_algorithm_1 import (
    get_efficacy_futility_thresholds,
)


def convert_simple_parameter_to_spending_function(
    simple_parameter: np.ndarray, alpha: float
) -> np.ndarray:
    """
    Convert a simple parameter to spending.
    simple parameter: 1D array of shape (n_looks - 1)
    alpha: the total spending value.
    Returns: 1D array of shape (n_looks) which is the spending values at each interval.
    """
    n_looks = len(simple_parameter) + 1
    spending = np.zeros(n_looks)

    cumulative_spending = spending[0] = alpha * simple_parameter[0]
    for i in range(1, n_looks - 1):
        spending[i] = simple_parameter[i] * (alpha - cumulative_spending)
        cumulative_spending += spending[i]
    spending[-1] = alpha - cumulative_spending
    return spending


def convert_spending_function_to_simple_parameter(
    spending: np.ndarray, alpha: float
) -> np.ndarray:
    """
    converts the spending function to a simple parameter.
    spending: 1D array of shape (n_looks)
    alpha: the total spending value.
    Returns: 1D array of shape (n_looks - 1) which is the simple parameter values at each interval.
    """
    n_looks = len(spending)
    simple_parameter = np.zeros(n_looks - 1)
    cumulative_spending = np.cumsum(spending)
    simple_parameter[0] = spending[0] / alpha
    for i in range(1, n_looks - 1):
        simple_parameter[i] = spending[i] / (alpha - cumulative_spending[i - 1])
    return simple_parameter


def find_best_general_spending(
    samples_h0: npt.NDArray[np.float64],
    samples_h1: npt.NDArray[np.float64],
    looks_fractions: npt.NDArray[np.float64],
    n_samples_per_arm_per_look: int,
    alpha: float,
    beta: float,
    is_binding: bool,
    null_weight: float = 0.5,
    alt_weight: float = 0.5,
    seed: int = 1729,
    verbose=False,
):
    """
    Find the best general spending strategy based on given parameters.
    The function uses differential evolution to optimize the spending strategy.
    samples_h0: 3D array of shape (n_trials, n_treatment_arms, n_looks) for the null hypothesis.
    samples_h1: 3D array of shape (n_trials, n_treatment_arms, n_looks) for the alternative hypothesis.
    looks_fractions: 1D array of shape (n_looks) the fraction of the total sample size (or of the information) at each look.
    n_samples_per_arm_per_look: int, the number of samples per arm per look.
    alpha: float, the total spending value for the null hypothesis.
    beta: float, the total spending value for the alternative hypothesis.
    is_binding: bool, whether the spending strategy is binding or not.
    null_weight: float, the weight for the null hypothesis.
    alt_weight: float, the weight for the alternative hypothesis.
    seed: int, the random seed for reproducibility.
    verbose: bool, whether to print the evaluations and outputs.
    Returns: a tuple of
        alpha_spending_differential_evolution: 1D array of shape (n_looks) for the best spending strategy for alpha.
        beta_spending_differential_evolution: 1D array of shape (n_looks) for the best spending strategy for beta.
        differential_evolution_result: the result of the differential evolution optimization.
    """
    global function_evaluations
    function_evaluations = 0

    def objective_function_best_spending(
        input: np.ndarray,
    ) -> float:
        """
        This function is the objective function for the differential evolution optimization.
        It calculates the cost based on the spending strategies and the samples for the null and alternative hypotheses.
        input: 1D array of shape (2 * (n_looks - 1)), where the first half is the simple alpha spending and
              the second half is the simple beta spending.
        Returns: a float value representing the cost, which is the sum of the average sample sizes for the null and alternative hypotheses.
        """
        global function_evaluations
        function_evaluations += 1
        assert len(input) == 2 * (len(looks_fractions) - 1)
        simple_alpha = input[: len(looks_fractions) - 1]
        simple_beta = input[len(looks_fractions) - 1 :]
        alpha_spending = convert_simple_parameter_to_spending_function(
            simple_alpha, alpha
        )
        beta_spending = convert_simple_parameter_to_spending_function(simple_beta, beta)
        efficacy_thresholds, futility_thresholds = get_efficacy_futility_thresholds(
            samples_h0, samples_h1, alpha_spending, beta_spending, is_binding=is_binding
        )
        stats_h1 = get_statistics_given_thresholds(
            samples_h1,
            efficacy_thresholds,
            futility_thresholds,
            n_samples_per_look_per_arm=n_samples_per_arm_per_look,
        )
        power = np.sum(stats_h1.efficacy_probs_trial_per_look)
        if power < 1 - beta:
            if verbose:
                print(
                    f"evaluations= {function_evaluations}, input = {input}, output = {10000*( 2 - power)}"
                )
            return 10000 * (
                2 - power
            )  # return a large cost if the power is too low, punishing lower power
        stats_h0 = get_statistics_given_thresholds(
            samples_h0,
            efficacy_thresholds,
            futility_thresholds,
            n_samples_per_look_per_arm=n_samples_per_arm_per_look,
        )
        cost = (
            null_weight * stats_h0.average_sample_size
            + alt_weight * stats_h1.average_sample_size
        )
        if verbose:
            print(
                f"evaluations= {function_evaluations}, input = {input}, output = {cost}"
            )
        return cost

    bounds = [(0, 1)] * (2 * (len(looks_fractions) - 1))
    differential_evolution_result = differential_evolution(
        objective_function_best_spending, bounds, seed=seed
    )

    best_result_differential_evolution = differential_evolution_result.x
    simple_alpha = best_result_differential_evolution[: len(looks_fractions) - 1]
    simple_beta = best_result_differential_evolution[len(looks_fractions) - 1 :]
    alpha_spending_differential_evolution = (
        convert_simple_parameter_to_spending_function(simple_alpha, alpha)
    )
    beta_spending_differential_evolution = (
        convert_simple_parameter_to_spending_function(simple_beta, beta)
    )

    return (
        alpha_spending_differential_evolution,
        beta_spending_differential_evolution,
        differential_evolution_result,
    )
