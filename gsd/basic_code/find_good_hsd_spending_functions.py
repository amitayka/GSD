import numpy as np
import numpy.typing as npt
from scipy.optimize import differential_evolution

from gsd.basic_code.gsd_statistics_calculator import get_statistics_given_thresholds
from gsd.basic_code.gsd_threshold_finder_algorithm_1 import (
    get_efficacy_futility_thresholds,
)
from gsd.basic_code.utils.spending_function import (
    generate_spending_from_spending_parameter,
)


def find_best_hsd_spending(
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
    Find the best Hwang-Shih-Decani spending strategy based on given parameters.
    The function uses 2D differential evolution to optimize the spending strategy.
    samples_h0: 3D array of shape (n_trials, n_treatment_arms, n_looks) for the null hypothesis.
    samples_h1: 3D array of shape (n_trials, n_treatment_arms, n_looks) for the alternative hypothesis.
    looks_fractions: 1D array of shape (n_looks) the fraction of the total sample size (or of the information) at each look.
    n_samples_per_arm_per_look: 1D array of shape (n_looks) representing the number of samples per arm at each look.
    alpha: float, the total spending value for the null hypothesis.
    beta: float, the total spending value for the alternative hypothesis.
    is_binding: bool, whether the spending strategy is binding or not.
    null_weight: float, weight for the null hypothesis in the objective function.
    alt_weight: float, weight for the alternative hypothesis in the objective function.
    seed: int, random seed for reproducibility.
    verbose: bool, whether to print the optimization process.
    Returns: tuple of
        alpha_spending_parameter: float, the optimized alpha spending parameter.
        beta_spending_parameter: float, the optimized beta spending parameter.
        differential_evolution_result: the result of the differential evolution optimization.
    """
    global func_evaluations
    func_evaluations = 0

    def objective_function(params: np.ndarray) -> float:
        """
        This function is the objective function for the differential evolution optimization.
        It calculates the cost based on the spending strategies and the samples for the null and alternative hypotheses.
        params: 1D array of shape (2,), where the first element is the alpha spending parameter
                    and the second element is the beta spending parameter.
        Returns: a float value representing the cost, which is the sum of the average sample sizes for
                    the null and alternative hypotheses.
        """
        global func_evaluations
        func_evaluations += 1
        alpha_spending_parameter = params[0]
        beta_spending_parameter = params[1]
        alpha_spending = generate_spending_from_spending_parameter(
            alpha_spending_parameter, alpha, looks_fractions
        )
        beta_spending = generate_spending_from_spending_parameter(
            beta_spending_parameter, beta, looks_fractions
        )
        efficacy_thresholds, futility_thresholds = get_efficacy_futility_thresholds(
            samples_h0,
            samples_h1,
            alpha_spending,
            beta_spending,
            is_binding=is_binding,
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
                    "evaluations = ",
                    func_evaluations,
                    "Not enough power, alpha_spending_parameter = ",
                    alpha_spending_parameter,
                    "beta_spending_parameter = ",
                    beta_spending_parameter,
                    "power = ",
                    power,
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
        # We optimize total average sample size, under h0 and h1.
        trial_cost = (
            null_weight * stats_h0.average_sample_size
            + alt_weight * stats_h1.average_sample_size
        )
        # Change here for a different objective function.
        if verbose:
            print(
                "evaluations = ",
                func_evaluations,
                "alpha_spending_parameter = ",
                alpha_spending_parameter,
                "beta_spending_parameter = ",
                beta_spending_parameter,
                "average_sample_size = ",
                trial_cost,
            )
        return trial_cost

    bounds = [(-10, 4), (-10, 4)]

    differential_evolution_result = differential_evolution(
        objective_function, bounds=bounds, seed=seed
    )

    alpha_spending_parameter = differential_evolution_result.x[0]
    beta_spending_parameter = differential_evolution_result.x[1]

    return (
        alpha_spending_parameter,
        beta_spending_parameter,
        differential_evolution_result,
    )


def find_beta_for_alpha(
    alpha_spending_parameter: float,
    samples_h0: npt.NDArray[np.float64],
    samples_h1: npt.NDArray[np.float64],
    looks_fractions: npt.NDArray[np.float64],
    n_samples_per_arm_per_look: int,
    alpha: float,
    beta: float,
    is_binding: bool,
    beta_spending_lower_bound: float = -10,
    beta_spending_upper_bound: float = 4,
    n_iterations: int = 20,  # number of iterations for binary search
) -> float:
    """
    Find the beta spending parameter for a given alpha spending parameter.
    """
    alpha_spending = generate_spending_from_spending_parameter(
        alpha_spending_parameter, alpha, looks_fractions
    )

    good_beta = beta_spending_lower_bound
    for _ in range(n_iterations):
        beta_spending_parameter = (
            beta_spending_lower_bound + beta_spending_upper_bound
        ) / 2

        beta_spending = generate_spending_from_spending_parameter(
            beta_spending_parameter, beta, looks_fractions
        )
        efficacy_thresholds, futility_thresholds = get_efficacy_futility_thresholds(
            samples_h0,
            samples_h1,
            alpha_spending,
            beta_spending,
            is_binding=is_binding,
        )
        stats = get_statistics_given_thresholds(
            samples_h1,
            efficacy_thresholds,
            futility_thresholds,
            n_samples_per_look_per_arm=n_samples_per_arm_per_look,
        )
        calculated_power = np.sum(stats.efficacy_probs_trial_per_look)
        if calculated_power >= 1 - beta:
            beta_spending_lower_bound = beta_spending_parameter
            good_beta = beta_spending_parameter
        else:
            beta_spending_upper_bound = beta_spending_parameter
    return good_beta


def find_best_hsd_spending_alternative(
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
    Find the best Hwang-Shih-Decani spending strategy based on given parameters.
    the function uses 1D differential evolution to optimize the spending strategy over the alpha spending parameter,
    and then finds the beta spending parameter using a binary search.
    """
    global func_evaluations
    func_evaluations = 0

    def objective_function(alpha_spending_parameter_ar: np.ndarray) -> float:
        global func_evaluations
        func_evaluations += 1
        alpha_spending_parameter = alpha_spending_parameter_ar[0]
        # return alpha_spending_parameter * alpha_spending_parameter
        beta_spending_parameter = find_beta_for_alpha(
            alpha_spending_parameter,
            samples_h0,
            samples_h1,
            looks_fractions,
            n_samples_per_arm_per_look,
            alpha,
            beta,
            is_binding,
        )
        alpha_spending = generate_spending_from_spending_parameter(
            alpha_spending_parameter, alpha, looks_fractions
        )
        beta_spending = generate_spending_from_spending_parameter(
            beta_spending_parameter, beta, looks_fractions
        )
        efficacy_thresholds, futility_thresholds = get_efficacy_futility_thresholds(
            samples_h0,
            samples_h1,
            alpha_spending,
            beta_spending,
            is_binding=is_binding,
        )
        stats_h1 = get_statistics_given_thresholds(
            samples_h1,
            efficacy_thresholds,
            futility_thresholds,
            n_samples_per_look_per_arm=n_samples_per_arm_per_look,
        )
        if np.sum(stats_h1.efficacy_probs_trial_per_look) < 1 - beta:
            if verbose:
                print(
                    "evaluations = ",
                    func_evaluations,
                    "Not enough power, alpha_spending_parameter = ",
                    alpha_spending_parameter,
                    "beta_spending_parameter = ",
                    beta_spending_parameter,
                    "power = ",
                    np.sum(stats_h1.efficacy_probs_trial_per_look),
                )
            return 10000 * (2 - np.sum(stats_h1.efficacy_probs_trial_per_look))
        stats_h0 = get_statistics_given_thresholds(
            samples_h0,
            efficacy_thresholds,
            futility_thresholds,
            n_samples_per_look_per_arm=n_samples_per_arm_per_look,
        )
        # We optimize total average sample size, under h0 and h1.
        trial_cost = (
            null_weight * stats_h0.average_sample_size
            + alt_weight * stats_h1.average_sample_size
        )
        # Change here for a different objective function.
        if verbose:
            print(
                "evaluations = ",
                func_evaluations,
                "alpha_spending_parameter = ",
                alpha_spending_parameter,
                "beta_spending_parameter = ",
                beta_spending_parameter,
                "cost = ",
                trial_cost,
            )
        return trial_cost

    bounds = [(-10, 4)]
    result = differential_evolution(objective_function, bounds=bounds, seed=seed)
    alpha_spending_parameter = result.x[0]
    beta_spending_parameter = find_beta_for_alpha(
        alpha_spending_parameter,
        samples_h0,
        samples_h1,
        looks_fractions,
        n_samples_per_arm_per_look,
        alpha,
        beta,
        is_binding,
    )
    print(
        "alpha_spending_parameter = ",
        alpha_spending_parameter,
        "beta_spending_parameter = ",
        beta_spending_parameter,
        "function value = ",
        result.fun,
        "success = ",
        result.success,
        "number of evaluations = ",
        result.nfev,
    )
    return (alpha_spending_parameter, beta_spending_parameter, result)
