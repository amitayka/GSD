import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from scipy.optimize import differential_evolution
from scipy.stats import norm

from gsd.basic_code.find_best_general_spending import find_best_general_spending
from gsd.basic_code.gsd_statistics_calculator import get_statistics_given_thresholds
from gsd.basic_code.gsd_threshold_finder_algorithm_1 import (
    get_efficacy_futility_thresholds,
)
from gsd.basic_code.utils.spending_function import (
    generate_spending_from_spending_parameter,
)

W1 = 0.4  # weight of ESS under null
W2 = 0.4  # weight of ESS under alternative
W3 = 0.2  # weight of maximal sample size
N_TRIALS = 1000000


@dataclass
class OptimizationResult:
    groupsize: int
    futility_thresholds: np.ndarray
    efficacy_thresholds: np.ndarray
    ess_null: float
    ess_alt: float
    type_I_error: float
    power: float


def get_normal_samples_statistics(
    groupsize, effect_size, n_looks, n_trials, rng: np.random.Generator
):
    samples_statistics = rng.normal(
        loc=effect_size * np.sqrt(groupsize / 2), scale=1.0, size=(n_trials, 1, n_looks)
    )
    samples_statistics = np.cumsum(samples_statistics, axis=2)
    for i in range(1, n_looks):
        samples_statistics[:, 0, i] /= np.sqrt(i + 1)
    return samples_statistics


def simulate_design_using_monte_carlo_with_spending_functions(
    effect_size,
    n_looks,
    groupsize,
    alpha,
    power,
    alpha_spending_parameter,
    beta_spending_parameter,
    n_trials,
    rng: np.random.Generator,
) -> OptimizationResult:
    samples_statistics_h0 = get_normal_samples_statistics(
        groupsize, 0, n_looks, n_trials, rng
    )
    samples_statistics_h1 = get_normal_samples_statistics(
        groupsize, effect_size, n_looks, n_trials, rng
    )
    looks_fractions = np.arange(1, n_looks + 1) / n_looks
    alpha_spending = generate_spending_from_spending_parameter(
        alpha_spending_parameter, alpha, looks_fractions
    )
    beta_spending = generate_spending_from_spending_parameter(
        beta_spending_parameter, 1 - power, looks_fractions
    )
    efficacy_thresholds, futility_thresholds = get_efficacy_futility_thresholds(
        samples_statistics_h0,
        samples_statistics_h1,
        alpha_spending,
        beta_spending,
        is_binding=True,
    )

    # we divide by 2 because we have 2 arms, and optgs uses sample size per arm
    n_samples_per_look_per_arm = groupsize * np.arange(1, n_looks + 1) / 2

    statistics_h0 = get_statistics_given_thresholds(
        samples_statistics_h0,
        efficacy_thresholds,
        futility_thresholds,
        n_samples_per_look_per_arm=n_samples_per_look_per_arm,
    )
    statistics_h1 = get_statistics_given_thresholds(
        samples_statistics_h1,
        efficacy_thresholds,
        futility_thresholds,
        n_samples_per_look_per_arm=n_samples_per_look_per_arm,
    )
    return OptimizationResult(
        groupsize=groupsize,
        futility_thresholds=futility_thresholds,
        efficacy_thresholds=efficacy_thresholds,
        ess_null=statistics_h0.average_sample_size,
        ess_alt=statistics_h1.average_sample_size,
        type_I_error=statistics_h0.disjunctive_power,
        power=statistics_h1.disjunctive_power,
    )

def simulate_design_using_monte_carlo_given_thresholds(
    groupsize,
    effect_size,
    n_looks,
    futility_thresholds,
    n_trials,
    efficacy_thresholds,
    rng: np.random.Generator,
) -> OptimizationResult:
    samples_statistics_h0 = get_normal_samples_statistics(
        groupsize, 0, n_looks, n_trials, rng
    )
    samples_statistics_h1 = get_normal_samples_statistics(
        groupsize, effect_size, n_looks, n_trials, rng
    )

    # we divide by 2 because we have 2 arms, and optgs uses sample size per arm
    n_samples_per_look_per_arm = groupsize * np.arange(1, n_looks + 1) / 2
    statistics_h0 = get_statistics_given_thresholds(
        samples_statistics_h0,
        efficacy_thresholds,
        futility_thresholds,
        n_samples_per_look_per_arm=n_samples_per_look_per_arm,
    )
    statistics_h1 = get_statistics_given_thresholds(
        samples_statistics_h1,
        efficacy_thresholds,
        futility_thresholds,
        n_samples_per_look_per_arm=n_samples_per_look_per_arm,
    )
    return OptimizationResult(
        groupsize=groupsize,
        futility_thresholds=futility_thresholds,
        efficacy_thresholds=efficacy_thresholds,
        ess_null=statistics_h0.average_sample_size,
        ess_alt=statistics_h1.average_sample_size,
        type_I_error=statistics_h0.disjunctive_power,
        power=statistics_h1.disjunctive_power,
    )



def optimize_design_using_hsd_monte_carlo(
    effect_size,
    n_looks,
    fixed_sample_size,
    alpha,
    power,
    n_trials,
    rng: np.random.Generator,
    groupsize: Optional[int] = None,
) -> tuple[OptimizationResult, float, float]:
    if groupsize is None:

        def objective_function(
            x: np.ndarray,
        ):
            groupsize = int(x[0])
            alpha_spending_parameter = x[1]
            beta_spending_parameter = x[2]
            result = simulate_design_using_monte_carlo_with_spending_functions(
                effect_size=effect_size,
                n_looks=n_looks,
                groupsize=groupsize,
                alpha=alpha,
                power=power,
                alpha_spending_parameter=alpha_spending_parameter,
                beta_spending_parameter=beta_spending_parameter,
                n_trials=n_trials,
                rng=rng,
            )
            if result.power < power:
                return 10000 * (2 - result.power)
            return W1 * result.ess_null + W2 * result.ess_alt + W3 * groupsize * n_looks

        minimal_group_size = int(fixed_sample_size / n_looks)
        maximal_group_size = int(
            fixed_sample_size / n_looks * 1.4
        )  # at most 40% increase in max sample size
        bounds = [
            (minimal_group_size, maximal_group_size),
            (-10, 4),
            (-10, 4),
        ]
        result = differential_evolution(objective_function, bounds=bounds, seed=42)
        groupsize = int(result.x[0])
        alpha_spending_parameter = result.x[1]
        beta_spending_parameter = result.x[2]
    else:
        samples_h0 = get_normal_samples_statistics(groupsize, 0, n_looks, n_trials, rng)
        samples_h1 = get_normal_samples_statistics(
            groupsize, effect_size, n_looks, n_trials, rng
        )
        find_best_spending_functions = find_best_general_spending(
            samples_h0=samples_h0,
            samples_h1=samples_h1,
            looks_fractions=np.arange(1, n_looks + 1) / n_looks,
            n_samples_per_arm_per_look=np.array(
                [i * groupsize for i in range(1, n_looks + 1)]
            ),
            alpha=alpha,
            beta=1 - power,
            is_binding=True,
        )
        alpha_spending_parameter = find_best_spending_functions[0]
        beta_spending_parameter = find_best_spending_functions[1]
    simulation_result = simulate_design_using_monte_carlo_with_spending_functions(
        effect_size=effect_size,
        n_looks=n_looks,
        groupsize=groupsize,
        alpha=alpha,
        power=power,
        alpha_spending_parameter=alpha_spending_parameter,
        beta_spending_parameter=beta_spending_parameter,
        n_trials=n_trials,
        rng=rng,
    )
    return (
        OptimizationResult(
            groupsize=groupsize,
            futility_thresholds=simulation_result.futility_thresholds,
            efficacy_thresholds=simulation_result.efficacy_thresholds,
            ess_null=simulation_result.ess_null,
            ess_alt=simulation_result.ess_alt,
            type_I_error=simulation_result.type_I_error,
            power=simulation_result.power,
        ),
        alpha_spending_parameter,
        beta_spending_parameter,
        result.nfev,
    )


def optimize_design_using_monte_carlo_general_spending(
    effect_size,
    n_looks,
    groupsize,
    alpha,
    power,
    n_trials,
    rng: np.random.Generator,
) -> tuple[OptimizationResult, int]:
    samples_statistics_h0 = get_normal_samples_statistics(
        groupsize, 0, n_looks, n_trials, rng
    )
    samples_statistics_h1 = get_normal_samples_statistics(
        groupsize, effect_size, n_looks, n_trials, rng
    )

    alpha_spending, beta_spending, differential_evolution_result = (
        find_best_general_spending(
            samples_h0=samples_statistics_h0,
            samples_h1=samples_statistics_h1,
            looks_fractions=np.arange(1, n_looks + 1) / n_looks,
            n_samples_per_arm_per_look=np.array(
                [i * groupsize for i in range(1, n_looks + 1)]
            ),
            alpha=alpha,
            beta=1 - power,
            is_binding=True,
        )
    )

    (
        efficacy_thresholds,
        futility_thresholds,
    ) = get_efficacy_futility_thresholds(
        samples_statistics_h0,
        samples_statistics_h1,
        alpha_spending,
        beta_spending,
        is_binding=True,
    )

    simulation_result = simulate_design_using_monte_carlo_given_thresholds(
        groupsize=groupsize,
        effect_size=effect_size,
        n_looks=n_looks,
        futility_thresholds=futility_thresholds,
        n_trials=n_trials,
        efficacy_thresholds=efficacy_thresholds,
        rng=rng,
    )
    return (
        OptimizationResult(
            groupsize=groupsize,
            futility_thresholds=simulation_result.futility_thresholds,
            efficacy_thresholds=simulation_result.efficacy_thresholds,
            ess_null=simulation_result.ess_null,
            ess_alt=simulation_result.ess_alt,
            type_I_error=simulation_result.type_I_error,
            power=simulation_result.power,
        ),
        differential_evolution_result.nfev,
    )




def optimize_design_using_optgs(
    alpha, power, n_looks, effect_size, weights
) -> OptimizationResult:

    # Import required R packages
    importr("OptGS")
    # Can try to install if it is not install in the system
    # utils = importr('utils')
    # utils.install_packages('OptGS')

    # Create R vector for weights
    if weights is None:
        weights = [W1, W2, 0, W3]

    # Set up the R code to execute
    r_code = f"""
    library(OptGS)
    res = optgs(alpha = {alpha}, power = {power}, weights = c({','.join(map(str, weights))}), J = {n_looks}, delta1 = {effect_size})
    result_list = list(
        groupsize = res$groupsize,
        futility = res$futility,
        efficacy = res$efficacy,
        ess_null = res$ess[1],
        ess_alt = res$ess[2],
        ess_max = res$ess[3],
        actual_alpha = res$typeIerror,
        actual_power = res$power
    )
    result_list
    """

    # Execute the R code
    results = robjects.r(r_code)

    # Convert R results to Python
    groupsize = int(results.rx2("groupsize")[0])
    futility = np.array(results.rx2("futility"))
    efficacy = np.array(results.rx2("efficacy"))
    ess_null = float(results.rx2("ess_null")[0])
    ess_alt = float(results.rx2("ess_alt")[0])
    ess_max = float(results.rx2("ess_max")[0])
    actual_alpha = float(results.rx2("actual_alpha")[0])
    actual_power = float(results.rx2("actual_power")[0])

    # Return as a Python dictionary
    return OptimizationResult(
        groupsize=groupsize,
        futility_thresholds=futility,
        efficacy_thresholds=efficacy,
        ess_null=ess_null,
        ess_alt=ess_alt,
        type_I_error=actual_alpha,
        power=actual_power,
    )


def simulate_design_using_gsdesign_given_spending_functions(
    alpha,
    power,
    alpha_spending_parameter,
    beta_spending_parameter,
    n_looks,
    effect_size,
) -> OptimizationResult:
    importr("gsDesign")
    # Can try to install if it is not install in the system
    # utils = importr('utils')
    # utils.install_packages('gsDesign')

    # Set up R code to perform verification
    r_code = f"""
    library(gsDesign)
    
    # Create cumulative sample sizes
    
    # Create a gsDesign object
    design <- gsDesign(
      k = {n_looks},
      test.type = 3,  # Two-sided test
      sfu = sfHSD,   # Hwang-Shih-DeCani spending function
      sfupar = {alpha_spending_parameter},    # Parameter for upper spending
      sfl = sfHSD,    # Hwang-Shih-DeCani spending function
      sflpar = {beta_spending_parameter},   # Parameter for lower spending
      delta = {effect_size/np.sqrt(2)},  # Effect size
      alpha = {alpha},
      beta= {1 - power},
    )

    # We want integer stopping times, to we take the ceiling of the sample size

    groupsize = ceiling(design$n.I[1])
    n.I = groupsize * (1:{n_looks})
    design =gsDesign(
      k = {n_looks},
      test.type = 3,  # Two-sided test
      sfu = sfHSD,   # Hwang-Shih-DeCani spending function
      sfupar = {alpha_spending_parameter},    # Parameter for upper spending
      sfl = sfHSD,    # Hwang-Shih-DeCani spending function
      sflpar = {beta_spending_parameter},   # Parameter for lower spending
      delta = {effect_size/np.sqrt(2)},  # Effect size
      alpha = {alpha},
      beta= {1 - power},
      n.I = n.I
    )

    
    # Calculate type I error
    alpha <- sum(gsProbability(d = design, theta = 0)$upper$prob)
    
    # Calculate power (assuming a standardized effect size of effect_size
    effect_size <- {effect_size/np.sqrt(2)}
    power <- sum(gsProbability(d = design, theta = effect_size)$upper$prob)
    
    # Calculate expected sample sizes
    # Under null hypothesis (theta = 0)
    en_null <- gsProbability(d = design, theta = 0)$en
    
    # Under alternative hypothesis (theta = effect_size)
    en_alt <- gsProbability(d = design, theta = effect_size)$en
    
    # Early stopping probabilities
    stop_prob_null <- gsProbability(d = design, theta = 0)$upper$prob
    stop_prob_alt <- gsProbability(d = design, theta = effect_size)$upper$prob
    
    # Return results as a list
    list(
      groupsize = design$n.I,
      type_I_error = alpha,
      power = power,
      expected_sample_null = en_null,
      expected_sample_alt = en_alt,
      stopping_probabilities_null = stop_prob_null,
      stopping_probabilities_alt = stop_prob_alt,
      futility = design$lower$bound,
      efficacy = design$upper$bound)
    """

    # Execute the R code
    results = robjects.r(r_code)

    # Convert R results to Python
    return OptimizationResult(
        groupsize=int(results.rx2("groupsize")[0]),
        futility_thresholds=np.array(results.rx2("futility")),
        efficacy_thresholds=np.array(results.rx2("efficacy")),
        ess_null=float(results.rx2("expected_sample_null")[0]),
        ess_alt=float(results.rx2("expected_sample_alt")[0]),
        type_I_error=float(results.rx2("type_I_error")[0]),
        power=float(results.rx2("power")[0]),
    )


def optimize_design_using_gsdesign(
    alpha,
    power,
    n_looks,
    effect_size,
) -> tuple[OptimizationResult, float, float, int]:
    def objective_function(
        x: np.ndarray,
    ):
        alpha_spending_parameter = x[0]
        beta_spending_parameter = x[1]
        result = simulate_design_using_gsdesign_given_spending_functions(
            alpha=alpha,
            power=power,
            alpha_spending_parameter=alpha_spending_parameter,
            beta_spending_parameter=beta_spending_parameter,
            n_looks=n_looks,
            effect_size=effect_size,
        )

        return (
            W1 * result.ess_null + W2 * result.ess_alt + W3 * result.groupsize * n_looks
        )

    bounds = [
        (-10, 4),
        (-10, 4),
    ]

    result = differential_evolution(objective_function, bounds=bounds, seed=42)

    alpha_spending_parameter = result.x[0]
    beta_spending_parameter = result.x[1]

    simulation_result = simulate_design_using_gsdesign_given_spending_functions(
        alpha=alpha,
        power=power,
        alpha_spending_parameter=alpha_spending_parameter,
        beta_spending_parameter=beta_spending_parameter,
        n_looks=n_looks,
        effect_size=effect_size,
    )
    return (
        OptimizationResult(
            groupsize=simulation_result.groupsize,
            futility_thresholds=simulation_result.futility_thresholds,
            efficacy_thresholds=simulation_result.efficacy_thresholds,
            ess_null=simulation_result.ess_null,
            ess_alt=simulation_result.ess_alt,
            type_I_error=simulation_result.type_I_error,
            power=simulation_result.power,
        ),
        alpha_spending_parameter,
        beta_spending_parameter,
        result.nfev,
    )


def simulate_design_using_gsdesign_given_thresholds(
    groupsize, futility_thresholds, efficacy_thresholds, J, effect_size
) -> OptimizationResult:
    """
    Verify the characteristics of a group sequential design given its parameters.

    Args:
        groupsize: Size of each group
        futility_thresholds: Array of futility boundaries
        efficacy_thresholds: Array of efficacy boundaries
        J: Number of stages

    Returns:
        dict: Dictionary containing type I error, power, and expected sample sizes
    """
    # Import required R packages
    importr("gsDesign")
    # Can try to install if it is not install in the system
    # utils = importr('utils')
    # utils.install_packages('gsDesign')

    # Set up R code to perform verification
    r_code = f"""
    library(gsDesign)
    
    # Create cumulative sample sizes
    cum_n <- {groupsize} * (1:{J})
    
    # Create a gsDesign object
    design <- gsDesign(
      k = {J},
      test.type = 2,  # Two-sided test
      n.I = cum_n,    # Information time (proportional to sample size)
      maxn.IPlan = cum_n[{J}]  # Maximum planned sample size
    )
    
    # Override the boundaries with specified values
    design$upper$bound <- c({','.join(map(str, efficacy_thresholds))})
    design$lower$bound <- c({','.join(map(str, futility_thresholds))})
    
    # Calculate type I error
    alpha <- sum(gsProbability(d = design, theta = 0)$upper$prob)
    
    # Calculate power (assuming a standardized effect size of effect_size
    effect_size <- {effect_size/np.sqrt(2)}
    power <- sum(gsProbability(d = design, theta = effect_size)$upper$prob)
    
    # Calculate expected sample sizes
    # Under null hypothesis (theta = 0)
    en_null <- gsProbability(d = design, theta = 0)$en
    
    # Under alternative hypothesis (theta = effect_size)
    en_alt <- gsProbability(d = design, theta = effect_size)$en
    
    # Early stopping probabilities
    stop_prob_null <- gsProbability(d = design, theta = 0)$upper$prob
    stop_prob_alt <- gsProbability(d = design, theta = effect_size)$upper$prob
    
    # Return results as a list
    list(
      type_I_error = alpha,
      power = power,
      expected_sample_null = en_null,
      expected_sample_alt = en_alt,
      stopping_probabilities_null = stop_prob_null,
      stopping_probabilities_alt = stop_prob_alt
    )
    """

    # Execute the R code
    results = robjects.r(r_code)

    # Convert R results to Python
    return OptimizationResult(
        groupsize=groupsize,
        futility_thresholds=futility_thresholds,
        efficacy_thresholds=efficacy_thresholds,
        ess_null=float(results.rx2("expected_sample_null")[0]),
        ess_alt=float(results.rx2("expected_sample_alt")[0]),
        type_I_error=float(results.rx2("type_I_error")[0]),
        power=float(results.rx2("power")[0]),
    )


if __name__ == "__main__":
    # Run with default values
    rng = np.random.default_rng(seed=42)
    effect_size = 0.3
    alpha = 0.025
    power = 0.8
    n_looks = 4
    weights = [W1, W2, 0, W3]
    # Calculate the fixed sample size for a normal one-arm trial
    z_alpha = norm.ppf(1 - alpha)
    z_beta = norm.ppf(power)
    fixed_sample_size = 2 * ((z_alpha + z_beta) / effect_size) ** 2
    print(f"fixed sample size per arm: {np.ceil(fixed_sample_size).astype(int)}")
    start = time.time()
    results_optgs = optimize_design_using_optgs(
        alpha=alpha,
        power=power,
        weights=weights,
        n_looks=n_looks,
        effect_size=effect_size,
    )
    print("\nOptGS optimization took: ", time.time() - start)
    print("OptGS results:")
    print(f"Group size: {results_optgs.groupsize}")
    print(f"Futility boundaries: {results_optgs.futility_thresholds}")
    print(f"Efficacy boundaries: {results_optgs.efficacy_thresholds}")
    print(f"ESS at null: {results_optgs.ess_null}")
    print(f"ESS at ALT: {results_optgs.ess_alt}")
    print(
        f"Cost: {W1 * results_optgs.ess_null + W2 * results_optgs.ess_alt+ W3 * results_optgs.groupsize * n_looks}"
    )

    start = time.time()

    results_gsdesign, alpha_spending_parameter, beta_spending_parameter, nfev = (
        optimize_design_using_gsdesign(
            alpha=alpha,
            power=power,
            n_looks=n_looks,
            effect_size=effect_size,
        )
    )
    print(
        "\ngsDesign optimization took: ",
        time.time() - start,
        "n. of evaluations: ",
        nfev,
    )
    print("gsDesign optimization results:")
    print(f"alpha spending parameter: {alpha_spending_parameter}")
    print(f"beta spending parameter: {beta_spending_parameter}")
    print(f"Group size: {results_gsdesign.groupsize}")
    print(f"Futility boundaries: {results_gsdesign.futility_thresholds}")
    print(f"Efficacy boundaries: {results_gsdesign.efficacy_thresholds}")
    print(f"ESS at null: {results_gsdesign.ess_null}")
    print(f"ESS at ALT: {results_gsdesign.ess_alt}")
    print(f"Type I error: {results_gsdesign.type_I_error}")
    print(f"Power: {results_gsdesign.power}")
    print(
        f"Cost: {W1 * results_gsdesign.ess_null + W2 * results_gsdesign.ess_alt+ W3 * results_gsdesign.groupsize * n_looks}"
    )
    for groupsize in [None, results_optgs.groupsize]:

        start = time.time()
        results_monte_carlo, alpha_spending_parameter, beta_spending_parameter, nfev = (
            optimize_design_using_hsd_monte_carlo(
                effect_size=effect_size,
                n_looks=n_looks,
                fixed_sample_size=fixed_sample_size,
                alpha=alpha,
                power=power,
                n_trials=N_TRIALS,
                rng=rng,
            )
        )

        print(
            "\nMonte Carlo HSD optimization took: ",
            time.time() - start,
            "n. of evaluations: ",
            nfev,
        )
        if groupsize is None:
            print("Finding also group size")
        else:
            print(f"Using group size of OptGS: {groupsize}")
        print("Monte Carlo results:")
        print(f"alpha spending parameter: {alpha_spending_parameter}")
        print(f"beta spending parameter: {beta_spending_parameter}")
        print(f"Group size: {results_monte_carlo.groupsize}")
        print(f"Futility boundaries: {results_monte_carlo.futility_thresholds}")
        print(f"Efficacy boundaries: {results_monte_carlo.efficacy_thresholds}")
        print(f"ESS at null: {results_monte_carlo.ess_null}")
        print(f"ESS at ALT: {results_monte_carlo.ess_alt}")
        print(f"Type I error: {results_monte_carlo.type_I_error}")
        print(f"Power: {results_monte_carlo.power}")
        print(
            f"Cost: {W1 * results_monte_carlo.ess_null + W2 * results_monte_carlo.ess_alt+ W3 * results_monte_carlo.groupsize * n_looks}"
        )

        verify_results_monte_carlo = simulate_design_using_gsdesign_given_thresholds(
            groupsize=results_monte_carlo.groupsize,
            futility_thresholds=results_monte_carlo.futility_thresholds,
            efficacy_thresholds=results_monte_carlo.efficacy_thresholds,
            J=n_looks,
            effect_size=effect_size,
        )

        print("Monte Carlo verification results with gsDesign:")
        print(f"Group size: {verify_results_monte_carlo.groupsize}")
        print(f"Futility boundaries: {verify_results_monte_carlo.futility_thresholds}")
        print(f"Efficacy boundaries: {verify_results_monte_carlo.efficacy_thresholds}")
        print(f"ESS at null: {verify_results_monte_carlo.ess_null}")
        print(f"ESS at ALT: {verify_results_monte_carlo.ess_alt}")
        print(f"Type I error: {verify_results_monte_carlo.type_I_error}")
        print(f"Power: {verify_results_monte_carlo.power}")
        print(
            f"Cost: {W1 * verify_results_monte_carlo.ess_null + W2 * verify_results_monte_carlo.ess_alt+ W3 * verify_results_monte_carlo.groupsize * n_looks}"
        )

    start = time.time()
    results_best_spending, nfev = (
        optimize_design_using_monte_carlo_general_spending(
            effect_size=effect_size,
            n_looks=n_looks,
            groupsize=results_optgs.groupsize,
            alpha=alpha,
            power=power,
            n_trials=N_TRIALS,
            rng=rng,
        )
    )
    print(
        "\nBest spending optimization took: ",
        time.time() - start,
        "n. of evaluations: ",
        nfev,
    )
    print("Best spending results:")
    print(f"Group size: {results_best_spending.groupsize}")
    print(f"Futility boundaries: {results_best_spending.futility_thresholds}")
    print(f"Efficacy boundaries: {results_best_spending.efficacy_thresholds}")
    print(f"ESS at null: {results_best_spending.ess_null}")
    print(f"ESS at ALT: {results_best_spending.ess_alt}")
    print(f"Type I error: {results_best_spending.type_I_error}")
    print(f"Power: {results_best_spending.power}")
    print(
        f"Cost: {W1 * results_best_spending.ess_null + W2 * results_best_spending.ess_alt+ W3 * results_best_spending.groupsize * n_looks}"
    )


# fixed sample size per arm: 175

# OptGS optimization took:  0.14446473121643066
# OptGS results:
# Group size: 49
# Futility boundaries: [-0.34805502  0.70940276  1.38883689  1.91486618]
# Efficacy boundaries: [4.42053062 2.90941997 2.27791469 1.91486618]
# ESS at null: 94.11834748125302
# ESS at ALT: 141.72394671436507
# Cost: 133.53691767824722

# gsDesign optimization took:  29.126142978668213 n. of evaluations:  285
# gsDesign optimization results:
# alpha spending parameter: -2.0221444845306076
# beta spending parameter: -0.2966435448461313
# Group size: 50
# Futility boundaries: [-0.19981625  0.65034805  1.34209094  1.98776843]
# Efficacy boundaries: [2.8058448  2.58209213 2.3313404  1.98776843]
# ESS at null: 93.53998937568012
# ESS at ALT: 133.4922508383499
# Type I error: 0.025000000141432718
# Power: 0.8010093065976658
# Cost: 130.812896085612

# Monte Carlo HSD optimization took:  381.0210428237915 n. of evaluations:  1650
# Finding also group size
# Monte Carlo results:
# alpha spending parameter: -2.888058464835864
# beta spending parameter: -0.19593417549247416
# Group size: 49
# Futility boundaries: [-0.19901328  0.63654262  1.31576431  1.95787731]
# Efficacy boundaries: [2.96279156 2.68261665 2.37949357 1.95787731]
# ESS at null: 92.31800900000002
# ESS at ALT: 135.654099
# Type I error: 0.025
# Power: 0.797253
# Cost: 130.3888432
# Monte Carlo verification results with gsDesign:
# Group size: 49
# Futility boundaries: [-0.19901328  0.63654262  1.31576431  1.95787731]
# Efficacy boundaries: [2.96279156 2.68261665 2.37949357 1.95787731]
# ESS at null: 92.24026945607892
# ESS at ALT: 135.6575622769861
# Type I error: 0.02452372574543464
# Power: 0.797467489161775
# Cost: 130.359132693226
# ^[[D^[[D
# Monte Carlo HSD optimization took:  265.479603767395 n. of evaluations:  1143
# Using group size of OptGS: 49
# Monte Carlo results:
# alpha spending parameter: -2.898615157599881
# beta spending parameter: -0.22302115915016651
# Group size: 49
# Futility boundaries: [-0.19976592  0.64033411  1.32227447  1.94459417]
# Efficacy boundaries: [2.95836672 2.67493669 2.36250146 1.94459417]
# ESS at null: 92.14283400000002
# ESS at ALT: 135.197664
# Type I error: 0.025
# Power: 0.800029
# Cost: 130.13619920000002
# Monte Carlo verification results with gsDesign:
# Group size: 49
# Futility boundaries: [-0.19976592  0.64033411  1.32227447  1.94459417]
# Efficacy boundaries: [2.95836672 2.67493669 2.36250146 1.94459417]
# ESS at null: 92.13982842978288
# ESS at ALT: 135.14387850228007
# Type I error: 0.025115986856866546
# Power: 0.7990571412406631
# Cost: 130.1134827728252

# Best spending optimization took:  262.9367983341217 n. of evaluations:  1987
# Best spending results:
# Group size: 49
# Futility boundaries: [-0.03587141  0.50815958  1.13220633  1.96505129]
# Efficacy boundaries: [3.08167034 2.56185083 2.43906672 1.96505129]
# ESS at null: 91.282688
# ESS at ALT: 135.441439
# Type I error: 0.024569
# Power: 0.799547
# Cost: 129.88965080000003
