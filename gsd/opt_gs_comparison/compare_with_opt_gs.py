import pickle
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from scipy.optimize import differential_evolution
from scipy.stats import norm

from gsd.basic_code.find_best_general_spending import find_best_general_spending
from gsd.basic_code.find_good_hsd_spending_functions import find_best_hsd_spending
from gsd.basic_code.gsd_statistics_calculator import get_statistics_given_thresholds
from gsd.basic_code.gsd_threshold_finder_algorithm_1 import (
    get_efficacy_futility_thresholds,
)
from gsd.basic_code.utils.spending_function import (
    generate_spending_from_spending_parameter,
)

default_weights = np.array([0.4, 0.4, 0, 0.2])  # W1, W2, 0, W3
N_TRIALS1 = 100000
N_TRIALS2 = 1000000
REPEATS1 = 30
REPEATS2 = 10


@dataclass
class InputParameters:
    alpha: float
    power: float
    n_looks: int
    effect_size: float
    weights: np.ndarray
    fixed_sample_size: int


class AlgorithmName(Enum):
    OPTGS = "optgs"
    GSDESIGN = "gsdesign"
    HSD_MONTE_CARLO = "hsd_monte_carlo"
    GENERAL_SPENDING_MONTE_CARLO = "general_spending_monte_carlo"


@dataclass
class OptimizationConfiguration:
    name: str
    algorithm: AlgorithmName
    n_trials: Optional[int]
    seed: Optional[int]
    groupsize: Optional[int]


@dataclass
class OptimizationResult:
    alpha_spending_parameter: Optional[float]
    beta_spending_parameter: Optional[float]
    nfev: Optional[int]
    groupsize: int
    futility_thresholds: np.ndarray
    efficacy_thresholds: np.ndarray
    ess_null: float
    ess_alt: float
    cost: float
    type_I_error: float
    power: float
    time: Optional[float] = None  # Time taken for the optimization run


@dataclass
class OptimizationRun:
    config: OptimizationConfiguration
    input_parameters: InputParameters
    optimization_result: OptimizationResult


def get_cost(
    input_parameters: InputParameters, ess_null: float, ess_alt: float, groupsize: int
) -> float:
    """
    Calculate the cost of the optimization run based on the input parameters and results.
    """
    return (
        input_parameters.weights[0] * ess_null
        + input_parameters.weights[1] * ess_alt
        + input_parameters.weights[3] * groupsize * input_parameters.n_looks
    )


def save_optimization_runs(runs: List[OptimizationRun], filename: str):
    """Serialize and save a list of OptimizationRun objects to a file."""
    with open(filename, "wb") as f:
        pickle.dump(runs, f)


def load_optimization_runs(filename: str) -> List[OptimizationRun]:
    """Load and deserialize a list of OptimizationRun objects from a file."""
    with open(filename, "rb") as f:
        return pickle.load(f)


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
    input_parameters: InputParameters,
    groupsize: int,
    alpha_spending_parameter: float,
    beta_spending_parameter: float,
    n_trials: int,
    rng: np.random.Generator,
) -> OptimizationResult:
    effect_size = input_parameters.effect_size
    n_looks = input_parameters.n_looks

    samples_statistics_h0 = get_normal_samples_statistics(
        groupsize, 0, n_looks, n_trials, rng
    )
    samples_statistics_h1 = get_normal_samples_statistics(
        groupsize, effect_size, n_looks, n_trials, rng
    )
    looks_fractions = np.arange(1, n_looks + 1) / n_looks
    alpha_spending = generate_spending_from_spending_parameter(
        alpha_spending_parameter, input_parameters.alpha, looks_fractions
    )
    beta_spending = generate_spending_from_spending_parameter(
        beta_spending_parameter, 1 - input_parameters.power, looks_fractions
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
        alpha_spending_parameter=alpha_spending_parameter,
        beta_spending_parameter=beta_spending_parameter,
        nfev=None,  # Not applicable for this simulation
        cost=get_cost(
            input_parameters=input_parameters,
            ess_null=statistics_h0.average_sample_size,
            ess_alt=statistics_h1.average_sample_size,
            groupsize=groupsize,
        ),
        groupsize=groupsize,
        futility_thresholds=futility_thresholds,
        efficacy_thresholds=efficacy_thresholds,
        ess_null=statistics_h0.average_sample_size,
        ess_alt=statistics_h1.average_sample_size,
        type_I_error=statistics_h0.disjunctive_power,
        power=statistics_h1.disjunctive_power,
    )


def optimize_design_using_hsd_monte_carlo(
    input_parameters: InputParameters,
    optimization_config: OptimizationConfiguration,
) -> OptimizationResult:
    alpha = input_parameters.alpha
    power = input_parameters.power
    n_looks = input_parameters.n_looks
    effect_size = input_parameters.effect_size
    fixed_sample_size = input_parameters.fixed_sample_size
    groupsize = optimization_config.groupsize
    n_trials = optimization_config.n_trials
    if groupsize is None:

        def objective_function(
            x: np.ndarray,
        ):
            rng = np.random.default_rng(seed=optimization_config.seed)
            groupsize = int(x[0])
            alpha_spending_parameter = x[1]
            beta_spending_parameter = x[2]
            result = simulate_design_using_monte_carlo_with_spending_functions(
                input_parameters=input_parameters,
                groupsize=groupsize,
                alpha_spending_parameter=alpha_spending_parameter,
                beta_spending_parameter=beta_spending_parameter,
                n_trials=n_trials,
                rng=rng,
            )
            if result.power < power:
                return 10000 * (2 - result.power)
            return (
                input_parameters.weights[0] * result.ess_null
                + input_parameters.weights[1] * result.ess_alt
                + input_parameters.weights[3] * groupsize * n_looks
            )

        minimal_group_size = int(fixed_sample_size / n_looks)
        maximal_group_size = int(
            fixed_sample_size / n_looks * 1.4
        )  # at most 40% increase in max sample size
        bounds = [
            (minimal_group_size, maximal_group_size),
            (-10, 4),
            (-10, 4),
        ]
        result = differential_evolution(
            objective_function, bounds=bounds, seed=optimization_config.seed
        )
        groupsize = int(result.x[0])
        alpha_spending_parameter = result.x[1]
        beta_spending_parameter = result.x[2]
        nfev = result.nfev
    else:
        rng = np.random.default_rng(seed=optimization_config.seed)
        samples_h0 = get_normal_samples_statistics(groupsize, 0, n_looks, n_trials, rng)
        samples_h1 = get_normal_samples_statistics(
            groupsize, effect_size, n_looks, n_trials, rng
        )
        find_best_spending_functions = find_best_hsd_spending(
            samples_h0=samples_h0,
            samples_h1=samples_h1,
            looks_fractions=np.arange(1, n_looks + 1) / n_looks,
            n_samples_per_arm_per_look=np.array(
                [i * groupsize for i in range(1, n_looks + 1)]
            ),
            alpha=alpha,
            beta=1 - power,
            is_binding=True,
            seed=optimization_config.seed,
            null_weight=input_parameters.weights[0],
            alt_weight=input_parameters.weights[1],
        )
        alpha_spending_parameter = find_best_spending_functions[0]
        beta_spending_parameter = find_best_spending_functions[1]
        nfev = find_best_spending_functions[2].nfev
    rng = np.random.default_rng(seed=optimization_config.seed)
    result = simulate_design_using_monte_carlo_with_spending_functions(
        input_parameters=input_parameters,
        groupsize=groupsize,
        alpha_spending_parameter=alpha_spending_parameter,
        beta_spending_parameter=beta_spending_parameter,
        n_trials=n_trials,
        rng=rng,
    )
    efficacy_thresholds = result.efficacy_thresholds
    futility_thresholds = result.futility_thresholds
    simulation_result_gsdesign = simulate_design_using_gsdesign_given_thresholds(
        input_parameters=input_parameters,
        futility_thresholds=futility_thresholds,
        efficacy_thresholds=efficacy_thresholds,
        groupsize=groupsize,
    )
    simulation_result_gsdesign.nfev = nfev
    simulation_result_gsdesign.alpha_spending_parameter = alpha_spending_parameter
    simulation_result_gsdesign.beta_spending_parameter = beta_spending_parameter

    return simulation_result_gsdesign


def optimize_design_using_monte_carlo_general_spending(
    input_parameters: InputParameters,
    optimization_config: OptimizationConfiguration,
) -> tuple[OptimizationResult, int]:
    alpha = input_parameters.alpha
    power = input_parameters.power
    n_looks = input_parameters.n_looks
    effect_size = input_parameters.effect_size
    groupsize = optimization_config.groupsize
    n_trials = optimization_config.n_trials
    rng = np.random.default_rng(seed=optimization_config.seed)
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
            null_weight=input_parameters.weights[0],
            alt_weight=input_parameters.weights[1],
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

    simulation_result = simulate_design_using_gsdesign_given_thresholds(
        input_parameters=input_parameters,
        groupsize=groupsize,
        futility_thresholds=futility_thresholds,
        efficacy_thresholds=efficacy_thresholds,
    )
    simulation_result.nfev = differential_evolution_result.nfev
    return simulation_result


def optimize_design_using_optgs(
    input_parameters: InputParameters,
) -> OptimizationResult:

    # Import required R packages
    importr("OptGS")
    # Can try to install if it is not install in the system
    # utils = importr('utils')
    # utils.install_packages('OptGS')

    # Create R vector for weights

    # Set up the R code to execute
    r_code = f"""
    library(OptGS)
    res = optgs(alpha = {input_parameters.alpha}, power = {input_parameters.power}, weights = c({','.join(map(str, input_parameters.weights))}), J = {input_parameters.n_looks}, delta1 = {input_parameters.effect_size})
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
        alpha_spending_parameter=None,  # Not applicable for OptGS
        beta_spending_parameter=None,  # Not applicable for OptGS
        nfev=None,  # Not applicable for OptGS
        groupsize=groupsize,
        futility_thresholds=futility,
        efficacy_thresholds=efficacy,
        ess_null=ess_null,
        ess_alt=ess_alt,
        cost=get_cost(input_parameters, ess_null, ess_alt, groupsize),
        type_I_error=actual_alpha,
        power=actual_power,
    )


def simulate_design_using_gsdesign_given_spending_functions(
    input_parameters: InputParameters,
    alpha_spending_parameter,
    beta_spending_parameter,
) -> OptimizationResult:
    importr("gsDesign")
    # Can try to install if it is not install in the system
    # utils = importr('utils')
    # utils.install_packages('gsDesign')

    # Set up R code to perform verification
    alpha = input_parameters.alpha
    power = input_parameters.power
    n_looks = input_parameters.n_looks
    effect_size = input_parameters.effect_size
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
    groupsize = int(results.rx2("groupsize")[0])
    futility = np.array(results.rx2("futility"))
    efficacy = np.array(results.rx2("efficacy"))
    ess_null = float(results.rx2("expected_sample_null")[0])
    ess_alt = float(results.rx2("expected_sample_alt")[0])
    type_I_error = float(results.rx2("type_I_error")[0])
    power = float(results.rx2("power")[0])

    # Convert R results to Python
    return OptimizationResult(
        alpha_spending_parameter=alpha_spending_parameter,
        beta_spending_parameter=beta_spending_parameter,
        nfev=None,  # Not applicable for this simulation
        groupsize=groupsize,
        futility_thresholds=futility,
        efficacy_thresholds=efficacy,
        ess_null=ess_null,
        ess_alt=ess_alt,
        cost=get_cost(input_parameters, ess_null, ess_alt, groupsize),
        type_I_error=type_I_error,
        power=power,
    )


def optimize_design_using_gsdesign(
    input_parameters: InputParameters,
    optimization_config: OptimizationConfiguration,
) -> tuple[OptimizationResult, float, float, int]:

    def objective_function(
        x: np.ndarray,
    ):
        alpha_spending_parameter = x[0]
        beta_spending_parameter = x[1]
        result = simulate_design_using_gsdesign_given_spending_functions(
            input_parameters=input_parameters,
            alpha_spending_parameter=alpha_spending_parameter,
            beta_spending_parameter=beta_spending_parameter,
        )

        return result.cost

    # gsDesign has issues when alpha_spending_parameter is too large, especially for larger n_looks.
    # So we give a tighter bound
    bounds = [
        (
            -10,
            1.5,
        ),
        (-10, 1.5),
    ]

    result = differential_evolution(
        objective_function, bounds=bounds, seed=optimization_config.seed
    )

    alpha_spending_parameter = result.x[0]
    beta_spending_parameter = result.x[1]

    simulation_result = simulate_design_using_gsdesign_given_spending_functions(
        input_parameters=input_parameters,
        alpha_spending_parameter=alpha_spending_parameter,
        beta_spending_parameter=beta_spending_parameter,
    )
    simulation_result.alpha_spending_parameter = alpha_spending_parameter
    simulation_result.beta_spending_parameter = beta_spending_parameter
    simulation_result.nfev = result.nfev
    return simulation_result


def simulate_design_using_gsdesign_given_thresholds(
    input_parameters: InputParameters,
    groupsize,
    futility_thresholds,
    efficacy_thresholds,
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
    effect_size = input_parameters.effect_size
    n_looks = input_parameters.n_looks
    # Set up R code to perform verification
    r_code = f"""
    library(gsDesign)
    
    # Create cumulative sample sizes
    cum_n <- {groupsize} * (1:{n_looks})
    
    # Create a gsDesign object
    design <- gsDesign(
      k = {n_looks},
      test.type = 2,  # Two-sided test
      n.I = cum_n,    # Information time (proportional to sample size)
      maxn.IPlan = cum_n[{n_looks}]  # Maximum planned sample size
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
    ess_null = float(results.rx2("expected_sample_null")[0])
    ess_alt = float(results.rx2("expected_sample_alt")[0])
    type_I_error = float(results.rx2("type_I_error")[0])
    power = float(results.rx2("power")[0])

    # Convert R results to Python
    return OptimizationResult(
        alpha_spending_parameter=None,  # Not applicable for this simulation
        beta_spending_parameter=None,  # Not applicable for this simulation
        nfev=None,  # Not applicable for this simulation
        groupsize=groupsize,
        futility_thresholds=futility_thresholds,
        efficacy_thresholds=efficacy_thresholds,
        ess_null=ess_null,
        ess_alt=ess_alt,
        cost=get_cost(
            input_parameters=input_parameters,
            ess_null=ess_null,
            ess_alt=ess_alt,
            groupsize=groupsize,
        ),
        type_I_error=type_I_error,
        power=power,
    )


def run_optimization(
    input_parameters: InputParameters,
    optimization_config: OptimizationConfiguration,
) -> OptimizationResult:
    start = time.time()
    if optimization_config.algorithm == AlgorithmName.OPTGS:
        result = optimize_design_using_optgs(input_parameters=input_parameters)
    elif optimization_config.algorithm == AlgorithmName.GSDESIGN:
        result = optimize_design_using_gsdesign(
            input_parameters=input_parameters, optimization_config=optimization_config
        )
    elif optimization_config.algorithm == AlgorithmName.HSD_MONTE_CARLO:
        result = optimize_design_using_hsd_monte_carlo(
            input_parameters=input_parameters, optimization_config=optimization_config
        )
    elif optimization_config.algorithm == AlgorithmName.GENERAL_SPENDING_MONTE_CARLO:
        result = optimize_design_using_monte_carlo_general_spending(
            input_parameters=input_parameters,
            optimization_config=optimization_config,
        )
    else:
        raise ValueError(
            f"Unknown optimization algorithm: {optimization_config.algorithm}"
        )
    result.time = time.time() - start
    return result


def run_scenario(input_parameters, output_filename: str, verbose: bool = False):
    """
    Run all optimization scenarios and save the results to a file.
    """
    z_alpha = norm.ppf(1 - input_parameters.alpha)
    z_beta = norm.ppf(input_parameters.power)
    fixed_sample_size = 2 * ((z_alpha + z_beta) / input_parameters.effect_size) ** 2
    input_parameters.fixed_sample_size = int(np.ceil(fixed_sample_size))
    fixed_multiplier_groupsize = int(fixed_sample_size / input_parameters.n_looks * 1.1)
    optgs_result = optimize_design_using_optgs(input_parameters=input_parameters)
    optgs_groupsize = optgs_result.groupsize

    optimization_configs = (
        [
            OptimizationConfiguration(
                name="0 optgs",
                algorithm=AlgorithmName.OPTGS,
                n_trials=None,
                seed=None,
                groupsize=None,
            )
        ]
        + [
            OptimizationConfiguration(
                name=f"1 hsd_monte_carlo_with_fixed_multiplier_groupsize_n_trials_{N_TRIALS1}",
                algorithm=AlgorithmName.HSD_MONTE_CARLO,
                n_trials=N_TRIALS1,
                seed=seed,
                groupsize=fixed_multiplier_groupsize,
            )
            for seed in range(REPEATS1)
        ]
        + [
            OptimizationConfiguration(
                name=f"2 hsd_monte_carlo_with_fixed_multiplier_groupsize_n_trials_{N_TRIALS2}",
                algorithm=AlgorithmName.HSD_MONTE_CARLO,
                n_trials=N_TRIALS2,
                seed=seed,
                groupsize=fixed_multiplier_groupsize,
            )
            for seed in range(REPEATS2)
        ]
        + [
            OptimizationConfiguration(
                name=f"3 hsd_monte_carlo_with_optgs_groupsize_n_trials_{N_TRIALS1}",
                algorithm=AlgorithmName.HSD_MONTE_CARLO,
                n_trials=N_TRIALS1,
                seed=seed,
                groupsize=optgs_groupsize,
            )
            for seed in range(REPEATS1)
        ]
        + [
            OptimizationConfiguration(
                name=f"4 hsd_monte_carlo_with_optgs_groupsize_n_trials_{N_TRIALS2}",
                algorithm=AlgorithmName.HSD_MONTE_CARLO,
                n_trials=N_TRIALS2,
                seed=seed,
                groupsize=optgs_groupsize,
            )
            for seed in range(REPEATS2)
        ]
        + [
            OptimizationConfiguration(
                name=f"5 hsd_monte_carlo_without_groupsize_n_trials_{N_TRIALS1}",
                algorithm=AlgorithmName.HSD_MONTE_CARLO,
                n_trials=N_TRIALS1,
                seed=seed,
                groupsize=None,
            )
            for seed in range(REPEATS1)
        ]
        + [
            OptimizationConfiguration(
                name=f"6 hsd_monte_carlo_without_groupsize_n_trials_{N_TRIALS2}",
                algorithm=AlgorithmName.HSD_MONTE_CARLO,
                n_trials=N_TRIALS2,
                seed=seed,
                groupsize=None,
            )
            for seed in range(REPEATS2)
        ]
        # + [ # Uncomment if you want to include GSDESIGN. This causes rare crushes so we removed it.
        #     OptimizationConfiguration(
        #         name="7 gsdesign",
        #         algorithm=AlgorithmName.GSDESIGN,
        #         n_trials=None,
        #         seed=seed,
        #         groupsize=None,
        #     )
        #     for seed in range(REPEATS2)
        # ]
        + [
            OptimizationConfiguration(
                name=f"8 general_spending_monte_carlo_with_optgs_groupsize_n_trials_{N_TRIALS1}",
                algorithm=AlgorithmName.GENERAL_SPENDING_MONTE_CARLO,
                n_trials=N_TRIALS1,
                seed=seed,
                groupsize=optgs_groupsize,
            )
            for seed in range(REPEATS1)
        ]
        + [
            OptimizationConfiguration(
                name=f"9 general_spending_monte_carlo_with_optgs_groupsize_n_trials_{N_TRIALS2}",
                algorithm=AlgorithmName.GENERAL_SPENDING_MONTE_CARLO,
                n_trials=N_TRIALS2,
                seed=seed,
                groupsize=optgs_groupsize,
            )
            for seed in range(REPEATS2)
        ]
    )

    runs = []
    for config in optimization_configs:
        result = run_optimization(input_parameters, config)
        runs.append(OptimizationRun(config, input_parameters, result))
        if verbose:
            print(f"\nCompleted run: {config.name} with seed: {config.seed}")
            print(f"Optimization took: {result.time:.2f} seconds")
            print(f"Groupsize: {result.groupsize}")
            print(
                f"ess_null: {result.ess_null}, ess_alt: {result.ess_alt}, cost: {result.cost}"
            )
            print(f"Type I error: {result.type_I_error}, Power: {result.power}")
            print(f"nfev: {result.nfev}")
        save_optimization_runs(runs, output_filename)


def analyse_design(input_filename: str):
    optimization_runs = load_optimization_runs(input_filename)
    run_names = sorted(set([run.config.name for run in optimization_runs]))
    optgs_cost = 100
    for run_name in run_names:

        runs = [run for run in optimization_runs if run.config.name == run_name]
        if len(runs) != REPEATS2 and len(runs) != 1:
            continue
        if run_name == f"4 hsd_monte_carlo_with_optgs_groupsize_n_trials_{N_TRIALS2}":
            continue
        print(f"====================")
        print(f"Algorithm: {run_name}")
        print(f"====================")
        print(f"Number of runs: {len(runs)}")

        # print(f"groupsizes: {[run.optimization_result.groupsize for run in runs]}")

        if runs[0].optimization_result.nfev is not None:
            print(
                f"avg n. function evaluations: ${np.mean([run.optimization_result.nfev for run in runs]):.0f} \pm {np.std([run.optimization_result.nfev for run in runs]):.0f}$"
            )
        # if runs[0].optimization_result.time is not None:
        #     print(
        #         f"avg time: {np.nanmean([run.optimization_result.time for run in runs])}"
        #     )
        # print(
        #     f"avg ess_null:{np.mean([run.optimization_result.ess_null for run in runs])}"
        # )
        # print(
        #     f"avg ess_alt:{np.mean([run.optimization_result.ess_alt for run in runs])}"
        # )
        if run_name == "0 optgs":
            optgs_cost = np.mean([run.optimization_result.cost for run in runs])
        print(f"OptGS cost: {optgs_cost:.1f}")
        print(
            f"avg cost rel to OptGS:${np.mean([run.optimization_result.cost / optgs_cost for run in runs]):.3f} \pm {np.std([run.optimization_result.cost/optgs_cost for run in runs]):.3f}$"
        )
        good_runs = [
            run
            for run in runs
            if (
                run.optimization_result.type_I_error
                <= run.input_parameters.alpha + 0.0001
                and run.optimization_result.power >= run.input_parameters.power - 0.001
            )
        ]
        if len(good_runs) > 0:
            print(
                f"min cost: {np.min([run.optimization_result.cost/optgs_cost for run in good_runs])} (n good runs: {len(good_runs)})"
            )

        print(
            f"avg type_I_error deviation: {np.mean([np.abs(run.optimization_result.type_I_error-run.input_parameters.alpha) for run in runs])}, std = {np.std([np.abs(run.optimization_result.type_I_error-run.input_parameters.alpha) for run in runs])}"
        )
        print(
            f"max type I error deviation: {np.max([np.abs(run.optimization_result.type_I_error - run.input_parameters.alpha) for run in runs])}"
        )
        print(
            f"avg power deviation: {np.mean([np.abs(run.optimization_result.power - run.input_parameters.power) for run in runs])}, std = {np.std([np.abs(run.optimization_result.power - run.input_parameters.power) for run in runs])}"
        )
        print(
            f"max power deviation: {np.max([np.abs(run.optimization_result.power - run.input_parameters.power) for run in runs])}"
        )


def compare_multiple_scenarios():
    effect_size = 0.3
    alpha = 0.025
    power = 0.8
    n_looks = 4
    weights = default_weights
    for n_looks in [2, 3, 4, 5, 6]:
        print("====================")
        print(f"n_looks: {n_looks}")
        print("====================")
        input_parameters = InputParameters(
            alpha=alpha,
            power=power,
            n_looks=n_looks,
            effect_size=effect_size,
            weights=weights,
            fixed_sample_size=0,  # Will be calculated based on alpha, power, and effect size
        )
        filename = f"optimization_results_n_looks_{n_looks}_old_run.json"

        # Uncomment the next line to run the scenarios and save the results
        # run_scenario(
        #     input_parameters=input_parameters,
        #     output_filename=filename,
        #     verbose=True,
        # )
        # This analyses the results of the previous runs

        # Uncomment the next line to analyse the results
        analyse_design(
            input_filename=filename,
        )


if __name__ == "__main__":
    compare_multiple_scenarios()
