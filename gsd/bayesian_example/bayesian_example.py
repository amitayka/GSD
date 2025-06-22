# gsd code for article - bayesian multi arm
import os
import pickle
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.optimize import differential_evolution

from gsd.basic_code.find_best_general_spending import find_best_general_spending
from gsd.basic_code.find_good_hsd_spending_functions import (
    find_best_hsd_spending,
    find_beta_for_alpha,
)
from gsd.basic_code.gsd_statistics_calculator import (
    GSDStatistics,
    get_statistics_given_thresholds,
)
from gsd.basic_code.gsd_threshold_finder_algorithm_1 import (
    get_efficacy_futility_thresholds,
)
from gsd.basic_code.utils.bayesian_approximation import generate_bayesian_samples
from gsd.basic_code.utils.find_fixed_sample_size import (
    find_fixed_sample_size_for_bayesian_endpoint,
)
from gsd.basic_code.utils.spending_function import (
    generate_spending_from_spending_parameter,
)

N_TRIALS_FOR_VERIFY = 10_000_000  # change to 10_000_000 for final version

# N_TRIALS1 = 10_000
# REPEATS1 = 1
# N_TRIALS_BEST_THRESHOLDS = 5_000
N_TRIALS1 = 200_000
REPEATS1 = 1
N_TRIALS_BEST_THRESHOLDS = 50_000
REPEATS_BEST_THRESHOLDS = 1

N_SCENARIOS_PER_OPTION = 5


@dataclass
class InputParameters:
    rate_per_arm_h0: np.ndarray
    rate_per_arm_h1: np.ndarray
    alpha: float
    power: float
    looks_fractions: np.ndarray
    sample_size_multiplier: float
    is_binding: bool
    option_name: str = "default"

    def __post_init__(self):
        self.n_looks = len(self.looks_fractions)
        self.n_arms = len(self.rate_per_arm_h0)
        self.beta = 1 - self.power

        self.fixed_size_per_arm = find_fixed_sample_size_for_bayesian_endpoint(
            alpha=self.alpha,
            power=self.power,
            rate_per_arm_h0=self.rate_per_arm_h0,
            rate_per_arm_h1=self.rate_per_arm_h1,
            seed=42,
        )
        self.fixed_size = self.fixed_size_per_arm * self.n_arms
        self.n_samples = (
            int(self.fixed_size * self.sample_size_multiplier)
            // self.n_arms
            * self.n_arms
        )
        self.n_samples_per_look_total = (
            (self.n_samples * self.looks_fractions).astype(int)
            // self.n_arms
            * self.n_arms
        )
        self.n_samples_per_arm_per_look = self.n_samples_per_look_total // self.n_arms
        self.looks_fractions = (
            self.n_samples_per_arm_per_look / self.n_samples_per_arm_per_look[-1]
        )

    def print_parameters(self):
        print("========================================")
        print("Input parameters:")
        print("========================================")
        print(f"rate_per_arm_h0 = {self.rate_per_arm_h0}")
        print(f"rate_per_arm_h1 = {self.rate_per_arm_h1}")
        print(f"alpha = {self.alpha}")
        print(f"power = {self.power}")
        print(f"looks_fractions = {self.looks_fractions}")
        print(f"sample_size_multiplier = {self.sample_size_multiplier}")
        print(f"is_binding = {self.is_binding}")
        print(f"fixed_size_per_arm = {self.fixed_size_per_arm}")
        print(f"fixed_size = {self.fixed_size}")
        print(f"n_looks = {self.n_looks}")
        print(f"n_samples_per_look_total = {self.n_samples_per_look_total}")
        print(f"n_samples_per_arm_per_look = {self.n_samples_per_arm_per_look}")


class AlgorithmName(Enum):
    FIND_BEST_GENERAL_SPENDING = "find_best_general_spending"
    FIND_BEST_HSD_SPENDING = "hsd"
    FIND_CONSTANT_THRESHOLDS = "constant_thresholds_binding"
    FIND_BEST_THRESHOLDS = "find_best_thresholds"


@dataclass
class OptimizationParameters:
    algorithm: AlgorithmName
    name: str
    seed: int
    n_trials: int


@dataclass
class OptimizationResult:
    alpha_spending_parameter: Optional[float]
    beta_spending_parameter: Optional[float]
    nfev: Optional[int]
    futility_thresholds: np.ndarray
    efficacy_thresholds: np.ndarray
    ess_null: float
    ess_alt: float
    cost: float
    type_I_error: float
    non_binding_type_I_error: float
    power: float
    time: float  # Time taken for the optimization run


@dataclass
class OptimizationRun:
    input_parameters: InputParameters
    optimization_parameters: OptimizationParameters
    result: OptimizationResult


def save_optimization_runs(runs: list[OptimizationRun], filename: str):
    """Serialize and save a list of OptimizationRun objects to a file."""
    with open(filename, "wb") as f:
        pickle.dump(runs, f)


def load_optimization_runs(filename: str) -> list[OptimizationRun]:
    """Load and deserialize a list of OptimizationRun objects from a file."""
    with open(filename, "rb") as f:
        return pickle.load(f)


###################################################################################################
# Generate samples
###################################################################################################


def generate_samples(
    input_parameters: InputParameters,
    optimization_parameters: OptimizationParameters,
):
    rng = np.random.default_rng(optimization_parameters.seed)
    samples_h0 = generate_bayesian_samples(
        n_samples_per_look_per_arm=input_parameters.n_samples_per_arm_per_look,
        rate_per_arm=input_parameters.rate_per_arm_h0,
        n_trials=optimization_parameters.n_trials,
        rng=rng,
    )
    samples_h1 = generate_bayesian_samples(
        n_samples_per_look_per_arm=input_parameters.n_samples_per_arm_per_look,
        rate_per_arm=input_parameters.rate_per_arm_h1,
        n_trials=optimization_parameters.n_trials,
        rng=rng,
    )
    return samples_h0, samples_h1


#####################################################################################################
# Find the best spending function of HSD-form, and generate the tables for the paper
#####################################################################################################
def find_best_hsd_alpha_beta_spending(
    input_parameters: InputParameters,
    optimization_parameters: OptimizationParameters,
    print_output=False,
    verbose=False,
):
    start = time.time()
    samples_h0, samples_h1 = generate_samples(input_parameters, optimization_parameters)
    (
        best_alpha_spending_parameter,
        best_beta_spending_parameter,
        differential_evolution_result,
    ) = find_best_hsd_spending(
        samples_h0,
        samples_h1,
        input_parameters.looks_fractions,
        input_parameters.n_samples_per_arm_per_look,
        input_parameters.alpha,
        1 - input_parameters.power,
        input_parameters.is_binding,
        seed=optimization_parameters.seed,
        verbose=verbose,
    )

    best_alpha_spending = generate_spending_from_spending_parameter(
        best_alpha_spending_parameter,
        input_parameters.alpha,
        input_parameters.looks_fractions,
    )
    best_beta_spending = generate_spending_from_spending_parameter(
        best_beta_spending_parameter,
        input_parameters.beta,
        input_parameters.looks_fractions,
    )

    samples_h0_verify = generate_bayesian_samples(
        n_samples_per_look_per_arm=input_parameters.n_samples_per_arm_per_look,
        rate_per_arm=input_parameters.rate_per_arm_h0,
        n_trials=N_TRIALS_FOR_VERIFY,
        rng=np.random.default_rng(optimization_parameters.seed),
    )
    samples_h1_verify = generate_bayesian_samples(
        n_samples_per_look_per_arm=input_parameters.n_samples_per_arm_per_look,
        rate_per_arm=input_parameters.rate_per_arm_h1,
        n_trials=N_TRIALS_FOR_VERIFY,
        rng=np.random.default_rng(optimization_parameters.seed),
    )

    best_efficacy_thresholds, best_futility_thresholds = (
        get_efficacy_futility_thresholds(
            samples_h0,
            samples_h1,
            best_alpha_spending,
            best_beta_spending,
            is_binding=input_parameters.is_binding,
        )
    )
    best_stats_h0_without_futility = get_statistics_given_thresholds(
        samples_h0_verify,
        efficacy_thresholds=best_efficacy_thresholds,
        futility_thresholds=np.zeros_like(best_futility_thresholds),
        n_samples_per_look_per_arm=input_parameters.n_samples_per_arm_per_look,
    )
    best_stats_h0 = get_statistics_given_thresholds(
        samples_h0_verify,
        efficacy_thresholds=best_efficacy_thresholds,
        futility_thresholds=best_futility_thresholds,
        n_samples_per_look_per_arm=input_parameters.n_samples_per_arm_per_look,
    )
    best_stats_h1 = get_statistics_given_thresholds(
        samples_h1_verify,
        efficacy_thresholds=best_efficacy_thresholds,
        futility_thresholds=best_futility_thresholds,
        n_samples_per_look_per_arm=input_parameters.n_samples_per_arm_per_look,
    )
    best_cost = (
        best_stats_h0.average_sample_size + best_stats_h1.average_sample_size
    ) / 2
    result = OptimizationResult(
        alpha_spending_parameter=best_alpha_spending_parameter,
        beta_spending_parameter=best_beta_spending_parameter,
        cost=best_cost,
        nfev=differential_evolution_result.nfev,
        futility_thresholds=best_futility_thresholds,
        efficacy_thresholds=best_efficacy_thresholds,
        ess_null=best_stats_h0.average_sample_size,
        ess_alt=best_stats_h1.average_sample_size,
        non_binding_type_I_error=best_stats_h0_without_futility.disjunctive_power,
        type_I_error=best_stats_h0.disjunctive_power,
        power=best_stats_h1.disjunctive_power,
        time=time.time() - start,
    )
    if not print_output:
        return result

    print("========================================")
    print("Best alpha and beta spending parameters:")
    print("========================================")
    print(f"Time to find hsd spending function = {time.time()-start:.2f} seconds")
    print("number of function evaluations = ", differential_evolution_result.nfev)
    print("best_alpha_spending_parameter = ", best_alpha_spending_parameter)
    print("best_beta_spending_parameter = ", best_beta_spending_parameter)
    print(f"best_alpha_spending_parameter = {best_alpha_spending_parameter}")
    print(f"best_beta_spending_parameter = {best_beta_spending_parameter}")
    print(f"best_alpha_spending = {best_alpha_spending}")
    print(f"best_beta_spending = {best_beta_spending}")
    print(f"best_efficacy_thresholds = {best_efficacy_thresholds}")
    print(f"best_futility_thresholds = {best_futility_thresholds}")

    print(
        f"best_alpha_spending_parameter = {best_alpha_spending_parameter}. best_beta_spending_parameter = {best_beta_spending_parameter}"
    )
    print(
        f"best_stats_h0.average_sample_size = {best_stats_h0.average_sample_size} (Relative to fixed: {best_stats_h0.average_sample_size/input_parameters.fixed_size})"
    )
    print(
        f"best_stats_h1.average_sample_size = {best_stats_h1.average_sample_size} (Relative to fixed: {best_stats_h1.average_sample_size/input_parameters.fixed_size})"
    )

    groups = ["0.5", "0.55", "0.6", "0.7", "H0", "H1"]
    efficacy_stop_probs_arm05 = best_stats_h0.efficacy_probs_per_arm_per_look[0]
    efficacy_stop_probs_arm055 = best_stats_h1.efficacy_probs_per_arm_per_look[2]
    efficacy_stop_probs_arm06 = best_stats_h1.efficacy_probs_per_arm_per_look[1]
    efficacy_stop_probs_arm07 = best_stats_h1.efficacy_probs_per_arm_per_look[0]
    efficacy_stop_probs_h0 = best_stats_h0.efficacy_probs_trial_per_look
    efficacy_stop_probs_h1 = best_stats_h1.efficacy_probs_trial_per_look

    # Combine all efficacy stop probabilities into a list of lists
    efficacy_stop_probs = [
        efficacy_stop_probs_arm05,
        efficacy_stop_probs_arm055,
        efficacy_stop_probs_arm06,
        efficacy_stop_probs_arm07,
        efficacy_stop_probs_h0,
        efficacy_stop_probs_h1,
    ]

    # Number of groups and number of bars per group
    n_groups = len(groups)
    n_bars = len(efficacy_stop_probs[0])  # Assuming each list has the same length

    # Define a list of colors for the bars
    blue = "#009CCC"
    gold = "#EBD10A"
    purple = "#362C9A"
    green = "#799527"
    colors = [blue, gold, purple, green, "c", "m", "y", "k"]

    # Create a figure and a set of subplots
    fig, ax = plt.subplots(dpi=800)
    plt.grid(True, color="gray", linestyle="--", linewidth=0.5)

    # Define the bar width and the index for the groups
    bar_width = 0.15
    index = np.arange(n_groups).astype(float)

    extra_diff = 0.5
    split_index = 4  # Index where the split should occur
    index[split_index:] += extra_diff
    # Plot the bars for each group
    for i in range(n_bars):
        ax.bar(
            index + i * bar_width,
            [efficacy_stop_probs[j][i] for j in range(n_groups)],
            bar_width,
            label=f"Interim {i+1}" if i != n_bars - 1 else "Final Analysis",
            color=colors[i],
            alpha=1,
        )

    # Add a vertical line to split the graph into two parts

    diff_size = 1 + extra_diff
    ax.axvline(
        x=index[split_index] + bar_width * (n_bars - 1) / 2 - diff_size / 2,
        color="black",
        linewidth=2,
    )

    # Add labels, title, and legend
    ax.set_xlabel("Rate / Scenario")
    ax.set_ylabel("Efficacy Stopping Probabilities")
    ax.set_title("Efficacy Stopping Probabilities")
    ax.set_xticks(index + bar_width * (n_bars - 1) / 2)
    ax.set_xticklabels(groups)
    ax.legend(framealpha=1)

    # Display the plot
    plt.savefig("efficacy_stop.eps", format="eps")

    futility_stop_probs_arm05 = best_stats_h0.futility_probs_per_arm_per_look[0]
    futility_stop_probs_arm055 = best_stats_h1.futility_probs_per_arm_per_look[2]
    futility_stop_probs_arm06 = best_stats_h1.futility_probs_per_arm_per_look[1]
    futility_stop_probs_arm07 = best_stats_h1.futility_probs_per_arm_per_look[0]
    futility_stop_probs_h0 = best_stats_h0.futility_probs_trial_per_look
    futility_stop_probs_h1 = best_stats_h1.futility_probs_trial_per_look

    # Combine all futility stop probabilities into a list of lists
    futility_stop_probs = [
        futility_stop_probs_arm05,
        futility_stop_probs_arm055,
        futility_stop_probs_arm06,
        futility_stop_probs_arm07,
        futility_stop_probs_h0,
        futility_stop_probs_h1,
    ]

    # Number of groups and number of bars per group
    n_groups = len(groups)
    n_bars = len(futility_stop_probs[0])  # Assuming each list has the same length

    # Create a figure and a set of subplots
    fig, ax = plt.subplots(dpi=800)
    plt.grid(True, color="gray", linestyle="--", linewidth=0.5)

    # Define the bar width and the index for the groups
    bar_width = 0.15
    index = np.arange(n_groups).astype(float)

    extra_diff = 0.5
    split_index = 4  # Index where the split should occur
    index[split_index:] += extra_diff

    # Plot each set of bars with specified colors
    for i in range(n_bars):
        bar_values = [futility_stop_probs[j][i] for j in range(n_groups)]
        label_name = f"Interim{i+1}" if i < n_bars - 1 else "Final Analysis"
        ax.bar(
            index + i * bar_width,
            bar_values,
            bar_width,
            label=label_name,
            color=colors[i % len(colors)],
            alpha=1,
        )

    diff_size = 1 + extra_diff
    ax.axvline(
        x=index[split_index] + bar_width * (n_bars - 1) / 2 - diff_size / 2,
        color="black",
        linewidth=2,
    )

    # Add labels, title, and legend
    ax.set_xlabel("Rate / Scenario")
    ax.set_ylabel("Futility Stopping Probabilities")
    ax.set_title("Futility Stopping Probabilities")
    ax.set_xticks(index + bar_width * (n_bars - 1) / 2)
    ax.set_xticklabels(groups)
    ax.legend(framealpha=1)

    # Display the plot
    plt.savefig("futility_stop.eps", format="eps")
    return result


################################################################################################
# Print the graphs for changing alpha for the paper
################################################################################################
@dataclass
class ParametersStatistics:
    alpha_spending_parameter: float
    beta_spending_parameter: float
    stats_h0: GSDStatistics
    stats_h1: GSDStatistics


def print_alpha_beta_graphs(
    input_parameters: InputParameters, optimization_parameters: OptimizationParameters
):
    start = time.time()
    samples_h0, samples_h1 = generate_samples(input_parameters, optimization_parameters)

    alpha_parameter_space = np.concatenate(
        (
            np.linspace(-10, -4, 4),
            np.linspace(-3.5, -2.5, 3),
            np.linspace(-2.25, 2, 18),
            np.linspace(2.5, 3.5, 3),
        )
    )

    results: list[ParametersStatistics] = []

    for alpha_spending_parameter in alpha_parameter_space:
        beta_spending_parameter = find_beta_for_alpha(
            alpha_spending_parameter=alpha_spending_parameter,
            samples_h0=samples_h0,
            samples_h1=samples_h1,
            looks_fractions=input_parameters.looks_fractions,
            n_samples_per_arm_per_look=input_parameters.n_samples_per_arm_per_look,
            alpha=input_parameters.alpha,
            beta=input_parameters.beta,
            is_binding=input_parameters.is_binding,
        )

        alpha_spending = generate_spending_from_spending_parameter(
            alpha_spending_parameter,
            input_parameters.alpha,
            input_parameters.looks_fractions,
        )
        beta_spending = generate_spending_from_spending_parameter(
            beta_spending_parameter,
            input_parameters.beta,
            input_parameters.looks_fractions,
        )
        # alpha_spending = np.zeros_like(alpha_spending)
        # alpha_spending[-1] = alpha
        # beta_spending = np.zeros_like(beta_spending)
        # beta_spending[-1] = beta
        efficacy_thresholds, futility_thresholds = get_efficacy_futility_thresholds(
            samples_h0,
            samples_h1,
            alpha_spending,
            beta_spending,
            is_binding=input_parameters.is_binding,
        )
        stats_h1 = get_statistics_given_thresholds(
            samples_h1,
            efficacy_thresholds,
            futility_thresholds,
            n_samples_per_look_per_arm=input_parameters.n_samples_per_arm_per_look,
        )
        stats_h0 = get_statistics_given_thresholds(
            samples_h0,
            efficacy_thresholds,
            futility_thresholds,
            n_samples_per_look_per_arm=input_parameters.n_samples_per_arm_per_look,
        )

        results.append(
            ParametersStatistics(
                alpha_spending_parameter=alpha_spending_parameter,
                beta_spending_parameter=beta_spending_parameter,
                stats_h0=stats_h0,
                stats_h1=stats_h1,
            )
        )
        relevant_results = [result for result in results]
    alpha_parameter = [result.alpha_spending_parameter for result in relevant_results]
    beta_parameter = [result.beta_spending_parameter for result in relevant_results]
    type_II_error = [
        1 - result.stats_h1.disjunctive_power for result in relevant_results
    ]
    type_I_error = [result.stats_h0.disjunctive_power for result in relevant_results]
    average_h0 = [result.stats_h0.average_sample_size for result in relevant_results]
    average_h1 = [result.stats_h1.average_sample_size for result in relevant_results]

    # Create a figure and a set of subplots
    fig, ax1 = plt.subplots(figsize=(14, 6), dpi=800)

    # Define the index for the groups
    index = np.arange(len(alpha_parameter))

    # Plot alpha vs beta on the primary y-axis
    ax1.plot(index, beta_parameter, "b-", label="Beta Parameter", linewidth=2, alpha=1)
    plt.grid(True, color="gray", linestyle="--", linewidth=0.5)

    ax1.set_xlabel("Alpha Parameter")
    ax1.set_ylabel("Beta Parameter", color="black")
    ax1.set_xticks(index)
    ax1.set_xticklabels(alpha_parameter)
    ax1.tick_params(axis="y", labelcolor="black")
    plt.title("Alpha Parameter vs Beta Parameter")

    plt.savefig("alpha_parameter_vs_beta_parameter.eps", format="eps")

    # Create a figure and a set of subplots
    fig, ax2 = plt.subplots(figsize=(14, 6), dpi=800)
    plt.grid(True, color="gray", linestyle="--", linewidth=0.5)

    # Define the index for the groups
    index = np.arange(len(alpha_parameter))
    # Create a secondary y-axis to plot type_I_error
    # ax2 = ax1.twinx()
    ax2.set_xlabel("Alpha Parameter")
    ax2.set_xticks(index)
    ax2.set_xticklabels(alpha_parameter)
    ax2.plot(index, type_I_error, "r-", label="Type I Error Rate", linewidth=2, alpha=1)
    ax2.set_ylabel("Type I Error Rate", color="r")
    ax2.tick_params(axis="y", labelcolor="r")
    ax2.set_ybound(0, 0.03)

    # Create a third y-axis to plot type_II_error
    ax3 = ax2.twinx()

    plt.grid(True, color="gray", linestyle="--", linewidth=0.5)
    ax3.spines["right"].set_position(("outward", 0))  # Offset the third axis
    ax3.plot(
        index,
        type_II_error,
        color="b",
        label="Type II Error Rate",
        linestyle="--",
        linewidth=2,
        alpha=1,
    )
    ax3.set_ylabel("Type II Error Rate", color="b")
    # ax3.tick_params(axis='y', labelcolor='b')
    ax3.set_ybound(0, 0.3)
    plt.grid(True, color="gray", linestyle="--", linewidth=0.5)

    # Add legends
    fig.legend(
        loc="upper right",
        bbox_to_anchor=(0.9, 0.7),
        bbox_transform=ax1.transAxes,
        framealpha=1,
    )

    # Add title
    plt.title("Alpha Parameter vs Type I and Type II Error Rates")
    plt.grid(True, color="gray", linestyle="--", linewidth=0.5)
    plt.savefig("alpha_parameter_vs_error_rates.eps", format="eps")

    # alpha_parameter_vs_error_rates
    fig, ax1 = plt.subplots(figsize=(14, 6), dpi=800)
    plt.grid(True, color="gray", linestyle="--", linewidth=0.5)
    max_sample_size = input_parameters.n_samples_per_look_total[-1]

    # Define the bar width
    bar_width = 0.35

    # Define the index for the groups
    index = np.arange(len(alpha_parameter))

    # Adjust the index for the second set of bars to place them side by side
    index2 = index + bar_width

    dark_purple = "#2C1E5C"
    dark_gold = "#C5AE00"

    # Plot the stacked bars for average_h0 and average_h1
    bars1 = ax1.bar(
        index,
        average_h0,
        bar_width,
        label="Average Sample Size Under H0",
        color=dark_purple,
        alpha=1,
    )
    bars2 = ax1.bar(
        index2,
        average_h1,
        bar_width,
        label="Average Sample Size under H1",
        color=dark_gold,
        alpha=1,
    )

    # Add a horizontal line for fixed_size
    ax1.axhline(
        y=input_parameters.fixed_size, color="r", linestyle="--", label="Fixed Size"
    )
    ax1.axhline(y=max_sample_size, color="b", linestyle="--", label="Max Sample Size")

    # Add labels, title, and legend for the first y-axis
    ax1.set_xlabel("Alpha Parameter")
    ax1.set_ylabel("Average Sample size under H0 and H1")
    ax1.set_title("Alpha Parameter vs Average Sample Size under H0 and H1")
    ax1.set_xticks(index + bar_width / 2)
    ax1.set_xticklabels(alpha_parameter)
    ax1.legend(framealpha=1)

    # Display the plot
    plt.savefig("average_sample_size.eps", format="eps")
    print(f"Time to generate alpha-beta graphs = {time.time() - start:.2f} seconds")


################################################################################################
# Find the constant thresholds
################################################################################################


def find_constant_thresholds_binding(
    input_parameters: InputParameters,
    optimization_parameters: OptimizationParameters,
    print_output: bool = False,
):
    start = time.time()
    n_iterations = 25
    efficacy_threshold_lower = 0.9
    efficacy_threshold_upper = 1
    samples_h0, samples_h1 = generate_samples(input_parameters, optimization_parameters)
    for i in range(n_iterations):
        mid_value = (efficacy_threshold_lower + efficacy_threshold_upper) / 2
        efficacy_thresholds_constant = np.array(
            [mid_value] * len(input_parameters.looks_fractions)
        )
        futility_thresholds_constant = np.array(
            [0.0] * len(input_parameters.looks_fractions)
        )
        stats_h0 = get_statistics_given_thresholds(
            samples_h0,
            efficacy_thresholds_constant,
            futility_thresholds_constant,
            n_samples_per_look_per_arm=input_parameters.n_samples_per_arm_per_look,
        )
        type_I_error = np.sum(stats_h0.efficacy_probs_trial_per_look)
        if type_I_error > input_parameters.alpha:
            efficacy_threshold_lower = mid_value
        else:
            efficacy_threshold_upper = mid_value
    efficacy_thresholds_constant = np.array(
        [efficacy_threshold_upper] * len(input_parameters.looks_fractions)
    )
    futility_threshold_lower = 0
    futility_threshold_upper = 0.9
    for i in range(n_iterations):
        mid_value = (futility_threshold_lower + futility_threshold_upper) / 2
        futility_thresholds_constant = np.array(
            [mid_value] * len(input_parameters.looks_fractions)
        )
        stats_h1 = get_statistics_given_thresholds(
            samples_h1,
            efficacy_thresholds_constant,
            futility_thresholds_constant,
            n_samples_per_look_per_arm=input_parameters.n_samples_per_arm_per_look,
        )
        power = np.sum(stats_h1.efficacy_probs_trial_per_look)
        if power > 1 - input_parameters.beta:
            futility_threshold_lower = mid_value
        else:
            futility_threshold_upper = mid_value
    stats_h0_with_futility = get_statistics_given_thresholds(
        samples_h0,
        efficacy_thresholds_constant,
        np.zeros_like(futility_thresholds_constant),
        n_samples_per_look_per_arm=input_parameters.n_samples_per_arm_per_look,
    )

    stats_h0 = get_statistics_given_thresholds(
        samples_h0,
        efficacy_thresholds_constant,
        futility_thresholds_constant,
        n_samples_per_look_per_arm=input_parameters.n_samples_per_arm_per_look,
    )
    stats_h1 = get_statistics_given_thresholds(
        samples_h1,
        efficacy_thresholds_constant,
        futility_thresholds_constant,
        n_samples_per_look_per_arm=input_parameters.n_samples_per_arm_per_look,
    )
    if print_output:
        print("===============================")
        print("Constant bounds:")
        print("===============================")
        print(f"efficacy_thresholds_constant = {efficacy_thresholds_constant}")
        print(f"futility_thresholds_constant = {futility_thresholds_constant}")
        print(f"actual type I error = {np.sum(stats_h0.efficacy_probs_trial_per_look)}")
        print(
            f"efficacy_probs_trial_per_look = {stats_h0.efficacy_probs_trial_per_look}"
        )
        print(f"actual power = {np.sum(stats_h1.efficacy_probs_trial_per_look)}")
        print(
            f"efficacy_probs_trial_per_look = {stats_h1.efficacy_probs_trial_per_look}"
        )
        print(f"average_sample_size_h0 = {stats_h0.average_sample_size}")
        print(f"average_sample_size_h1 = {stats_h1.average_sample_size}")

    return OptimizationResult(
        alpha_spending_parameter=None,
        beta_spending_parameter=None,
        nfev=None,
        futility_thresholds=futility_thresholds_constant,
        efficacy_thresholds=efficacy_thresholds_constant,
        ess_null=stats_h0.average_sample_size,
        ess_alt=stats_h1.average_sample_size,
        cost=0.5 * stats_h0.average_sample_size + 0.5 * stats_h1.average_sample_size,
        non_binding_type_I_error=stats_h0_with_futility.disjunctive_power,
        type_I_error=stats_h0.disjunctive_power,
        power=stats_h1.disjunctive_power,
        time=time.time() - start,
    )


#################################################################################################
# Search for the best spending function, without limiting to hsw spending functions
#################################################################################################


def find_best_general_spending_function(
    input_parameters: InputParameters,
    optimization_parameters: OptimizationParameters,
    verbose=False,
    print_output=False,
):
    start = time.time()
    samples_h0, samples_h1 = generate_samples(input_parameters, optimization_parameters)
    alpha_spending, beta_spending, differential_evolution_result = (
        find_best_general_spending(
            samples_h0=samples_h0,
            samples_h1=samples_h1,
            looks_fractions=input_parameters.looks_fractions,
            n_samples_per_arm_per_look=input_parameters.n_samples_per_arm_per_look,
            alpha=input_parameters.alpha,
            beta=input_parameters.beta,
            is_binding=input_parameters.is_binding,
            seed=optimization_parameters.seed,
            verbose=verbose,
        )
    )
    samples_h0_verify = generate_bayesian_samples(
        n_samples_per_look_per_arm=input_parameters.n_samples_per_arm_per_look,
        rate_per_arm=input_parameters.rate_per_arm_h0,
        n_trials=N_TRIALS_FOR_VERIFY,
        rng=np.random.default_rng(optimization_parameters.seed),
    )
    samples_h1_verify = generate_bayesian_samples(
        n_samples_per_look_per_arm=input_parameters.n_samples_per_arm_per_look,
        rate_per_arm=input_parameters.rate_per_arm_h1,
        n_trials=N_TRIALS_FOR_VERIFY,
        rng=np.random.default_rng(optimization_parameters.seed),
    )
    (
        efficacy_thresholds_differential_evolution,
        futility_thresholds_differential_evolution,
    ) = get_efficacy_futility_thresholds(
        samples_h0,
        samples_h1,
        alpha_spending,
        beta_spending,
        is_binding=input_parameters.is_binding,
    )
    stats_h0_without_futility = get_statistics_given_thresholds(
        samples_h0_verify,
        efficacy_thresholds_differential_evolution,
        futility_thresholds=np.zeros_like(futility_thresholds_differential_evolution),
        n_samples_per_look_per_arm=input_parameters.n_samples_per_arm_per_look,
    )
    stats_h0 = get_statistics_given_thresholds(
        samples_h0_verify,
        efficacy_thresholds_differential_evolution,
        futility_thresholds_differential_evolution,
        n_samples_per_look_per_arm=input_parameters.n_samples_per_arm_per_look,
    )
    stats_h1 = get_statistics_given_thresholds(
        samples_h1_verify,
        efficacy_thresholds_differential_evolution,
        futility_thresholds_differential_evolution,
        n_samples_per_look_per_arm=input_parameters.n_samples_per_arm_per_look,
    )
    if print_output:
        print("===============================")
        print("Best spending functions found:")
        print("===============================")
        print(f"Time to find spending function = {time.time()-start:.2f} seconds")
        print(f"number of function evaluations = {differential_evolution_result.nfev}")
        print(f"alpha_spending_differential_evolution = {alpha_spending}")
        print(f"beta_spending_differential_evolution = {beta_spending}")
        print(
            f"efficacy_thresholds_differential_evolution = {efficacy_thresholds_differential_evolution}"
        )
        print(
            f"futility_thresholds_differential_evolution = {futility_thresholds_differential_evolution}"
        )
        print(f"actual type I error = {np.sum(stats_h0.efficacy_probs_trial_per_look)}")
        print(f"actual power = {np.sum(stats_h1.efficacy_probs_trial_per_look)}")
        print(f"average_sample_size_h0 = {stats_h0.average_sample_size}")
        print(f"average_sample_size_h1 = {stats_h1.average_sample_size}")
    return OptimizationResult(
        alpha_spending_parameter=differential_evolution_result.x[0],
        beta_spending_parameter=differential_evolution_result.x[1],
        nfev=differential_evolution_result.nfev,
        futility_thresholds=futility_thresholds_differential_evolution,
        efficacy_thresholds=efficacy_thresholds_differential_evolution,
        ess_null=stats_h0.average_sample_size,
        ess_alt=stats_h1.average_sample_size,
        cost=0.5 * stats_h0.average_sample_size + 0.5 * stats_h1.average_sample_size,
        non_binding_type_I_error=stats_h0_without_futility.disjunctive_power,
        type_I_error=stats_h0.disjunctive_power,
        power=stats_h1.disjunctive_power,
        time=time.time() - start,
    )


################################################################################################
# Find best thresholds directly
################################################################################################

# search for the best thresholds directly
function_evaluations = 0


def objective_function_thresholds(
    input: np.ndarray,
    input_parameters: InputParameters,
    samples_h0,
    samples_h1,
    verbose=False,
):
    global function_evaluations
    function_evaluations += 1
    assert input.shape[0] == 2 * len(input_parameters.looks_fractions) - 1

    logit_futility_thresholds = input[: len(input_parameters.looks_fractions)]
    diffs = input[len(input_parameters.looks_fractions) :]
    logit_efficacy_thresholds = logit_futility_thresholds + np.concatenate((diffs, [0]))
    futility_thresholds = sp.special.expit(logit_futility_thresholds)
    efficacy_thresholds = sp.special.expit(logit_efficacy_thresholds)
    if input_parameters.is_binding:
        futility_thresholds_for_type_I_error = futility_thresholds
    else:
        futility_thresholds_for_type_I_error = np.zeros(
            len(input_parameters.looks_fractions)
        )

    stats_h0_for_alpha = get_statistics_given_thresholds(
        samples_h0,
        efficacy_thresholds,
        futility_thresholds_for_type_I_error,
        n_samples_per_look_per_arm=input_parameters.n_samples_per_arm_per_look,
    )
    type_I_error = np.sum(stats_h0_for_alpha.efficacy_probs_trial_per_look)
    if verbose:
        print(
            f"evaluations= {function_evaluations}, input = {input}, futility_thresholds = {futility_thresholds}, efficacy_thresholds = {efficacy_thresholds}, type_I_error = {type_I_error}"
        )
    if type_I_error > input_parameters.alpha:
        if verbose:
            print(f"type I error = {type_I_error}")
        return 10000 * (3 - type_I_error)

    stats_h1 = get_statistics_given_thresholds(
        samples_h1,
        efficacy_thresholds,
        futility_thresholds,
        n_samples_per_look_per_arm=input_parameters.n_samples_per_arm_per_look,
    )
    power = np.sum(stats_h1.efficacy_probs_trial_per_look)
    if power < 1 - input_parameters.beta:
        if verbose:
            print(f"type I error = {type_I_error},power = {power}")
        return 10000 * (2 - power)
    stats_h0 = get_statistics_given_thresholds(
        samples_h0,
        efficacy_thresholds,
        futility_thresholds,
        n_samples_per_look_per_arm=input_parameters.n_samples_per_arm_per_look,
    )
    if verbose:
        print(
            f"type I error = {type_I_error}, power = {power}, average sample size = {stats_h1.average_sample_size + stats_h0.average_sample_size}"
        )
    return stats_h1.average_sample_size + stats_h0.average_sample_size


def find_best_thresholds(
    input_parameters, optimization_parameters, verbose=False, print_output=False
):
    max_prob = 0.999  # for efficacy
    min_prob = 0.1  # for futility
    global function_evaluations
    function_evaluations = 0
    samples_h0, samples_h1 = generate_samples(input_parameters, optimization_parameters)
    min_logit = float(np.log(min_prob / (1 - min_prob)))
    max_logit = float(np.log(max_prob / (1 - max_prob)))
    bounds = [(min_logit, max_logit)] * len(input_parameters.looks_fractions) + [
        (0.0, max_logit - min_logit)
    ] * (len(input_parameters.looks_fractions) - 1)
    bounds = np.array(bounds)
    start = time.time()

    def objective_function_thresholds_wrapper(input):
        return objective_function_thresholds(
            input,
            input_parameters,
            samples_h0,
            samples_h1,
            verbose=verbose,
        )

    differential_evolution_result = differential_evolution(
        objective_function_thresholds_wrapper, bounds
    )
    input = differential_evolution_result.x
    logit_futility_thresholds = input[: len(input_parameters.looks_fractions)]
    diffs = input[len(input_parameters.looks_fractions) :]
    logit_efficacy_thresholds = logit_futility_thresholds + np.concatenate((diffs, [0]))
    futility_thresholds = sp.special.expit(logit_futility_thresholds)
    efficacy_thresholds = sp.special.expit(logit_efficacy_thresholds)
    samples_h0_verify = generate_bayesian_samples(
        n_samples_per_look_per_arm=input_parameters.n_samples_per_arm_per_look,
        rate_per_arm=input_parameters.rate_per_arm_h0,
        n_trials=N_TRIALS_FOR_VERIFY,
        rng=np.random.default_rng(optimization_parameters.seed),
    )
    samples_h1_verify = generate_bayesian_samples(
        n_samples_per_look_per_arm=input_parameters.n_samples_per_arm_per_look,
        rate_per_arm=input_parameters.rate_per_arm_h1,
        n_trials=N_TRIALS_FOR_VERIFY,
        rng=np.random.default_rng(optimization_parameters.seed),
    )
    stats_h0_without_futility = get_statistics_given_thresholds(
        samples_h0_verify,
        efficacy_thresholds,
        np.zeros_like(futility_thresholds),
        n_samples_per_look_per_arm=input_parameters.n_samples_per_arm_per_look,
    )

    stats_h0 = get_statistics_given_thresholds(
        samples_h0_verify,
        efficacy_thresholds,
        futility_thresholds,
        n_samples_per_look_per_arm=input_parameters.n_samples_per_arm_per_look,
    )
    stats_h1 = get_statistics_given_thresholds(
        samples_h1_verify,
        efficacy_thresholds,
        futility_thresholds,
        n_samples_per_look_per_arm=input_parameters.n_samples_per_arm_per_look,
    )
    if print_output:
        print("===========================")
        print("Best thresholds found:")
        print("===========================")
        print(f"Time to find thresholds = {time.time()-start:.2f} seconds")
        print(
            f"differential evolution function evaluations = {differential_evolution_result.nfev}"
        )
        print(f"efficacy_thresholds = {efficacy_thresholds}")
        print(f"futility_thresholds = {futility_thresholds}")
        print(f"actual type I error = {np.sum(stats_h0.efficacy_probs_trial_per_look)}")
        print(f"actual power = {np.sum(stats_h1.efficacy_probs_trial_per_look)}")
        print(f"average_sample_size_h0 = {stats_h0.average_sample_size}")
        print(f"average_sample_size_h1 = {stats_h1.average_sample_size}")
    return OptimizationResult(
        alpha_spending_parameter=None,
        beta_spending_parameter=None,
        nfev=differential_evolution_result.nfev,
        time=time.time() - start,
        futility_thresholds=futility_thresholds,
        efficacy_thresholds=efficacy_thresholds,
        ess_null=stats_h0.average_sample_size,
        ess_alt=stats_h1.average_sample_size,
        cost=0.5 * stats_h0.average_sample_size + 0.5 * stats_h1.average_sample_size,
        non_binding_type_I_error=stats_h0_without_futility.disjunctive_power,
        type_I_error=stats_h0.disjunctive_power,
        power=stats_h1.disjunctive_power,
    )


def generate_tables_for_input_parameters(input_parameters: InputParameters):

    input_parameters.print_parameters()

    optimization_parameters = OptimizationParameters(
        algorithm="hsd",
        name="generate_tables",
        seed=1729,
        n_trials=1_000_000,
    )
    find_best_hsd_alpha_beta_spending(
        input_parameters=input_parameters,
        optimization_parameters=optimization_parameters,
        print_output=True,
        verbose=False,
    )
    print_alpha_beta_graphs(
        input_parameters=input_parameters,
        optimization_parameters=optimization_parameters,
    )
    optimization_parameters = OptimizationParameters(
        algorithm="general spending",
        name="generate_tables_general_spending",
        seed=1729,
        n_trials=1_000_000,
    )
    find_best_general_spending_function(
        input_parameters=input_parameters,
        optimization_parameters=optimization_parameters,
        verbose=False,
        print_output=True,
    )
    optimization_parameters = OptimizationParameters(
        algorithm="find best thresholds",
        name="generate_tables_best_thresholds",
        seed=1729,
        n_trials=1_000_000,
    )
    find_best_thresholds(
        input_parameters=input_parameters,
        optimization_parameters=optimization_parameters,
        verbose=False,
        print_output=True,
    )
    optimization_parameters = OptimizationParameters(
        algorithm="constant thresholds binding",
        name="generate_tables_constant_thresholds_binding",
        seed=1729,
        n_trials=1_000_000,
    )
    find_constant_thresholds_binding(
        input_parameters=input_parameters,
        optimization_parameters=optimization_parameters,
        print_output=True,
    )


def generate_tables_for_single_scenario():
    input_parameters = InputParameters(
        rate_per_arm_h0=np.array([0.5, 0.5, 0.5, 0.5]),
        rate_per_arm_h1=np.array([0.5, 0.7, 0.6, 0.55]),
        alpha=0.025,
        power=0.8,
        looks_fractions=np.array([0.3, 0.5, 0.7, 1.0]),
        sample_size_multiplier=1.15,
        is_binding=False,
    )
    generate_tables_for_input_parameters(input_parameters)


def run_scenario(
    input_parameters: InputParameters,
    filename: Optional[str],
    verbose: bool = False,
    base_seed=1729,
):
    optimization_parameters_list = (
        [
            OptimizationParameters(
                algorithm=AlgorithmName.FIND_BEST_HSD_SPENDING,
                name=f"0 find_best_hsd_spending_n_trials_{N_TRIALS1}",
                seed=i + base_seed,
                n_trials=N_TRIALS1,
            )
            for i in range(REPEATS1)
        ]
        + [
            OptimizationParameters(
                algorithm=AlgorithmName.FIND_BEST_GENERAL_SPENDING,
                name=f"1 find_best_general_spending_n_trials_{N_TRIALS1}",
                seed=i + base_seed,
                n_trials=N_TRIALS1,
            )
            for i in range(REPEATS1)
        ]
        + [
            OptimizationParameters(
                algorithm=AlgorithmName.FIND_BEST_THRESHOLDS,
                name=f"2 find_best_thresholds_n_trials_{N_TRIALS_BEST_THRESHOLDS}",
                seed=i + base_seed,
                n_trials=N_TRIALS_BEST_THRESHOLDS,
            )
            for i in range(REPEATS_BEST_THRESHOLDS)
        ]
        + [
            OptimizationParameters(
                algorithm=AlgorithmName.FIND_CONSTANT_THRESHOLDS,
                name=f"3 constant_thresholds_binding_n_trials_{N_TRIALS1}",
                seed=i + base_seed,
                n_trials=N_TRIALS1,
            )
            for i in range(REPEATS_BEST_THRESHOLDS)
        ]
    )
    optimization_runs = []
    for optimization_parameters in optimization_parameters_list:
        if optimization_parameters.algorithm == AlgorithmName.FIND_BEST_HSD_SPENDING:
            result = find_best_hsd_alpha_beta_spending(
                input_parameters=input_parameters,
                optimization_parameters=optimization_parameters,
                verbose=verbose,
                print_output=False,
            )
        elif (
            optimization_parameters.algorithm
            == AlgorithmName.FIND_BEST_GENERAL_SPENDING
        ):
            result = find_best_general_spending_function(
                input_parameters=input_parameters,
                optimization_parameters=optimization_parameters,
                verbose=verbose,
                print_output=False,
            )
        elif optimization_parameters.algorithm == AlgorithmName.FIND_BEST_THRESHOLDS:
            result = find_best_thresholds(
                input_parameters=input_parameters,
                optimization_parameters=optimization_parameters,
                verbose=verbose,
                print_output=False,
            )
        elif (
            optimization_parameters.algorithm == AlgorithmName.FIND_CONSTANT_THRESHOLDS
        ):
            result = find_constant_thresholds_binding(
                input_parameters=input_parameters,
                optimization_parameters=optimization_parameters,
                print_output=verbose,
            )
        else:
            raise ValueError(f"Unknown algorithm: {optimization_parameters.algorithm}")
        optimization_runs.append(
            OptimizationRun(
                input_parameters=input_parameters,
                optimization_parameters=optimization_parameters,
                result=result,
            )
        )
        print(f"\nFinished run {optimization_parameters.name}")
        print(f"Function evaluations = {result.nfev}")
        print(f"Time = {result.time:.2f} seconds")
        print(f"Average sample size under H0 = {result.ess_null:.2f}")
        print(f"Average sample size under H1 = {result.ess_alt:.2f}")
        print(f"Cost = {result.cost:.2f}")
        print(
            f"Type I error = {result.type_I_error:.4f} (non-binding = {result.non_binding_type_I_error:.4f})"
        )
        print(f"Power = {result.power:.4f}")

        if filename is not None:
            save_optimization_runs(optimization_runs, filename)
    return optimization_runs


def compare_multiple_scenarios():
    alpha = 0.025
    power = 0.8
    p0 = 0.5
    p3 = 0.55
    p2 = 0.6
    p1 = 0.7
    p4 = 0.8

    rng = np.random.default_rng(1729)

    input_parameters_list = []
    for n_arms in range(2, 5):
        for is_binding in [False, True]:
            for n_looks in range(6, 2, -1):
                for j in range(N_SCENARIOS_PER_OPTION):
                    if n_looks == 2:
                        looks_fractions = np.array([0.5, 1.0])
                    else:
                        looks_fractions = np.linspace(0.3, 1.0, n_looks)
                    rate_per_arm_h0 = np.full(n_arms, p0)

                    rate_per_arm_h1 = np.array([0.5])
                    rate_per_arm_h1 = np.append(
                        rate_per_arm_h1,
                        rng.choice([p1, p2, p3, p4], size=n_arms - 1, replace=True),
                    )
                    sample_size_multiplier = rng.uniform(1.05, 1.25)
                    print(
                        f"n_looks: {n_looks}, n_arms: {n_arms}, is_binding: {is_binding}"
                    )
                    print(f"looks_fractions: {looks_fractions}")
                    print(f"inflation factor: {sample_size_multiplier:.2f}")
                    print(f"h1 rates: {rate_per_arm_h1}")
                    input_parameters_list.append(
                        InputParameters(
                            rate_per_arm_h0=rate_per_arm_h0,
                            rate_per_arm_h1=rate_per_arm_h1,
                            alpha=alpha,
                            power=power,
                            looks_fractions=looks_fractions,
                            sample_size_multiplier=sample_size_multiplier,
                            is_binding=is_binding,
                            option_name=f"n_arms_{n_arms}_n_looks_{n_looks}_is_binding_{is_binding}",
                        )
                    )
                    print(f"fixed sample size: {input_parameters_list[-1].fixed_size}")

    for i, input_parameters in enumerate(input_parameters_list):
        print(
            f"Running scenario {i+1}/{len(input_parameters_list)}: {input_parameters.option_name}"
        )

        filename = f"scenario_{i+1}.json"
        run_scenario(
            input_parameters=input_parameters,
            filename=filename,
            verbose=False,
            base_seed=1729 + i * 1000,
        )


def analyse_scenario(filename: str):

    optimization_runs = load_optimization_runs(filename)
    input_parameters = optimization_runs[0].input_parameters
    if input_parameters.n_looks != 4:
        return
    print(f"\n\nAnalysing scenario {filename}")
    print(f"Input parameters: {input_parameters}")
    for run in optimization_runs:
        if (
            run.optimization_parameters.algorithm
            == AlgorithmName.FIND_CONSTANT_THRESHOLDS
        ):
            break
        print(f"\nOptimization: {run.optimization_parameters.name}")
        print(f"Algorithm: {run.optimization_parameters.algorithm}")
        print(f"n. function evaluations: {run.result.nfev}")
        # print(f"Result: {run.result}")
        if input_parameters.is_binding:
            print(f"Type I error = {run.result.type_I_error:.4f}")
        else:
            print(
                f"Type I error = {run.result.type_I_error:.4f} (non-binding = {run.result.non_binding_type_I_error:.4f})"
            )
        print(f"Power = {run.result.power:.4f}")
        # print(f"Average sample size under H0 = {run.result.ess_null:.2f}")
        # print(f"Average sample size under H1 = {run.result.ess_alt:.2f}")
        print(f"Cost = {run.result.cost:.2f}")


def analyse_scenarios():
    i = 0
    filename = f"scenario_{i+1}.json"
    # check if the file exists
    scenarios = []
    while os.path.exists(filename):
        scenario_runs = load_optimization_runs(filename)
        i += 1
        filename = f"scenario_{i+1}.json"
        scenarios.append(scenario_runs)

    for n_looks in [[3], [4], [5], [6]]:  # [[3, 4, 5, 6]]:  #
        for is_binding in [[False, True]]:  # #[[False], [True]]: #
            # for n_arms in range(2, 5):
            for n_arms in [[2, 3, 4]]:  # [[2],[3],[4]]:#
                # n_looks = [n_looks]
                relevant_runs = [
                    scenario
                    for scenario in scenarios
                    if scenario[0].input_parameters.n_looks in n_looks
                    and scenario[0].input_parameters.n_arms in n_arms
                    and scenario[0].input_parameters.is_binding in is_binding
                ]
                print(
                    f"\n\nAnalysing scenarios with n_looks_options={n_looks}, n_arms_options={n_arms}, is_binding_options={is_binding}"
                )
                hsd_spending_runs = [run[0] for run in relevant_runs]
                general_spending_runs = [run[1] for run in relevant_runs]
                thresholds_runs = [run[2] for run in relevant_runs]
                for runs, name in [
                    (hsd_spending_runs, "HSD_SPENDING"),
                    (general_spending_runs, "GENERAL_SPENDING"),
                    (thresholds_runs, "THRESHOLDS"),
                ]:
                    if len(runs) == 0:
                        print(f"No runs found for {name}")
                        continue
                    print(
                        f"\n\nAnalysing runs for {runs[0].optimization_parameters.name}"
                    )
                    print(f"number of runs = {len(runs)}")
                    print(
                        f"average function evaluations = {np.mean([run.result.nfev for run in runs]):.2f}, std = {np.std([run.result.nfev for run in runs]):.5f}"
                    )
                    print(
                        f"average cost relative to hsd = {np.mean([run.result.cost / hsd_spending_runs[i].result.cost for i, run in enumerate(runs)]):.5f}, std = {np.std([run.result.cost / hsd_spending_runs[i].result.cost for i, run in enumerate(runs)]):.5f}"
                    )
                    run_type_I_errors = []
                    for run in runs:
                        if run.input_parameters.is_binding:
                            run_type_I_errors.append(run.result.type_I_error)
                        else:
                            run_type_I_errors.append(
                                run.result.non_binding_type_I_error
                            )
                    print(
                        f"average type I error deviation = {np.mean([np.abs(type_I_error-runs[0].input_parameters.alpha) for type_I_error in run_type_I_errors]):.5f}, std = {np.std([np.abs(type_I_error-runs[0].input_parameters.alpha) for type_I_error in run_type_I_errors]):.5f}"
                    )
                    print(
                        f"average positive deviation of type I error = {np.mean([np.maximum(0, type_I_error-runs[0].input_parameters.alpha) for type_I_error in run_type_I_errors]):.5f}, std = {np.std([np.maximum(0, type_I_error-runs[0].input_parameters.alpha) for type_I_error in run_type_I_errors]):.5f}"
                    )
                    print(
                        f"avg power deviation: {np.mean([np.abs(run.result.power-run.input_parameters.power) for run in runs]):.5f}, std = {np.std([np.abs(run.result.power-run.input_parameters.power) for run in runs]):.5f}"
                    )
                    print(
                        f"avg positive power deviation: {np.mean([np.maximum(0, run.result.power-run.input_parameters.power) for run in runs]):.5f}, std = {np.std([np.maximum(0, run.result.power-run.input_parameters.power) for run in runs]):.5f}"
                    )


if __name__ == "__main__":

    # Uncomment the following line to run the Bayesian example of Section 4
    generate_tables_for_single_scenario()

    # Uncomment the following line to generate the data for Section 5:
    # compare_multiple_scenarios()

    # Uncomment the following line to analyse the scenarios generated in Section 5:
    # analyse_scenarios()
