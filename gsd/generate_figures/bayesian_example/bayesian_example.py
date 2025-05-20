# gsd code for article - bayesian multi arm
import os
import pickle
import time
from dataclasses import dataclass

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
from gsd.basic_code.utils.spending_function import (
    generate_spending_from_spending_parameter,
)

###################################################################################################
# Generate samples
###################################################################################################


def generate_samples(reload_samples=False):
    start = time.time()
    file_name = f"samples_h0_{n_trials}.pkl"
    if os.path.exists(file_name) and reload_samples:
        with open(file_name, "rb") as file:
            samples_h0 = pickle.load(file)
    else:
        samples_h0 = generate_bayesian_samples(
            n_samples_per_look_per_arm=n_samples_per_arm_per_look,
            rate_per_arm=arm_parameters_h0,
            n_trials=n_trials,
            rng=rng,
        )
        with open(file_name, "wb") as file:
            pickle.dump(samples_h0, file)

    file_name_h1 = f"samples_h1_{n_trials}.pkl"
    if os.path.exists(file_name_h1) and reload_samples:
        with open(file_name_h1, "rb") as file:
            samples_h1 = pickle.load(file)
    else:
        samples_h1 = generate_bayesian_samples(
            n_samples_per_look_per_arm=n_samples_per_arm_per_look,
            rate_per_arm=arm_parameters_h1,
            n_trials=n_trials,
            rng=rng,
        )
        with open(file_name_h1, "wb") as file:
            pickle.dump(samples_h1, file)
    print(f"Time to generate samples = {time.time()-start:.2f} seconds")
    return samples_h0, samples_h1


#####################################################################################################
# Find the best spending function of HSD-form, and generate the tables for the paper
#####################################################################################################
def find_best_hsd_alpha_beta_spending():
    start = time.time()
    (
        best_alpha_spending_parameter,
        best_beta_spending_parameter,
        differential_evolution_result,
    ) = find_best_hsd_spending(
        samples_h0,
        samples_h1,
        looks_fractions,
        n_samples_per_arm_per_look,
        alpha,
        beta,
        is_binding,
        seed=1729,
        verbose=verbose,
    )

    best_alpha_spending = generate_spending_from_spending_parameter(
        best_alpha_spending_parameter, alpha, looks_fractions
    )
    best_beta_spending = generate_spending_from_spending_parameter(
        best_beta_spending_parameter, beta, looks_fractions
    )
    best_efficacy_thresholds, best_futility_thresholds = (
        get_efficacy_futility_thresholds(
            samples_h0,
            samples_h1,
            best_alpha_spending,
            best_beta_spending,
            is_binding=is_binding,
        )
    )
    best_stats_h0 = get_statistics_given_thresholds(
        samples_h0,
        efficacy_thresholds=best_efficacy_thresholds,
        futility_thresholds=best_futility_thresholds,
        n_samples_per_look_per_arm=n_samples_per_arm_per_look,
    )
    best_stats_h1 = get_statistics_given_thresholds(
        samples_h1,
        efficacy_thresholds=best_efficacy_thresholds,
        futility_thresholds=best_futility_thresholds,
        n_samples_per_look_per_arm=n_samples_per_arm_per_look,
    )
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
        f"best_stats_h0.average_sample_size = {best_stats_h0.average_sample_size} (Relative to fixed: {best_stats_h0.average_sample_size/fixed_size})"
    )
    print(
        f"best_stats_h1.average_sample_size = {best_stats_h1.average_sample_size} (Relative to fixed: {best_stats_h1.average_sample_size/fixed_size})"
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


################################################################################################
# Print the graphs for changing alpha for the paper
################################################################################################
@dataclass
class ParametersStatistics:
    alpha_spending_parameter: float
    beta_spending_parameter: float
    stats_h0: GSDStatistics
    stats_h1: GSDStatistics


def print_alpha_beta_graphs():

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
            looks_fractions=looks_fractions,
            n_samples_per_arm_per_look=n_samples_per_arm_per_look,
            alpha=alpha,
            beta=beta,
            is_binding=is_binding,
        )

        alpha_spending = generate_spending_from_spending_parameter(
            alpha_spending_parameter, alpha, looks_fractions
        )
        beta_spending = generate_spending_from_spending_parameter(
            beta_spending_parameter, beta, looks_fractions
        )
        # alpha_spending = np.zeros_like(alpha_spending)
        # alpha_spending[-1] = alpha
        # beta_spending = np.zeros_like(beta_spending)
        # beta_spending[-1] = beta
        efficacy_thresholds, futility_thresholds = get_efficacy_futility_thresholds(
            samples_h0, samples_h1, alpha_spending, beta_spending, is_binding=is_binding
        )
        stats_h1 = get_statistics_given_thresholds(
            samples_h1,
            efficacy_thresholds,
            futility_thresholds,
            n_samples_per_look_per_arm=n_samples_per_arm_per_look,
        )
        stats_h0 = get_statistics_given_thresholds(
            samples_h0,
            efficacy_thresholds,
            futility_thresholds,
            n_samples_per_look_per_arm=n_samples_per_arm_per_look,
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
    max_sample_size = n_samples_per_look_total[-1]

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
    ax1.axhline(y=fixed_size, color="r", linestyle="--", label="Fixed Size")
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


################################################################################################
# Find the constant thresholds
################################################################################################


def find_constant_thresholds_binding():
    n_iterations = 25
    efficacy_threshold_lower = 0.9
    efficacy_threshold_upper = 1
    for i in range(n_iterations):
        mid_value = (efficacy_threshold_lower + efficacy_threshold_upper) / 2
        efficacy_thresholds_constant = np.array([mid_value] * len(looks_fractions))
        futility_thresholds_constant = np.array([0.0] * len(looks_fractions))
        stats_h0 = get_statistics_given_thresholds(
            samples_h0,
            efficacy_thresholds_constant,
            futility_thresholds_constant,
            n_samples_per_look_per_arm=n_samples_per_arm_per_look,
        )
        type_I_error = np.sum(stats_h0.efficacy_probs_trial_per_look)
        if type_I_error > alpha:
            efficacy_threshold_lower = mid_value
        else:
            efficacy_threshold_upper = mid_value
    efficacy_thresholds_constant = np.array(
        [efficacy_threshold_upper] * len(looks_fractions)
    )
    futility_threshold_lower = 0
    futility_threshold_upper = 0.9
    for i in range(n_iterations):
        mid_value = (futility_threshold_lower + futility_threshold_upper) / 2
        futility_thresholds_constant = np.array([mid_value] * len(looks_fractions))
        stats_h1 = get_statistics_given_thresholds(
            samples_h1,
            efficacy_thresholds_constant,
            futility_thresholds_constant,
            n_samples_per_look_per_arm=n_samples_per_arm_per_look,
        )
        power = np.sum(stats_h1.efficacy_probs_trial_per_look)
        if power > 1 - beta:
            futility_threshold_lower = mid_value
        else:
            futility_threshold_upper = mid_value
    stats_h0 = get_statistics_given_thresholds(
        samples_h0,
        efficacy_thresholds_constant,
        futility_thresholds_constant,
        n_samples_per_look_per_arm=n_samples_per_arm_per_look,
    )
    stats_h1 = get_statistics_given_thresholds(
        samples_h1,
        efficacy_thresholds_constant,
        futility_thresholds_constant,
        n_samples_per_look_per_arm=n_samples_per_arm_per_look,
    )
    print("===============================")
    print("Constant bounds:")
    print("===============================")
    print(f"efficacy_thresholds_constant = {efficacy_thresholds_constant}")
    print(f"futility_thresholds_constant = {futility_thresholds_constant}")
    print(f"actual type I error = {np.sum(stats_h0.efficacy_probs_trial_per_look)}")
    print(f"efficacy_probs_trial_per_look = {stats_h0.efficacy_probs_trial_per_look}")
    print(f"actual power = {np.sum(stats_h1.efficacy_probs_trial_per_look)}")
    print(f"efficacy_probs_trial_per_look = {stats_h1.efficacy_probs_trial_per_look}")
    print(f"average_sample_size_h0 = {stats_h0.average_sample_size}")
    print(f"average_sample_size_h1 = {stats_h1.average_sample_size}")


#################################################################################################
# Search for the best spending function, without limiting to hsw spending functions
#################################################################################################


def find_best_general_spending_function():
    start = time.time()
    alpha_spending, beta_spending, differential_evolution_result = (
        find_best_general_spending(
            samples_h0=samples_h0,
            samples_h1=samples_h1,
            looks_fractions=looks_fractions,
            n_samples_per_arm_per_look=n_samples_per_arm_per_look,
            alpha=alpha,
            beta=beta,
            is_binding=is_binding,
            seed=1729,
            verbose=verbose,
        )
    )
    (
        efficacy_thresholds_differential_evolution,
        futility_thresholds_differential_evolution,
    ) = get_efficacy_futility_thresholds(
        samples_h0,
        samples_h1,
        alpha_spending,
        beta_spending,
        is_binding=is_binding,
    )
    stats_h0 = get_statistics_given_thresholds(
        samples_h0,
        efficacy_thresholds_differential_evolution,
        futility_thresholds_differential_evolution,
        n_samples_per_look_per_arm=n_samples_per_arm_per_look,
    )
    stats_h1 = get_statistics_given_thresholds(
        samples_h1,
        efficacy_thresholds_differential_evolution,
        futility_thresholds_differential_evolution,
        n_samples_per_look_per_arm=n_samples_per_arm_per_look,
    )

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


################################################################################################
# Find best thresholds directly
################################################################################################

# search for the best thresholds directly
function_evaluations = 0


def objective_function_thresholds(input: np.ndarray):
    global function_evaluations
    function_evaluations += 1
    assert input.shape[0] == 2 * len(looks_fractions) - 1

    logit_futility_thresholds = input[: len(looks_fractions)]
    diffs = input[len(looks_fractions) :]
    logit_efficacy_thresholds = logit_futility_thresholds + np.concatenate((diffs, [0]))
    futility_thresholds = sp.special.expit(logit_futility_thresholds)
    efficacy_thresholds = sp.special.expit(logit_efficacy_thresholds)
    if is_binding:
        futility_thresholds_for_type_I_error = futility_thresholds
    else:
        futility_thresholds_for_type_I_error = np.zeros(len(looks_fractions))

    stats_h0_for_alpha = get_statistics_given_thresholds(
        samples_h0,
        efficacy_thresholds,
        futility_thresholds_for_type_I_error,
        n_samples_per_look_per_arm=n_samples_per_arm_per_look,
    )
    type_I_error = np.sum(stats_h0_for_alpha.efficacy_probs_trial_per_look)
    if verbose:
        print(
            f"evaluations= {function_evaluations}, input = {input}, futility_thresholds = {futility_thresholds}, efficacy_thresholds = {efficacy_thresholds}, type_I_error = {type_I_error}"
        )
    if type_I_error > alpha:
        if verbose:
            print(f"type I error = {type_I_error}")
        return 10000 * (3 - type_I_error)

    stats_h1 = get_statistics_given_thresholds(
        samples_h1,
        efficacy_thresholds,
        futility_thresholds,
        n_samples_per_look_per_arm=n_samples_per_arm_per_look,
    )
    power = np.sum(stats_h1.efficacy_probs_trial_per_look)
    if power < 1 - beta:
        if verbose:
            print(f"type I error = {type_I_error},power = {power}")
        return 10000 * (2 - power)
    stats_h0 = get_statistics_given_thresholds(
        samples_h0,
        efficacy_thresholds,
        futility_thresholds,
        n_samples_per_look_per_arm=n_samples_per_arm_per_look,
    )
    if verbose:
        print(
            f"type I error = {type_I_error}, power = {power}, average sample size = {stats_h1.average_sample_size + stats_h0.average_sample_size}"
        )
    return stats_h1.average_sample_size + stats_h0.average_sample_size


def find_best_thresholds(samples_h0, samples_h1):
    max_prob = 0.999  # for efficacy
    min_prob = 0.1  # for futility
    min_logit = float(np.log(min_prob / (1 - min_prob)))
    max_logit = float(np.log(max_prob / (1 - max_prob)))
    bounds = [(min_logit, max_logit)] * len(looks_fractions) + [
        (0.0, max_logit - min_logit)
    ] * (len(looks_fractions) - 1)
    bounds = np.array(bounds)
    start = time.time()
    differential_evolution_result = differential_evolution(
        objective_function_thresholds, bounds
    )
    input = differential_evolution_result.x
    logit_futility_thresholds = input[: len(looks_fractions)]
    diffs = input[len(looks_fractions) :]
    logit_efficacy_thresholds = logit_futility_thresholds + np.concatenate((diffs, [0]))
    futility_thresholds = sp.special.expit(logit_futility_thresholds)
    efficacy_thresholds = sp.special.expit(logit_efficacy_thresholds)
    stats_h0 = get_statistics_given_thresholds(
        samples_h0,
        efficacy_thresholds,
        futility_thresholds,
        n_samples_per_look_per_arm=n_samples_per_arm_per_look,
    )
    stats_h1 = get_statistics_given_thresholds(
        samples_h1,
        efficacy_thresholds,
        futility_thresholds,
        n_samples_per_look_per_arm=n_samples_per_arm_per_look,
    )
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


alpha = 0.025
beta = 0.2
power = 1 - beta

p0 = 0.5
p3 = 0.55
p2 = 0.6
p1 = 0.7

arm_parameters_h1 = np.array([[p0], [p1], [p2], [p3]])
arm_parameters_h0 = np.array([[p0], [p0], [p0], [p0]])
n_arms = len(arm_parameters_h1)

rng = np.random.default_rng(seed=1729)

fixed_size = 468
n_trials = 1000000  # should be 1000000, but can use smaller number for testing
sample_size_multiplier = 1.15

is_binding = False
n_samples = int(fixed_size * sample_size_multiplier) // n_arms * n_arms


looks_fractions = np.array([0.3, 0.5, 0.7, 1])

n_samples_per_look_total = (n_samples * looks_fractions).astype(int) // n_arms * n_arms
n_samples_per_arm_per_look = n_samples_per_look_total // n_arms
max_n = int(np.max(n_samples_per_arm_per_look))

print(f"arm parameters = {arm_parameters_h1}")
print(f"fixed_size = {fixed_size}")
print(f"fixed_size per arm = {fixed_size // n_arms}")
print(f"sample size multiplier = {sample_size_multiplier}")
print(f"look_fractions = {looks_fractions}")
print(f"n_samples_per_look_total = {n_samples_per_look_total}")
print(f"n_samples_per_look_per_arm = {n_samples_per_look_total // n_arms}")
print(f"Is binding = {is_binding}")

verbose = False

samples_h0, samples_h1 = generate_samples()


find_best_hsd_alpha_beta_spending()
print_alpha_beta_graphs()
find_constant_thresholds_binding()
find_best_general_spending_function()
find_best_thresholds(samples_h0, samples_h1)

# Non-Binding Output:
# arm parameters = [[0.5 ]
#  [0.7 ]
#  [0.6 ]
#  [0.55]]
# fixed_size = 468
# fixed_size per arm = 117
# sample size multiplier = 1.15
# look_fractions = [0.3 0.5 0.7 1. ]
# n_samples_per_look_total = [160 268 372 536]
# n_samples_per_look_per_arm = [ 40  67  93 134]
# Time to generate samples = 71.33 seconds
# ========================================
# Best alpha and beta spending parameters:
# ========================================
# Time to find hsd spending function = 331.26 seconds
# number of function evaluations =  453
# best_alpha_spending_parameter =  -1.3278071760852683
# best_beta_spending_parameter =  -1.2821354132893799
# best_alpha_spending_parameter = -1.3278071760852683
# best_beta_spending_parameter = -1.2821354132893799
# best_alpha_spending = [0.00441215 0.00408443 0.00532676 0.01117665]
# best_beta_spending = [0.03602359 0.03297745 0.04261691 0.08838204]
# best_efficacy_thresholds = [0.99882722 0.99787663 0.9976659  0.99332793]
# best_futility_thresholds = [0.59183833 0.80867878 0.93130259 0.99332793]
# best_alpha_spending_parameter = -1.3278071760852683. best_beta_spending_parameter = -1.2821354132893799
# best_stats_h0.average_sample_size = 275.175363 (Relative to fixed: 0.5879815448717949)
# best_stats_h1.average_sample_size = 342.16691999999995 (Relative to fixed: 0.7311258974358973)
# ===============================
# Constant bounds:
# ===============================
# efficacy_thresholds_constant = [0.99727044 0.99727044 0.99727044 0.99727044]
# futility_thresholds_constant = [2.68220901e-08 2.68220901e-08 2.68220901e-08 2.68220901e-08]
# actual type I error = 0.024411
# efficacy_probs_trial_per_look = [0.008268 0.007168 0.004745 0.00423 ]
# actual power = 0.77213
# efficacy_probs_trial_per_look = [0.21048  0.200476 0.167988 0.193186]
# average_sample_size_h0 = 570.192028
# average_sample_size_h1 = 415.5819200000001
# ===============================
# Best spending functions found:
# ===============================
# Time to find spending function = 2041.04 seconds
# number of function evaluations = 3157
# alpha_spending_differential_evolution = [0.00197222 0.00582137 0.00651243 0.01069398]
# beta_spending_differential_evolution = [0.04792871 0.03165346 0.0251041  0.09531374]
# efficacy_thresholds_differential_evolution = [0.9995089  0.99771612 0.99633448 0.99340519]
# futility_thresholds_differential_evolution = [0.67377698 0.81003449 0.90814002 0.99340519]
# actual type I error = 0.022057
# actual power = 0.8001670000000001
# average_sample_size_h0 = 267.24591599999997
# average_sample_size_h1 = 337.086801
# ===========================
# Best thresholds found:
# ===========================
# Time to find thresholds = 8215.98 seconds
# differential evolution function evaluations = 17753
# efficacy_thresholds = [0.99889331 0.9976776  0.99782773 0.99328937]
# futility_thresholds = [0.67267502 0.80145916 0.90954739 0.99328937]
# actual type I error = 0.022149000000000002
# actual power = 0.80034
# average_sample_size_h0 = 271.02232000000004
# average_sample_size_h1 = 337.6241730000001

# Binding Output:
# arm parameters = [[0.5 ]
#  [0.7 ]
#  [0.6 ]
#  [0.55]]
# fixed_size = 468
# fixed_size per arm = 117
# sample size multiplier = 1.15
# look_fractions = [0.3 0.5 0.7 1. ]
# n_samples_per_look_total = [160 268 372 536]
# n_samples_per_look_per_arm = [ 40  67  93 134]
# Is binding = True
# Time to generate samples = 68.21 seconds
# ========================================
# Best alpha and beta spending parameters:
# ========================================
# Time to find hsd spending function = 508.34 seconds
# number of function evaluations =  693
# best_alpha_spending_parameter =  -3.0495427906498493
# best_beta_spending_parameter =  0.7047530053272499
# best_alpha_spending_parameter = -3.0495427906498493
# best_beta_spending_parameter = 0.7047530053272499
# best_alpha_spending = [0.00186071 0.00260829 0.00479994 0.01573106]
# best_beta_spending = [0.07535875 0.04208    0.03654783 0.04601341]
# best_efficacy_thresholds = [0.9995089  0.99930741 0.99770385 0.98699327]
# best_futility_thresholds = [0.74970676 0.85354892 0.93284394 0.98699327]
# best_alpha_spending_parameter = -3.0495427906498493. best_beta_spending_parameter = 0.7047530053272499
# best_stats_h0.average_sample_size = 251.711965 (Relative to fixed: 0.5378460790598291)
# best_stats_h1.average_sample_size = 336.67780100000004 (Relative to fixed: 0.7193970106837608)
# ===============================
# Constant bounds:
# ===============================
# efficacy_thresholds_constant = [0.99727044 0.99727044 0.99727044 0.99727044]
# futility_thresholds_constant = [2.68220901e-08 2.68220901e-08 2.68220901e-08 2.68220901e-08]
# actual type I error = 0.024411
# efficacy_probs_trial_per_look = [0.008268 0.007168 0.004745 0.00423 ]
# actual power = 0.77213
# efficacy_probs_trial_per_look = [0.21048  0.200476 0.167988 0.193186]
# average_sample_size_h0 = 570.192028
# average_sample_size_h1 = 415.5819200000001
# ===============================
# Best spending functions found:
# ===============================
# Time to find spending function = 6060.97 seconds
# number of function evaluations = 4147
# alpha_spending_differential_evolution = [0.00288709 0.00564093 0.00573721 0.01073478]
# beta_spending_differential_evolution = [0.04786609 0.07286729 0.01759404 0.06167258]
# efficacy_thresholds_differential_evolution = [0.99948629 0.99766113 0.99638859 0.99052228]
# futility_thresholds_differential_evolution = [0.67377698 0.88979895 0.90878878 0.99052228]
# actual type I error = 0.024901
# actual power = 0.800741
# average_sample_size_h0 = 258.57577000000003
# average_sample_size_h1 = 328.338384
# ===========================
# Best thresholds found:
# ===========================
# Time to find thresholds = 59715.67 seconds
# differential evolution function evaluations = 23423
# efficacy_thresholds = [0.99958638 0.99803038 0.99663432 0.99056285]
# futility_thresholds = [0.74799867 0.78690785 0.92667282 0.99056285]
# actual type I error = 0.02446
# actual power = 0.801138
# average_sample_size_h0 = 261.882071
# average_sample_size_h1 = 334.6357349999999