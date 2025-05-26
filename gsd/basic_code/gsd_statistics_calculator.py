from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class GSDStatistics:
    efficacy_probs_per_arm_per_look: npt.NDArray[
        np.float64
    ]  # 2D array, of shape (n_treatment_arms, n_looks)
    # probabilities of stopping for efficacy for each treatment arm and look.

    futility_probs_per_arm_per_look: npt.NDArray[
        np.float64
    ]  # 2D array, of shape (n_treatment_arms, n_looks)
    # probabilities of stopping for futility for each treatment arm and look.

    efficacy_probs_trial_per_look: npt.NDArray[
        np.float64
    ]  # 1D array, of shape (n_looks)
    # probabilities of stopping the trial for efficacy (of some arm), for each look.

    futility_probs_trial_per_look: npt.NDArray[
        np.float64
    ]  # 1D array, of shape (n_looks)
    # probabilities of stopping the trial for futility (of all arms), for each look.

    average_sample_size: float  # the average sample size of the trial

    def __post_init__(self):
        # Calculate the disjunctive power, which is the sum of the efficacy probabilities across all looks
        # i.e, the total probability of stopping for efficacy.
        self.disjunctive_power = np.sum(self.efficacy_probs_trial_per_look)


def get_statistics_given_thresholds(
    samples_statistics: npt.NDArray[np.float64],
    efficacy_thresholds: npt.NDArray[np.float64],
    futility_thresholds: npt.NDArray[np.float64],
    n_samples_per_look_per_arm: npt.NDArray[np.integer],
) -> GSDStatistics:
    """
    Return various statistics of the trial, given the efficacy and futility thresholds.
    parameters:
    samples_statistics: 3D array, of shape (n_trials, n_treatment_arms, n_looks).
    efficacy_thresholds: 1D array, of shape (n_looks).
    futility_thresholds: 1D array, of shape (n_looks).
    n_samples_per_look_per_arm: 1D array, of shape (n_looks).
    Returns: a GSDStatistics object, with various statistics of the trial.
    """
    if samples_statistics.ndim == 2:
        samples_statistics = samples_statistics[:, np.newaxis, :]

    n_trials, n_treatment_arms, n_looks = samples_statistics.shape
    n_arms = n_treatment_arms + 1

    assert futility_thresholds.shape[0] == n_looks
    assert efficacy_thresholds.shape[0] == n_looks
    assert n_samples_per_look_per_arm.shape[0] == n_looks

    stopped_arm_for_efficacy_at_look = np.zeros(
        (n_trials, n_treatment_arms, n_looks), dtype=bool
    )
    stopped_arm_for_futility_at_look = np.zeros(
        (n_trials, n_treatment_arms, n_looks), dtype=bool
    )
    stopped_for_efficacy_at_look = np.zeros((n_trials, n_looks), dtype=bool)
    stopped_for_futility_at_look = np.zeros((n_trials, n_looks), dtype=bool)
    stopped_arm_so_far = np.zeros((n_trials, n_arms - 1), dtype=bool)
    stopped_trial_so_far = np.zeros((n_trials), dtype=bool)

    for j in range(n_looks):
        for i in range(n_treatment_arms):
            stopped_arm_for_efficacy_at_look[:, i, j] = (
                samples_statistics[:, i, j] > efficacy_thresholds[j]
            ) & ~stopped_arm_so_far[:, i]
            stopped_arm_so_far[:, i] = (
                stopped_arm_so_far[:, i] | stopped_arm_for_efficacy_at_look[:, i, j]
            )
            stopped_arm_for_futility_at_look[:, i, j] = (
                samples_statistics[:, i, j] < futility_thresholds[j]
            ) & ~stopped_arm_so_far[:, i]
            if j == n_looks - 1:
                stopped_arm_for_futility_at_look[:, i, j] = (
                    stopped_arm_for_futility_at_look[:, i, j]
                    | ~stopped_arm_so_far[:, i]
                )
            stopped_arm_so_far[:, i] = (
                stopped_arm_so_far[:, i] | stopped_arm_for_futility_at_look[:, i, j]
            )

        # trial stopped if all arms stopped
        stopped_for_efficacy_at_look[:, j] = np.any(
            stopped_arm_for_efficacy_at_look[:, :, j], axis=1
        )

        stopped_trial_so_far = stopped_trial_so_far | stopped_for_efficacy_at_look[:, j]
        stopped_for_futility_at_look[:, j] = (
            np.all(stopped_arm_so_far, axis=1) & ~stopped_trial_so_far
        )
        stopped_trial_so_far = stopped_trial_so_far | stopped_for_futility_at_look[:, j]

        for i in range(n_treatment_arms):
            stopped_arm_for_futility_at_look[:, i, j] = np.where(
                stopped_arm_so_far[:, i],
                stopped_arm_for_futility_at_look[:, i, j],
                stopped_trial_so_far,
            )
            stopped_arm_so_far[:, i] = stopped_arm_so_far[:, i] | stopped_trial_so_far

    efficacy_probs_per_arm_per_look = np.mean(
        stopped_arm_for_efficacy_at_look, axis=0
    )  # shape (n_treatment_arms, n_looks)
    futility_probs_per_arm_per_look = np.mean(
        stopped_arm_for_futility_at_look, axis=0
    )  # shape (n_treatment_arms, n_looks)
    efficacy_probs = np.mean(stopped_for_efficacy_at_look, axis=0)  # shape (n_looks)
    futility_probs = np.mean(stopped_for_futility_at_look, axis=0)  # shape (n_looks)
    average_sample_size = 0
    for j in range(n_looks):
        # This is the average sample size for the control arm
        average_sample_size += n_samples_per_look_per_arm[j] * (
            efficacy_probs[j] + futility_probs[j]
        )
        # This is the average sample size for the treatment arms
        for i in range(n_treatment_arms):
            average_sample_size += n_samples_per_look_per_arm[j] * (
                efficacy_probs_per_arm_per_look[i, j]
                + futility_probs_per_arm_per_look[i, j]
            )

    return GSDStatistics(
        efficacy_probs_per_arm_per_look=efficacy_probs_per_arm_per_look,
        futility_probs_per_arm_per_look=futility_probs_per_arm_per_look,
        efficacy_probs_trial_per_look=efficacy_probs,
        futility_probs_trial_per_look=futility_probs,
        average_sample_size=average_sample_size,
    )
