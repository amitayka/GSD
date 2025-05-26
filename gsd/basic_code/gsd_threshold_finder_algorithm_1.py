import numpy as np
import numpy.typing as npt

EPSILON = 1e-10


def get_kth_value_by_size(
    array: npt.NDArray[np.float64], k: int
) -> npt.NDArray[np.float64]:
    """
    Get the k-th value by size in a 1D array.
    array: 1D array of shape (n_samples,).
    k: int, the index of the value to return.
    Returns: the k-th value by size in the array.
    """
    return np.partition(array, k)[k]


def check_samples_statistics(
    samples_statistics_h0: npt.NDArray[np.float64],
    samples_statistics_h1: npt.NDArray[np.float64],
) -> None:
    """
    Check that the samples statistics are valid.
    """
    assert samples_statistics_h0.ndim == 3
    assert samples_statistics_h1.ndim == 3
    assert samples_statistics_h0.shape[0] == samples_statistics_h1.shape[0]
    assert samples_statistics_h0.shape[1] == samples_statistics_h1.shape[1]
    assert samples_statistics_h0.shape[2] == samples_statistics_h1.shape[2]


def get_efficacy_futility_thresholds(
    samples_statistics_h0: npt.NDArray[np.float64],
    samples_statistics_h1: npt.NDArray[np.float64],
    alpha_spending: npt.NDArray[np.float64],
    beta_spending: npt.NDArray[np.float64],
    is_binding: bool = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """get the efficacy and futility thresholds using Monte Carlo simulation, for multiarm trials.
    parameters:
    samples_statistics_h0: 3D array, of shape (n_trials, n_treatment_arms, n_looks).
        It is the samples statistics under the null hypothesis.
    samples_statistics_h1: 3D array, of shape (n_trials, n_treatment_arms, n_looks).
        It is the samples statistics under the alternative hypothesis.
    alpha_spending: 1D array, of shape (n_looks). It is the percentages of samples that are needed to be
        removed from the top in each look.
    beta_spending: 1D array, of shape (n_looks). It is the percentages of samples that are needed to be
        removed from the bottom in each look.
    is_binding: if True, the futility is binding, meaning that if the futility threshold is crossed, the trial stops.
        If the trial is non-binding, even if the futility thresholds are not held, we guarantee the type I error.
        This results in lower power.
    Returns:
    1D array, of shape (n_looks)- the efficacy thresholds.
    1D array, of shape (n_looks)- the futility thresholds.
    """
    if is_binding:
        assert beta_spending is not None
        return _get_efficacy_futility_thresholds_binding(
            samples_statistics_h0, samples_statistics_h1, alpha_spending, beta_spending
        )
    # In the non-binding case, we first find the efficacy thresholds, and then the futility thresholds.
    efficacy_thresholds = _get_efficacy_thresholds(
        samples_statistics_h0, alpha_spending
    )

    futility_thresholds = _get_futility_thresholds_given_efficacy_thresholds(
        samples_statistics_h1, beta_spending, efficacy_thresholds
    )

    return efficacy_thresholds, futility_thresholds


def _get_efficacy_thresholds(
    samples_statistics_h0: npt.NDArray[np.float64],
    spending_function: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Get the efficacy thresholds using Monte Carlo simulation, for multiarm trials.
    parameters:
    samples_statistics_h0: 3D array, of shape (n_trials, n_treatment_arms, n_looks).
        It is the samples statistics under the null hypothesis.
    spending_function: 1D array, of shape (n_looks). It is the percentages of samples that are needed to be
        removed from the top in each look.
    Returns:
    1D array, of shape (n_looks)- the efficacy thresholds.
    """
    n_trials, _, n_looks = samples_statistics_h0.shape
    assert len(spending_function) == n_looks
    samples_statistics = np.max(samples_statistics_h0, axis=1)
    thresholds = np.full(n_looks, np.inf)
    surviving_trials = np.full(n_trials, True, dtype=bool)
    n_surviving_trials = np.sum(surviving_trials)

    # If we didn't spend all the alpha, we can use it in the next look.
    # This is used mainly in the discrete case.
    extra_spending = 0.0

    for i in range(0, n_looks):
        cur_spending = spending_function[i] + extra_spending

        if cur_spending == 0:
            thresholds[i] = np.inf
            continue
        m = np.floor(n_trials * cur_spending).astype(int)
        cur_look_samples = np.where(surviving_trials, samples_statistics[:, i], -np.inf)

        thresholds[i] = get_kth_value_by_size(cur_look_samples, n_trials - m - 1)

        surviving_trials = surviving_trials & (cur_look_samples <= thresholds[i])
        n_surviving_trials_tmp = np.sum(surviving_trials)
        extra_spending = max(
            cur_spending - (n_surviving_trials - n_surviving_trials_tmp) / n_trials,
            0.0,
        )
        n_surviving_trials = n_surviving_trials_tmp

    return thresholds


def _get_futility_thresholds_given_efficacy_thresholds(
    samples_statistics_h1: npt.NDArray[np.float64],
    spending_function: npt.NDArray[np.float64],
    efficacy_thresholds: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Get the futility thresholds using Monte Carlo simulation, for multiarm trials.
    We assume that the futility is non-binding, and that the efficacy thresholds were found before.
    parameters:
    samples_statistics_h1: 3D array, of shape (n_trials, n_treatment_arms, n_looks).
        It is the samples statistics under the alternative hypothesis.
    spending_function: 1D array, of shape (n_looks). It is the percentages of samples that are needed to be
        removed from the bottom in each look.
    efficacy_thresholds: 1D array, of shape (n_looks). It is the efficacy thresholds that were found before.
    Returns:
    1D array, of shape (n_looks)- the futility thresholds.
    """
    n_trials, n_treatment_arms, n_looks = samples_statistics_h1.shape
    thresholds = np.full(n_looks, -np.inf)

    survivors = np.full((n_trials, n_treatment_arms), True, dtype=bool)
    surviving_trials = np.full(n_trials, True, dtype=bool)
    n_surviving_trials = np.sum(surviving_trials)
    extra_spending = 0.0
    for i in range(0, n_looks):
        cur_samples = np.where(survivors, samples_statistics_h1[:, :, i], -np.inf)
        max_cur_samples = np.max(cur_samples, axis=1)
        surviving_trials = surviving_trials & (
            max_cur_samples <= efficacy_thresholds[i]
        )
        survivors = survivors & surviving_trials[:, np.newaxis]
        n_surviving_trials = np.sum(surviving_trials)

        cur_spending = spending_function[i] + extra_spending
        if cur_spending == 0:
            thresholds[i] = -np.inf
            continue
        m = np.floor(n_trials * cur_spending).astype(int)
        if m >= n_surviving_trials:
            # If m is too large, we finish in this look.
            thresholds[i] = efficacy_thresholds[i] + EPSILON
            return thresholds
        cur_samples = np.where(survivors, samples_statistics_h1[:, :, i], -np.inf)
        max_cur_samples = np.max(cur_samples, axis=1)

        thresholds[i] = get_kth_value_by_size(
            max_cur_samples, n_trials - n_surviving_trials + m
        )
        survivors = survivors & (samples_statistics_h1[:, :, i] >= thresholds[i])

        surviving_trials = np.any(survivors, axis=1)
        n_surviving_trials_tmp = np.sum(surviving_trials)
        extra_spending = max(
            cur_spending - (n_surviving_trials - n_surviving_trials_tmp) / n_trials,
            0.0,
        )
        n_surviving_trials = n_surviving_trials_tmp

    thresholds[-1] = efficacy_thresholds[-1]
    return thresholds


def _get_efficacy_futility_thresholds_binding(
    samples_statistics_h0: npt.NDArray[np.float64],
    samples_statistics_h1: npt.NDArray[np.float64],
    alpha_spending: npt.NDArray[np.float64],
    beta_spending: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    get the efficacy and futility thresholds using Monte Carlo simulation.
    We assume that futility is binding, meaning that if the futility threshold is crossed, the trial most stop.
    This requires a different and slightly more complicated algorithm that computes both types of threshold together.
    The inputs and outputs are the same as get_efficacy_futility_thresholds.
    """
    n_trials, n_treatment_arms, n_looks = samples_statistics_h0.shape

    n_surviving_trials_h0 = n_trials
    n_surviving_trials_h1 = n_trials
    efficacy_thresholds = np.full(n_looks, np.inf)
    futility_thresholds = np.full(n_looks, -np.inf)
    survivors_h0 = np.full((n_trials, n_treatment_arms), True, dtype=bool)
    surviving_trials_h0 = np.full(n_trials, True, dtype=bool)
    survivors_h1 = np.full((n_trials, n_treatment_arms), True, dtype=bool)
    surviving_trials_h1 = np.full(n_trials, True, dtype=bool)

    # If we didn't spend all the alpha, we can use it in the next look. This is used
    # mainly in the discrete case.
    extra_alpha_spending = 0.0

    for i in range(0, n_looks):
        # if the alpha spending is 0, we set the threshold to -inf
        alpha_spending_cur = alpha_spending[i] + extra_alpha_spending
        if alpha_spending_cur == 0:
            efficacy_thresholds[i] = np.inf
        else:
            m = np.floor(n_trials * alpha_spending_cur).astype(int)
            if m >= n_surviving_trials_h0:
                # This means that everything that was not futile at the last look is efficacious.
                # So we change the efficacy threshold of the last look and return.
                # We put the threshold to be a bit smaller than the futility threshold,
                # since we always work with strict inequalities.
                efficacy_thresholds[i - 1] = futility_thresholds[i - 1] - EPSILON
                return efficacy_thresholds, futility_thresholds
            cur_samples_h0 = np.where(
                survivors_h0, samples_statistics_h0[:, :, i], -np.inf
            )
            max_cur_samples_h0 = np.max(cur_samples_h0, axis=1)
            efficacy_thresholds[i] = get_kth_value_by_size(
                max_cur_samples_h0, n_trials - m - 1
            )
            surviving_trials_h0 = surviving_trials_h0 & (
                max_cur_samples_h0 <= efficacy_thresholds[i]
            )
            survivors_h0 = survivors_h0 & surviving_trials_h0[:, np.newaxis]

            cur_samples_h1 = np.where(
                survivors_h1, samples_statistics_h1[:, :, i], -np.inf
            )
            max_cur_samples_h1 = np.max(cur_samples_h1, axis=1)
            surviving_trials_h1 = surviving_trials_h1 & (
                max_cur_samples_h1 <= efficacy_thresholds[i]
            )
            survivors_h1 = survivors_h1 & surviving_trials_h1[:, np.newaxis]
            n_surviving_trials_h1 = np.sum(surviving_trials_h1)

            n_surviving_trials_h0_new = np.sum(surviving_trials_h0)
            actual_alpha_spent = (
                n_surviving_trials_h0 - n_surviving_trials_h0_new
            ) / n_trials

            extra_alpha_spending = max(
                alpha_spending_cur - actual_alpha_spent,
                0.0,
            )
            n_surviving_trials_h0 = n_surviving_trials_h0_new

        if beta_spending[i] == 0:
            futility_thresholds[i] = -np.inf
        else:
            m = np.floor(n_trials * beta_spending[i]).astype(int)
            if m >= n_surviving_trials_h1:
                # We run out of samples. So everything not effective in the current look is futile.
                futility_thresholds[i] = efficacy_thresholds[i] + EPSILON
                return efficacy_thresholds, futility_thresholds
            cur_samples_h1 = np.where(
                survivors_h1, samples_statistics_h1[:, :, i], -np.inf
            )
            max_cur_samples_h1 = np.max(cur_samples_h1, axis=1)
            futility_thresholds[i] = get_kth_value_by_size(
                max_cur_samples_h1, n_trials - n_surviving_trials_h1 + m
            )
            survivors_h1 = survivors_h1 & (cur_samples_h1 >= futility_thresholds[i])
            surviving_trials_h1 = np.any(survivors_h1, axis=1)

            cur_samples_h0 = np.where(
                survivors_h0, samples_statistics_h0[:, :, i], -np.inf
            )
            survivors_h0 = survivors_h0 & (cur_samples_h0 >= futility_thresholds[i])
            surviving_trials_h0 = np.any(survivors_h0, axis=1)

            n_surviving_trials_h0 = np.sum(surviving_trials_h0)
            n_surviving_trials_h1 = np.sum(surviving_trials_h1)

    futility_thresholds[-1] = efficacy_thresholds[-1]

    return efficacy_thresholds, futility_thresholds
