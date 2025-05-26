import numpy as np

from gsd.basic_code.utils.bayesian_approximation import generate_bayesian_samples

n_trials_list = [10_000, 50_000, 200_000, 1_000_000, 5_000_000]

multiplier_list = [2, 1.5, 1.2, 1.1, 1.05]     # UDI: add comment for what is a multiplier


def get_power_given_sample_size(
    alpha: float,
    rate_per_arm_h0: np.ndarray,
    rate_per_arm_h1: np.ndarray,
    sample_size: int,
    n_trials: int,
    rng: np.random.Generator,
) -> float:
    samples_h0 = generate_bayesian_samples(
        n_samples_per_look_per_arm=np.array([sample_size]),
        rate_per_arm=rate_per_arm_h0,
        n_trials=n_trials,
        rng=rng,
    )[
        :, :, 0
    ]  # shape (n_trials, n_arms), since we can ignore the last look         # UDI: it's not "since we can ignore the last look", but we have no looks, only final
    maximal_sample_h0 = np.max(samples_h0, axis=1)
    threshold = np.quantile(maximal_sample_h0, 1 - alpha)
    samples_h1 = generate_bayesian_samples(
        n_samples_per_look_per_arm=np.array([sample_size]),
        rate_per_arm=rate_per_arm_h1,
        n_trials=n_trials,
        rng=rng,
    )[
        :, :, 0
    ]  # shape (n_trials, n_arms), since we can ignore the last look         # UDI: see above
    maximal_sample_h1 = np.max(samples_h1, axis=1)

    return np.mean(maximal_sample_h1 > threshold)


def find_fixed_sample_size_for_bayesian_endpoint(   # UDI: give n_trials_list and multiplier_list as parameters, not globals
    alpha: float,
    power: float,
    rate_per_arm_h0: np.ndarray,
    rate_per_arm_h1: np.ndarray,
    seed: int = 1729,
    verbose=False,
):                                                  # UDI: add return type
    sample_size = 150  # some initial guess
    rng = np.random.default_rng(seed)
    for i, multiplier in enumerate(multiplier_list):
        n_trials = n_trials_list[i]  # Start with the smallest number of trials
        if verbose:
            print(f"Trying n_trials={n_trials}")

        current_power = get_power_given_sample_size(
            alpha=alpha,
            rate_per_arm_h0=rate_per_arm_h0,
            rate_per_arm_h1=rate_per_arm_h1,
            sample_size=sample_size,
            n_trials=n_trials,
            rng=rng,
        )
        if verbose:
            print(f"sample_size={sample_size}, current_power={current_power}")
        if current_power >= power:
            while current_power >= power:
                upper_power = current_power
                sample_size_upper_bound = sample_size
                sample_size = int(sample_size / multiplier)
                current_power = get_power_given_sample_size(
                    alpha=alpha,
                    rate_per_arm_h0=rate_per_arm_h0,
                    rate_per_arm_h1=rate_per_arm_h1,
                    sample_size=sample_size,
                    n_trials=n_trials,
                    rng=rng,
                )
                if verbose:
                    print(f"sample_size={sample_size}, current_power={current_power}")
            lower_power = current_power
            sample_size_lower_bound = sample_size
        else:
            while current_power < power:
                lower_power = current_power
                sample_size_lower_bound = sample_size
                sample_size = int(sample_size * multiplier)
                current_power = get_power_given_sample_size(
                    alpha=alpha,
                    rate_per_arm_h0=rate_per_arm_h0,
                    rate_per_arm_h1=rate_per_arm_h1,
                    sample_size=sample_size,
                    n_trials=n_trials,
                    rng=rng,
                )
                if verbose:
                    print(f"sample_size={sample_size}, current_power={current_power}")
            upper_power = current_power
            sample_size_upper_bound = sample_size
        # Now we have a range [sample_size_lower_bound, sample_size_upper_bound]
        while sample_size_upper_bound - sample_size_lower_bound > 1:
            sample_size = int((sample_size_lower_bound + sample_size_upper_bound) / 2)
            current_power = get_power_given_sample_size(
                alpha=alpha,
                rate_per_arm_h0=rate_per_arm_h0,
                rate_per_arm_h1=rate_per_arm_h1,
                sample_size=sample_size,
                n_trials=n_trials,
                rng=rng,
            )
            if verbose:
                print(
                    f"sample_size={sample_size}, current_power={current_power}, "
                    f"lower_bound={sample_size_lower_bound}, upper_bound={sample_size_upper_bound}"
                )
            if current_power >= power:
                sample_size_upper_bound = sample_size
                upper_power = current_power
            else:
                sample_size_lower_bound = sample_size
                lower_power = current_power
        if np.abs(upper_power - power) < np.abs(lower_power - power):
            sample_size = sample_size_upper_bound
        else:
            sample_size = sample_size_lower_bound
        # UDI: add a printout (if verbose) with final sample size, power

    return sample_size


rates_h0 = np.array([0.5, 0.5, 0.5])
rates_h1 = np.array([0.5, 0.6, 0.7])
# UDI: move n_trials_list and multiplier_list to here and give explicitly
if __name__ == "__main__":
    print(
        f"Sample size for alpha=0.05, power=0.8, rates_h0={rates_h0}, rates_h1={rates_h1}, n_trials={n_trials}:",
        find_fixed_sample_size_for_bayesian_endpoint(
            alpha=0.025,
            power=0.8,
            rate_per_arm_h0=rates_h0,
            rate_per_arm_h1=rates_h1,
            seed=1729,
            verbose=True,
        ),
    )
