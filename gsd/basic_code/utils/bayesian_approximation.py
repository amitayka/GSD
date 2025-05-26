import numpy as np
import scipy as sp

DEFAULT_PRIOR_HYPERPARAM_ALPHA = 1.0
DEFAULT_PRIOR_HYPERPARAM_BETA = 1.0


def generate_bayesian_samples(
    n_samples_per_look_per_arm: np.ndarray,  # shape (n_looks)
    rate_per_arm: np.ndarray,  # shape (n_arms)
    n_trials: int,
    rng: np.random.Generator,
    prior_hyperparam_alpha: float = DEFAULT_PRIOR_HYPERPARAM_ALPHA,
    prior_hyperparam_beta: float = DEFAULT_PRIOR_HYPERPARAM_BETA,
) -> np.ndarray:
    """
    Generate sample statistics for a given set of arms and looks.
    The trial is a binomial trial and the samples are generated using an approximation of the Bayesian probability
    of the treatment arm being better than the control arm.
    The approximation is done using a normal distribution, following Cook, "Fast approximation of Beta inequalities".
    returns samples_statistics: shape (n_trials, n_arms, n_looks) each value is the approximate Bayesian probability
                                that the treatment arm is better than the control arm
    """
    n_arms = rate_per_arm.shape[0]
    n_looks = n_samples_per_look_per_arm.shape[0]
    n_successes = np.zeros((n_trials, n_arms, n_looks))
    for i in range(n_arms):
        n_successes[:, i, 0] = rng.binomial(
            n=n_samples_per_look_per_arm[0], p=rate_per_arm[i], size=n_trials
        )
        for j in range(1, n_looks):
            n_successes[:, i, j] = (
                rng.binomial(
                    n=n_samples_per_look_per_arm[j] - n_samples_per_look_per_arm[j - 1],
                    p=rate_per_arm[i],
                    size=n_trials,
                )
                + n_successes[:, i, j - 1]
            )

    samples_statistics = np.zeros((n_trials, n_arms, n_looks))
    for i in range(n_arms - 1):
        for j in range(n_looks):
            samples_statistics[:, i, j] = get_bayesian_probability_approximation(
                n_successes_control=n_successes[:, 0, j],
                n_successes_treatment=n_successes[:, i + 1, j],
                n_samples_control=n_samples_per_look_per_arm[j],
                n_samples_treatment=n_samples_per_look_per_arm[j],
                prior_hyperparam_alpha=prior_hyperparam_alpha,
                prior_hyperparam_beta=prior_hyperparam_beta,
            )
    return samples_statistics


def get_bayesian_probability_approximation(
    n_successes_control: np.ndarray,
    n_successes_treatment: np.ndarray,
    n_samples_control: int,
    n_samples_treatment: int,
    prior_hyperparam_alpha: float = DEFAULT_PRIOR_HYPERPARAM_ALPHA,
    prior_hyperparam_beta: float = DEFAULT_PRIOR_HYPERPARAM_BETA,
) -> np.ndarray:
    """
    Approximate the Bayesian probability of a treatment arm being better than the control arm.
    This is based on approximating beta distributions by normal distributions,
    following Cook, "Fast approximation of Beta inequalities".
    n_successes_control: shape (n_trials,)
    n_successes_treatment: shape (n_trials,)
    n_samples_control: int, number of samples in the control arm
    n_samples_treatment: int, number of samples in the treatment arm
    returns: array of shape (n_trials,), the approximate Bayesian probability that the treatment arm is better than the control arm
    """
    (
        hyperparam_mean_control,
        hyperparam_std_control,
    ) = get_bayesian_approximation(
        n_successes_control,
        n_samples_control,
        prior_hyperparam_alpha,
        prior_hyperparam_beta,
    )
    (
        hyperparam_mean_treatment,
        hyperparam_std_treatment,
    ) = get_bayesian_approximation(
        n_successes_treatment,
        n_samples_treatment,
        prior_hyperparam_alpha,
        prior_hyperparam_beta,
    )
    hyperparam_mean_diff = hyperparam_mean_treatment - hyperparam_mean_control
    hyperparam_std_diff = np.sqrt(
        hyperparam_std_control**2 + hyperparam_std_treatment**2
    )
    z_score = hyperparam_mean_diff / hyperparam_std_diff
    return sp.stats.norm.cdf(z_score)


def get_bayesian_approximation(
    n_successes: np.ndarray,
    n_samples: int,
    prior_hyperparam_alpha: float,
    prior_hyperparam_beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Approximate the Bayesian probability of a rate of a single arm trial.
    This is based on approximating beta distributions by normal distributions,
    following Cook, "Fast approximation of Beta inequalities".
    n_successes: shape (n_trials,)
    n_samples: int, number of samples in the trial
    returns tuple
        hyperparam_mean: array of shape (n_trials,), the approximate Bayesian mean of the rate
        hyperparam_std: array of shape (n_trials,), the approximate Bayesian standard deviation of the rate
    """

    hyperparam_alpha = prior_hyperparam_alpha + n_successes
    hyperparam_beta = prior_hyperparam_beta + n_samples - n_successes

    hyperparam_mean = hyperparam_alpha / (hyperparam_alpha + hyperparam_beta)

    hyperparam_std = np.sqrt(
        (hyperparam_alpha * hyperparam_beta)
        / (
            (hyperparam_alpha + hyperparam_beta) ** 2
            * (hyperparam_alpha + hyperparam_beta + 1)
        )
    )
    return hyperparam_mean, hyperparam_std
