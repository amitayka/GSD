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
    The trial is a binomial trial and the samples are generated using an approximation of the Beyesian probability
    of the treatment arm being better than the control arm.
    The approximation is done using a normal distribution, following Cook, "Fast approximation of Beta inequalities".
    """
    n_arms = rate_per_arm.shape[0]
    n_looks = n_samples_per_look_per_arm.shape[0]
    max_samples = n_samples_per_look_per_arm[-1]
    measurements = np.zeros((n_trials, n_arms, max_samples))
    for i in range(n_arms):
        measurements[:, i, :] = rng.binomial(
            1, rate_per_arm[i], size=(n_trials, max_samples)
        )
    samples_statistics = np.zeros((n_trials, n_arms, n_looks))
    for i in range(n_arms - 1):
        for j in range(n_looks):
            samples_statistics[:, i, j] = get_bayesian_probability_approximation(
                data_control=measurements[:, 0, : n_samples_per_look_per_arm[j]],
                data_treatment=measurements[:, i + 1, : n_samples_per_look_per_arm[j]],
                prior_hyperparam_alpha=prior_hyperparam_alpha,
                prior_hyperparam_beta=prior_hyperparam_beta,
            )
    return samples_statistics


def get_bayesian_probability_approximation(
    data_control: np.ndarray,
    data_treatment: np.ndarray,
    prior_hyperparam_alpha: float = DEFAULT_PRIOR_HYPERPARAM_ALPHA,
    prior_hyperparam_beta: float = DEFAULT_PRIOR_HYPERPARAM_BETA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Approximate 2-arm bernoulli data using a normal distribution.
    This is based on approximating beta distributions by normal distributions,
    following Cook, "Fast approximation of Beta inequalities"
    """
    (
        hyperparam_mean_control,
        hyperparam_std_control,
    ) = get_bayesian_approximation(
        data_control,
        prior_hyperparam_alpha,
        prior_hyperparam_beta,
    )
    (
        hyperparam_mean_treatment,
        hyperparam_std_treatment,
    ) = get_bayesian_approximation(
        data_treatment,
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
    data: np.ndarray,
    prior_hyperparam_alpha: float,
    prior_hyperparam_beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Approximate the Bayesian probability of a rate of a single arm trial.
    This is based on approximating beta distributions by normal distributions,
    following Cook, "Fast approximation of Beta inequalities".
    data: shape (n_trials, n_samples)
    """

    n_samples = data.shape[1]
    n_success = np.sum(data, axis=1)

    hyperparam_alpha = prior_hyperparam_alpha + n_success
    hyperparam_beta = prior_hyperparam_beta + n_samples - n_success

    hyperparam_mean = hyperparam_alpha / (hyperparam_alpha + hyperparam_beta)

    hyperparam_std = np.sqrt(
        (hyperparam_alpha * hyperparam_beta)
        / (
            (hyperparam_alpha + hyperparam_beta) ** 2
            * (hyperparam_alpha + hyperparam_beta + 1)
        )
    )
    return hyperparam_mean, hyperparam_std
