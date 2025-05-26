import numpy as np
import numpy.typing as npt


def generate_spending_from_spending_function(
    f,
    look_fractions: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    generate the spending value at each look from the spending function.
    f: the spending function. Should satisfy f(0)=0 and f(1)=alpha, where alpha is the total value needed to be spent.
    look_fractions: the fraction of the total sample size (or of the information) at each look.
    Returns: 1D array shape (n_looks) which is the spending values at each interval.
    """
    n_looks = len(look_fractions)
    spending_function = np.zeros(n_looks)
    cumulative_spending = [f(i) for i in look_fractions]
    spending_function[0] = cumulative_spending[0]
    spending_function[1:] = np.diff(cumulative_spending)
    assert np.all(spending_function >= 0)

    return spending_function


def generate_spending_from_spending_parameter(
    gamma: float, alpha: float, look_fractions: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    Generate spending values based on the Hwang-Shih-DeCani (HSD) spending function.
    gamma: the parameter of the HSD spending function.
    alpha: the total spending value.
    look_fractions: the fraction of the total sample size (or of the information) at each look.
    Returns: 1D array shape (n_looks) which is the spending values at each interval.
    """
    assert look_fractions[0] > 0
    assert look_fractions[-1] == 1
    assert np.all(np.diff(look_fractions) > 0)

    def f_HSD(x):  # This is Hwang-Shih-DeCani spending function
        if np.abs(gamma) < 0.00001:
            return alpha * x
        return alpha * (1 - np.exp(-gamma * x)) / (1 - np.exp(-gamma))

    return generate_spending_from_spending_function(f_HSD, look_fractions)
