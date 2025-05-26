# draw error spending functions for the article
import matplotlib.pyplot as plt
import numpy as np


def f_HSD(x, gamma, alpha):
    # The Hwang-Shih-DeCani spending function
    if np.abs(gamma) < 0.0001:
        return alpha * x
    return alpha * (1 - np.exp(-gamma * x)) / (1 - np.exp(-gamma))


def generate_f_HSD(gamma, alpha):
    def f4beta(x):
        return f_HSD(x, gamma, alpha)

    return f4beta


# Generate x values
x = np.linspace(0.0001, 1, 100)

# Define the gamma values and corresponding function names
gamma_values = [-20, -5, -2, 0, 2, 5, 20]
print_location = [85, 70, 60, 50, 40, 30, 15]
function_names = [r"$\gamma=" + str(gamma) + "$" for gamma in gamma_values]

# Define shades of red
colorsCycle = plt.cm.Reds(np.linspace(0.3, 1, len(gamma_values)))

# Create the plot
plt.figure(figsize=(10, 6), dpi=800)
plt.title(r"Hwang-Shih-DeCani alpha-spending functions, $\alpha$=0.05")
plt.grid(True, color="gray", linestyle="--", linewidth=0.5)

# Plot each function with a different shade of red
alpha = 0.05
for gamma, name, color, print_index in zip(
    gamma_values, function_names, colorsCycle, print_location
):
    f = generate_f_HSD(gamma, alpha)
    y = f(x)
    plt.plot(x, y, label=name, color=color)
    plt.text(
        x[print_index],
        y[print_index] - 0.0005,
        name,
        color="black",
        fontsize=12,
        verticalalignment="center",
    )

# Show the plot
plt.xlabel("Information Fraction")
plt.ylabel("Cumulative Type I Error")
plt.savefig("spending_functions_hsd.eps", format="eps")
