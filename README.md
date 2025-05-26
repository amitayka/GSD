# GSD
Code accompanying the paper "Group Sequential Trial Design using Stepwise Monte Carlo for Increased Flexibility and Robustness".

## Project Structure

### Main Code: `gsd/basic_code`

The main code for the project is located in the `gsd/basic_code` folder, which contains the following files:
- **`gsd_threshold_finder_algorithm_1.py`**: Implements algorithms to determine efficacy and futility thresholds for 
    a multiarm group sequential design, given the required alpha and beta spending. This is the main code of the 
    article.
- **`find_good_hsd_spending_functions.py`**: Contains algorithms to find optimal Hwang-Shih-Decani spending functions.
- **`find_best_general_spending.py`**: Contains an algorithm to find an optimal general spending function.
- **`gsd_statistics_calculator.py`**: Provides utilities for calculating group sequential trial statistics, 
    such as efficacy and futility thresholds, type I and type II errors, and average sample sizes.

- **`utils/`**: A subfolder containing utility scripts:
  - **`utils/bayesian_approximation.py`**: Compute approximated Bayesian statistics of multiarm binomial trials.
  - **`utils/spending_function.py`**: Helper functions for spending function calculations.
  - **`utils/find_fixed_sample_size.py`**: Used to find the fixed sample size of a design based on Bayesian endpoints.

### Other Folders and Files

- **`gsd/generate_figures/`**: Contains scripts for generating plots and figures used in the paper. 
  - `bayesian_example.py`: A script for generating all the data in Section 4 of the paper. See this file for an example of how to use the other files.
  - `compare_with_opt_gs.py`: Compare our method with OptGS.
  - `generate_spending_function_plot.py`: Plot the hsd spending function plot.


## Installation

To set up the project, ensure you have Python installed and follow these steps:

1. Install dependencies using Poetry:
   ```bash
   poetry install
