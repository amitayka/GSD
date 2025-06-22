# GSD
Code accompanying the paper "Group Sequential Trial Design using Stepwise Monte Carlo for Increased Flexibility and Robustness".

## Project Structure

### Main Code: `gsd/basic_code`

The main code for the project is located in the `gsd/basic_code` folder, which contains the following files:
- **`gsd_threshold_finder_algorithm_1.py`**: Implements algorithms to determine efficacy and futility thresholds  
  for a multiarm group sequential design, given the required alpha and beta spending. This is the main code of the 
  article.
- **`find_good_hsd_spending_functions.py`**: Contains algorithms to find optimal Hwang-Shih-Decani spending functions.
- **`find_best_general_spending.py`**: Contains an algorithm to find an optimal general spending function.
- **`gsd_statistics_calculator.py`**: Provides utilities for calculating group sequential trial statistics, 
    such as efficacy and futility thresholds, type I and type II errors, and average sample sizes.

- **`utils/`**: A subfolder containing utility scripts:
  - **`utils/bayesian_approximation.py`**: Compute approximated Bayesian statistics of multiarm binomial trials.
  - **`utils/spending_function.py`**: Helper functions for spending function calculations.
  - **`utils/find_fixed_sample_size.py`**: Used to find the fixed sample size of a design based on Bayesian endpoints.

### Code for the article itself:
- **`gsd/bayesian_example/` A folder for outputs of the Bayesian designs in Sections 4 and Section 5.
  - **`bayesian_example/bayesian_example` The main code script the output and tables for Section 4 and Section 5.
    This is controlled by uncommenting parts of the code.
  - **`bayesian_example/section_4_run` Outputs of the code for Section 4.
  - **`bayesian_example/section_5_run_results` Output files used for Section 5.

- **`gsd/opt_gs_comparison/` Comparison of our method with OptGS
  - **`opt_gs_comparison/compare_with_opt_gs.py` A script to compare our method with OptGS.
  - **`opt_gs_comparison/output_files` output files of a run to get fast statistics.

- **`gsd/plot_spending_functions/generate_spending_function_plot.py` Generate Figure 2.

## Installation

To set up the project, ensure you have Python installed and follow these steps:

1. Install dependencies using Poetry:
   ```bash
   poetry install
