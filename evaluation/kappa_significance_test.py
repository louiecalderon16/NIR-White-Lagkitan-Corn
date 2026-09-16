"""
Calculates Cohen's Kappa, confidence interval,
z-statistic, and p-value for the 50-sample comparative evaluation.
"""

import numpy as np
from statsmodels.stats.inter_rater import cohens_kappa

# Confusion matrix from the 50-sample Comparative Evaluation
#                     Actual Sweet    Actual Not Sweet
# Predicted Sweet          28                1
# Predicted Not Sweet       8               13

table = np.array([
    [28, 1],
    [8, 13]
])

result = cohens_kappa(table)

print(result)

# Access individual values directly if needed for reporting:
print("\nKappa:", round(result.kappa, 4))
print("ASE under H0:", round(result.std_kappa0, 4))
print("Z statistic:", round(result.z_value, 4))
print("One-sided p-value:", result.pvalue_one_sided)
