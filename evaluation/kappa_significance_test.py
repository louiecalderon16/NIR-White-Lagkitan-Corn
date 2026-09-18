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
print("\nKappa:", result.kappa)
print("SE of Kappa:", result.std_kappa)
print("SE under H0:", result.std_kappa0)
print("95% CI lower:", result.kappa_low)
print("95% CI upper:", result.kappa_upp)
print("z-value:", result.z_value)
print("One-sided p-value:", result.pvalue_one_sided)
print("Two-sided p-value:", result.pvalue_two_sided)
