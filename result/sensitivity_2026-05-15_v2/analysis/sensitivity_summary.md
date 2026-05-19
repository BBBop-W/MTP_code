# Sensitivity Experiment Summary

Total runs: 225. Scales: small, medium, large.

## Solver Status

| problem_scale   | experiment_family   |   runs |   optimal_rate |   time_limit_rate |   avg_runtime_min |   avg_gap_pct |
|:----------------|:--------------------|-------:|---------------:|------------------:|------------------:|--------------:|
| large           | chunking            |     21 |       0.761905 |         0.238095  |        15.0115    |     0.10582   |
| large           | objective           |      3 |       1        |         0         |         0.322152  |     0         |
| large           | optional_ratio      |     18 |       0.722222 |         0.277778  |        17.2402    |     0.113884  |
| large           | proportion          |     33 |       0.818182 |         0.181818  |        11.595     |     0.0864845 |
| medium          | chunking            |     21 |       0.666667 |         0.333333  |        20.1375    |     0.118118  |
| medium          | objective           |      3 |       1        |         0         |         0.0048024 |     0         |
| medium          | optional_ratio      |     18 |       0.944444 |         0.0555556 |         3.90888   |     0.0536518 |
| medium          | proportion          |     33 |       0.727273 |         0.272727  |        16.4704    |     0.17411   |
| small           | chunking            |     21 |       0.333333 |         0.666667  |        40.7531    |     0.63641   |
| small           | objective           |      3 |       0.333333 |         0.666667  |        40.1301    |     1.35215   |
| small           | optional_ratio      |     18 |       0.666667 |         0.333333  |        22.3528    |     0.142559  |
| small           | proportion          |     33 |       0.393939 |         0.606061  |        36.4576    |     0.445243  |

## Objective Comparison

| problem_scale   |   len_gain_m_per_wagon |   qty_change_per_wagon |
|:----------------|-----------------------:|-----------------------:|
| small           |               0.793226 |             -0.0416667 |
| medium          |               0.810737 |              0         |
| large           |               0.560105 |             -0.2       |

## Best Composition by Scale

| problem_scale   |   p_small |   avg_len_per_wagon_m |   avg_qty_per_wagon |
|:----------------|----------:|----------------------:|--------------------:|
| large           |       0.5 |               45.1795 |            10       |
| medium          |       0.5 |               45.1573 |            10       |
| small           |       0.4 |               43.9699 |             9.66667 |

## Figure Files

- /Users/songtaowang/Documents/MTP_code/result/sensitivity_2026-05-15_v2/analysis/fig_objective_comparison.png
- /Users/songtaowang/Documents/MTP_code/result/sensitivity_2026-05-15_v2/analysis/fig_composition_sensitivity.png
- /Users/songtaowang/Documents/MTP_code/result/sensitivity_2026-05-15_v2/analysis/fig_optional_ratio_sensitivity.png
- /Users/songtaowang/Documents/MTP_code/result/sensitivity_2026-05-15_v2/analysis/fig_chunking_sensitivity.png
- /Users/songtaowang/Documents/MTP_code/result/sensitivity_2026-05-15_v2/analysis/fig_status_summary.png
