# Tables (EXPLORATORY; aggregates only)

## Scores by point: primary judge (llama-3.3-70b-instruct-q8_0)

| point | rows | F ok | A ok | analysed | fairness mean | altruism mean | F refusals |
|---|---|---|---|---|---|---|---|
| base | 120 | 120 | 120 | 120 | 85.17 | 84.42 | 0 |
| D-3 | 120 | 120 | 120 | 116 | 83.42 | 77.54 | 0 |
| D+3 | 120 | 120 | 120 | 118 | 86.92 | 85.58 | 0 |
| T-3 | 120 | 120 | 120 | 120 | 81.83 | 63.58 | 0 |
| T+3 | 120 | 120 | 120 | 114 | 58.78 | 94.42 | 0 |
| S-3 | 120 | 120 | 120 | 120 | 86.13 | 85.66 | 0 |
| S+3 | 120 | 120 | 120 | 120 | 85.86 | 84.58 | 0 |
| O-3 | 120 | 120 | 120 | 120 | 86.14 | 84.77 | 0 |
| O+3 | 120 | 120 | 120 | 120 | 85.94 | 83.85 | 0 |

## Scores by point: secondary judge (gpt-oss-20b-mxfp4)

| point | rows | F ok | A ok | analysed | fairness mean | altruism mean | F refusals |
|---|---|---|---|---|---|---|---|
| base | 30 | 30 | 30 | 30 | 67.33 | 67.50 | 0 |
| D-3 | 30 | 30 | 30 | 30 | 64.67 | 60.33 | 0 |
| D+3 | 30 | 30 | 30 | 30 | 72.17 | 66.00 | 0 |
| T-3 | 30 | 30 | 30 | 30 | 70.33 | 37.00 | 0 |
| T+3 | 30 | 30 | 30 | 30 | 44.67 | 79.33 | 0 |
| S-3 | 30 | 30 | 30 | 30 | 67.67 | 69.00 | 0 |
| S+3 | 30 | 30 | 30 | 30 | 66.50 | 72.50 | 0 |
| O-3 | 30 | 30 | 30 | 30 | 69.17 | 61.33 | 0 |
| O+3 | 30 | 30 | 30 | 30 | 67.00 | 71.17 | 0 |

## Contrasts: primary judge (95% question-clustered bootstrap)

| contrast | estimate | 95% CI | questions | rows | verdict |
|---|---|---|---|---|---|
| a1: F(D+3)-F(base) | 1.75 | [-1.95, 6.21] | 40 | 238 | descriptive |
| a2: F(D+3)-F(T+3)  [P1] | 28.35 | [19.03, 38.17] | 40 | 232 | supported |
| a2 at -3: F(D-3)-F(T-3) | 1.28 | [-4.75, 7.64] | 40 | 236 | descriptive |
| F(D+3)-F(S+3) | 1.06 | [-1.41, 3.18] | 40 | 238 | descriptive |
| F(D+3)-F(O+3) | 0.98 | [-1.00, 2.80] | 40 | 238 | descriptive |
| F(D-3)-F(S-3) | -3.02 | [-7.91, 1.43] | 40 | 236 | descriptive |
| F(D-3)-F(O-3) | -3.03 | [-8.37, 1.92] | 40 | 236 | descriptive |
| A(D+3)-A(base) | 1.17 | [-2.56, 6.11] | 40 | 238 | descriptive |
| A(D+3)-A(T+3)  [P2] | -8.36 | [-11.57, -4.24] | 40 | 232 | supported |
| A(D-3)-A(T-3) | 14.40 | [6.84, 22.12] | 40 | 236 | descriptive |
| b1 interaction +3 vs T  [P3] | 36.70 | [26.41, 47.14] | 40 | 232 | supported |
| c1 interaction -3 vs T | -13.13 | [-20.89, -5.77] | 40 | 236 | descriptive |
| c2 interaction +3 vs S | 0.05 | [-2.66, 2.70] | 40 | 238 | descriptive |
| c3 interaction +3 vs O | -0.76 | [-3.72, 2.16] | 40 | 238 | descriptive |
| c4 interaction -3 vs S | 4.66 | [-1.89, 11.40] | 40 | 236 | descriptive |
| c5 interaction -3 vs O | 3.76 | [-3.96, 11.13] | 40 | 236 | descriptive |

## Contrasts: secondary judge (95% question-clustered bootstrap)

| contrast | estimate | 95% CI | questions | rows | verdict |
|---|---|---|---|---|---|
| a1: F(D+3)-F(base) | 3.17 | [-2.88, 9.81] | 26 | 60 | descriptive |
| a2: F(D+3)-F(T+3)  [P1] | 26.92 | [18.08, 35.58] | 26 | 60 | descriptive |
| a2 at -3: F(D-3)-F(T-3) | -8.27 | [-16.92, -0.38] | 26 | 60 | descriptive |
| F(D+3)-F(S+3) | 6.73 | [0.58, 13.85] | 26 | 60 | descriptive |
| F(D+3)-F(O+3) | 4.81 | [-1.35, 11.73] | 26 | 60 | descriptive |
| F(D-3)-F(S-3) | -5.19 | [-15.96, 5.19] | 26 | 60 | descriptive |
| F(D-3)-F(O-3) | -5.77 | [-17.12, 4.81] | 26 | 60 | descriptive |
| A(D+3)-A(base) | 0.77 | [-8.85, 10.38] | 26 | 60 | descriptive |
| A(D+3)-A(T+3)  [P2] | -11.44 | [-19.04, -4.04] | 26 | 60 | descriptive |
| A(D-3)-A(T-3) | 21.06 | [10.77, 32.02] | 26 | 60 | descriptive |
| b1 interaction +3 vs T  [P3] | 38.37 | [24.81, 51.83] | 26 | 60 | descriptive |
| c1 interaction -3 vs T | -29.33 | [-42.98, -16.63] | 26 | 60 | descriptive |
| c2 interaction +3 vs S | 12.60 | [2.50, 23.37] | 26 | 60 | descriptive |
| c3 interaction +3 vs O | 7.98 | [-1.92, 18.65] | 26 | 60 | descriptive |
| c4 interaction -3 vs S | 4.71 | [-7.12, 16.44] | 26 | 60 | descriptive |
| c5 interaction -3 vs O | -3.75 | [-17.98, 10.58] | 26 | 60 | descriptive |

## Discriminant validity (baseline and null rows, primary judge)

Pearson r = 0.22, 95% CI [-0.06, 0.50]; Spearman = 0.31; rows = 600, questions = 40; r > 0.8: False

