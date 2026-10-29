# Part 2 results: P(Positive | keyword) on IMDb 50K reviews

## Positive-sentiment keywords

### Keyword: `excellent`

| Quantity | Formula | Counts | Value |
|---|---|---|---|
| Prior P(Positive) | positive reviews / all reviews | 25000 / 50000 | 0.5000 |
| Likelihood P(excellent \| Positive) | positive reviews with word / positive reviews | 2865 / 25000 | 0.1146 |
| Marginal P(excellent) | reviews with word / all reviews | 3549 / 50000 | 0.0710 |
| **Posterior P(Positive \| excellent)** | likelihood × prior / marginal | 0.1146 × 0.5000 / 0.0710 | **0.8073** |

### Keyword: `wonderful`

| Quantity | Formula | Counts | Value |
|---|---|---|---|
| Prior P(Positive) | positive reviews / all reviews | 25000 / 50000 | 0.5000 |
| Likelihood P(wonderful \| Positive) | positive reviews with word / positive reviews | 2255 / 25000 | 0.0902 |
| Marginal P(wonderful) | reviews with word / all reviews | 2777 / 50000 | 0.0555 |
| **Posterior P(Positive \| wonderful)** | likelihood × prior / marginal | 0.0902 × 0.5000 / 0.0555 | **0.8120** |

### Keyword: `masterpiece`

| Quantity | Formula | Counts | Value |
|---|---|---|---|
| Prior P(Positive) | positive reviews / all reviews | 25000 / 50000 | 0.5000 |
| Likelihood P(masterpiece \| Positive) | positive reviews with word / positive reviews | 876 / 25000 | 0.0350 |
| Marginal P(masterpiece) | reviews with word / all reviews | 1200 / 50000 | 0.0240 |
| **Posterior P(Positive \| masterpiece)** | likelihood × prior / marginal | 0.0350 × 0.5000 / 0.0240 | **0.7300** |

### Keyword: `superb`

| Quantity | Formula | Counts | Value |
|---|---|---|---|
| Prior P(Positive) | positive reviews / all reviews | 25000 / 50000 | 0.5000 |
| Likelihood P(superb \| Positive) | positive reviews with word / positive reviews | 1026 / 25000 | 0.0410 |
| Marginal P(superb) | reviews with word / all reviews | 1199 / 50000 | 0.0240 |
| **Posterior P(Positive \| superb)** | likelihood × prior / marginal | 0.0410 × 0.5000 / 0.0240 | **0.8557** |

## Negative-sentiment keywords

### Keyword: `worst`

| Quantity | Formula | Counts | Value |
|---|---|---|---|
| Prior P(Positive) | positive reviews / all reviews | 25000 / 50000 | 0.5000 |
| Likelihood P(worst \| Positive) | positive reviews with word / positive reviews | 409 / 25000 | 0.0164 |
| Marginal P(worst) | reviews with word / all reviews | 4428 / 50000 | 0.0886 |
| **Posterior P(Positive \| worst)** | likelihood × prior / marginal | 0.0164 × 0.5000 / 0.0886 | **0.0924** |

### Keyword: `waste`

| Quantity | Formula | Counts | Value |
|---|---|---|---|
| Prior P(Positive) | positive reviews / all reviews | 25000 / 50000 | 0.5000 |
| Likelihood P(waste \| Positive) | positive reviews with word / positive reviews | 174 / 25000 | 0.0070 |
| Marginal P(waste) | reviews with word / all reviews | 2533 / 50000 | 0.0507 |
| **Posterior P(Positive \| waste)** | likelihood × prior / marginal | 0.0070 × 0.5000 / 0.0507 | **0.0687** |

### Keyword: `boring`

| Quantity | Formula | Counts | Value |
|---|---|---|---|
| Prior P(Positive) | positive reviews / all reviews | 25000 / 50000 | 0.5000 |
| Likelihood P(boring \| Positive) | positive reviews with word / positive reviews | 591 / 25000 | 0.0236 |
| Marginal P(boring) | reviews with word / all reviews | 3049 / 50000 | 0.0610 |
| **Posterior P(Positive \| boring)** | likelihood × prior / marginal | 0.0236 × 0.5000 / 0.0610 | **0.1938** |

### Keyword: `awful`

| Quantity | Formula | Counts | Value |
|---|---|---|---|
| Prior P(Positive) | positive reviews / all reviews | 25000 / 50000 | 0.5000 |
| Likelihood P(awful \| Positive) | positive reviews with word / positive reviews | 281 / 25000 | 0.0112 |
| Marginal P(awful) | reviews with word / all reviews | 2875 / 50000 | 0.0575 |
| **Posterior P(Positive \| awful)** | likelihood × prior / marginal | 0.0112 × 0.5000 / 0.0575 | **0.0977** |
