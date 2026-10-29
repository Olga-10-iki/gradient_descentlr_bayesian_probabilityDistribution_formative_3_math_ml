# Math Director

Group assignment on **probability distributions, Bayesian probability and gradient descent**.

| Part | What it does | Code | Output |
|---|---|---|---|
| 1 | EM algorithm (from scratch) separating **fathers vs children** heights in Galton's data | `part1_em/em_gmm.py` | `outputs/part1_em_tracking_table.md`, `outputs/part1_em_fit.png` |
| 2 | Bayes' theorem, **P(Positive \| keyword)** on IMDb 50K reviews, standard library only | `part2_bayes/bayes_keywords.py` | `outputs/part2_bayes_results.md` |
| 3 | Gradient descent worked by hand in matrix form (3 updates) | `part3_manual/manual_gradient_descent.md` | |
| 4 | Same gradient descent in NumPy, gradients verified with SciPy, Matplotlib plots | `part4_gd_code/gradient_descent.py` | `outputs/part4_*.png`, `outputs/part4_gd_log.txt` |

## Setup

```bash
pip install -r requirements.txt
python part1_em/em_gmm.py --height 70         # the coach's test height goes here (inches)
python part1_em/em_gmm.py --height 178 --units cm
python part2_bayes/bayes_keywords.py          # downloads IMDB-Dataset.csv into data/ on first run
python part4_gd_code/gradient_descent.py      # --members 4 if the group has 4 people
```

Data:
- `data/GaltonFamilies.csv` (included): 934 children from 205 families, heights in inches.
- `data/IMDB-Dataset.csv` (≈66 MB, not committed): 50,000 reviews, 25,000 positive and 25,000 negative. The script downloads it automatically from a public GitHub mirror of the Kaggle "IMDB Dataset of 50K Movie Reviews". Add `data/IMDB-Dataset.csv` to `.gitignore`; it is close to GitHub's size limit.

---

## Part 1: EM on Galton heights

**Setup.** We pool the 205 father heights (one per family; the CSV repeats the father on every child row) with the 934 child heights. That gives 1,139 unlabeled numbers, modelled as

p(x) = π₁·N(x | μ₁, σ₁²) + π₂·N(x | μ₂, σ₂²), with component 1 = Children and component 2 = Parents (Fathers).

**Initialisation.** μ at the 25th and 75th percentiles, both σ² at the overall variance, and π = 0.5 / 0.5. No labels are used, and the split is not at the global mean.

### Tracking table (printed by the code)

| Iteration | μ1 (Children) | μ2 (Fathers) | σ1² | σ2² | π1 | π2 | Log-Likelihood |
|---|---|---|---|---|---|---|---|
| 0 (Initialization) | 64.5000 | 70.0000 | 12.7241 | 12.7241 | 0.5000 | 0.5000 | −3119.8222 |
| 1 | 65.2205 | 69.2054 | 9.0603 | 8.4468 | 0.5011 | 0.4989 | −3057.9135 |
| 2 | 65.1609 | 69.2600 | 8.9420 | 8.1040 | 0.5005 | 0.4995 | −3056.9719 |
| 319 (converged) | 64.1772 | 69.5186 | 5.3024 | 6.0410 | 0.4325 | 0.5675 | −3046.9895 |

The full per-iteration table is in `outputs/part1_em_tracking_table.md`. Note that the log-likelihood rises every iteration, which EM guarantees.

### Classification demo (for the coach's height)

`python part1_em/em_gmm.py --height <h>` prints the likelihoods, the prior×likelihood, the evidence p(x), and the exact posteriors. Two examples:

| Test height | P(Child \| x) | P(Parent \| x) |
|---|---|---|
| 64 in | 0.9098 | 0.0902 |
| 70 in | 0.0328 | 0.9672 |

The assignment's table says "Pros / Basketball Player"; that is template wording. In our version, component 2 is **Fathers**.

### Question: should we just draw a line at the global mean and average each pile?

**No.** Our own numbers show why:

1. **The groups are not the same size.** There are 934 children and only 205 fathers (82% vs 18%). The global mean (67.21 in) is pulled towards the children, so it is not the boundary between the two groups. **403 of 934 children (43%) land in the "tall" pile**, and 44 of 205 fathers land in the "short" pile.
2. **The piles' statistics are biased.** Cutting a bell curve and averaging each side pushes the two means apart and shrinks the variances. The naive piles have variances of 4.14 and 3.57, while the true children's variance is 12.80. The cut chops each distribution's tail off and gives it to the other group.
3. **Hard assignment throws information away.** A 67.0 in person and a 60 in person both count 100% as "short". EM instead gives every point a **soft responsibility**. Points near the overlap count partly towards each group, which is exactly how the overlap should be treated.
4. **No variances or mixing weights.** A mean split gives two means only. EM learns μ, σ² and π for each group, and that is what lets us compute a real posterior P(Child | x) for a new height. The mean split can only say "left or right of 67.21".

### Honest finding 

Look at the sanity check the script prints: EM's "children" component (μ = 64.2) is **not** the true children mean (66.7). The reason is that "children" in Galton's data are **sons (≈69.2 in) and daughters (≈64.1 in)**, and sons are about as tall as fathers (≈69.3 in). So the biggest structure in the pooled data is **sex, not generation**. EM, which only sees heights, finds "shorter people (mostly daughters)" versus "taller people (fathers + sons)". That is the maximum-likelihood answer for this data. Our takeaway: unsupervised mixture models find the strongest *statistical* split, which may not be the *label* we care about. The model is only as meaningful as the assumption that each group is one bell curve.

---

## Part 2: Bayes on IMDb reviews

Conditional probability chosen: **P(Positive | keyword)**. A review "contains" a keyword if the word appears at least once (HTML `<br />` tags removed, lower-cased, whole words only).

| Keyword | Group | Prior P(Pos) | Likelihood P(kw\|Pos) | Marginal P(kw) | **Posterior P(Pos\|kw)** |
|---|---|---|---|---|---|
| excellent | positive | 0.5000 | 0.1146 | 0.0710 | **0.8073** |
| wonderful | positive | 0.5000 | 0.0902 | 0.0555 | **0.8120** |
| masterpiece | positive | 0.5000 | 0.0350 | 0.0240 | **0.7300** |
| superb | positive | 0.5000 | 0.0410 | 0.0240 | **0.8557** |
| worst | negative | 0.5000 | 0.0164 | 0.0886 | **0.0924** |
| waste | negative | 0.5000 | 0.0070 | 0.0507 | **0.0687** |
| boring | negative | 0.5000 | 0.0236 | 0.0610 | **0.1938** |
| awful | negative | 0.5000 | 0.0112 | 0.0575 | **0.0977** |

Per-keyword markdown blocks with the raw counts are in `outputs/part2_bayes_results.md`.

**What the numbers say:**
- The prior is exactly 0.5 (a balanced dataset), so every move away from 0.5 comes purely from the keyword.
- Negative words are **stronger evidence** than positive ones. Seeing "waste" drops P(Positive) to 6.9%, while the best positive word, "superb", only lifts it to 85.6%.
- **"masterpiece" is the weakest positive word (0.73).** It appears in sarcasm and comparisons ("this is no masterpiece", "compared to the original masterpiece..."), and the keyword approach cannot see negation.
- **"boring" is the weakest negative word (0.19).** Positive reviews often say "never boring" or "boring at first, but...".
- The code also re-derives P(kw) with the law of total probability, P(kw|Pos)P(Pos) + P(kw|Neg)P(Neg), and prints `OK` when it matches.

---

## Part 3 and Part 4: Gradient descent

The manual working is in `part3_manual/manual_gradient_descent.md`. Each update has a blank for the member who did it.

The code prints the same steps (prediction → error → MSE → gradient → update) for each iteration. It checks every analytic gradient against `scipy.differentiate.derivative` (agreement within 1e-13). It also saves:
- `outputs/part4_params_over_iterations.png`: m₁, m₂, b₁, b₂ over iterations
- `outputs/part4_error_over_iterations.png`: MSE over iterations (log scale)

| After update | m | b | MSE |
|---|---|---|---|
| 0 | [−1, 2] | [1, 1] | 61.0000 |
| 1 | [−1.45, 0.87] | [0.99, 0.89] | 6.5033 |
| 2 | [−1.3331, 1.1765] | [1.0185, 0.9121] | 2.4974 |
| 3 | [−1.369037, 1.095583] | [1.036351, 0.898653] | 2.1645 |

Final predictions with m⁽³⁾, b⁽³⁾: ŷ = [2.9541, 6.3783] (true y = [5, 6]).

Learning rate α = 0.01 (not given in the brief). Stability requires α < 2/λ_max ≈ 0.0157.

---

## Contribution

Each member must add at least one proof of contribution (commits, or the updates signed in Part 3). Fill in:

| Member | Part(s) | Proof |
|---|---|---|
| | | |
