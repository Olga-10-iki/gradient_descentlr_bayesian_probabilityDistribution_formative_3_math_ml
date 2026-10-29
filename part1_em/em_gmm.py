"""
Part 1 - Expectation-Maximization (EM) for a 2-component Gaussian mixture, written from scratch.

Population pair chosen: FATHERS vs CHILDREN (Galton Families dataset).

We pool every father height (one per family) and every child height into ONE unlabeled list,
then let EM discover the two Gaussians hidden inside it:

    p(x) = pi1 * N(x | mu1, var1) + pi2 * N(x | mu2, var2)

    component 1 -> Children
    component 2 -> Parents (Fathers)   (the "Pros" column in the assignment table)

No sklearn / scipy.stats is used for the model. Only `math` and plain Python lists.
The gender/father labels are used ONLY at the very end to check how good the unsupervised result is.

Usage:
    python em_gmm.py                     # fit, print tracking table, save plot
    python em_gmm.py --height 70.5       # also classify a test height (the coach's number)
    python em_gmm.py --height 70.5 --units cm   # if the coach gives centimetres
"""

import argparse
import csv
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "GaltonFamilies.csv")
OUT = os.path.join(HERE, "..", "outputs")


# ----------------------------------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------------------------------
def load_heights(path=DATA):
    """Return (fathers, children). Fathers are de-duplicated: the CSV repeats the father on
    every child row, so we keep one father height per family id."""
    fathers_by_family = {}
    children = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            fathers_by_family[row["family"]] = float(row["father"])
            children.append(float(row["childHeight"]))
    return list(fathers_by_family.values()), children


# ----------------------------------------------------------------------------------------------
# 2. Small helpers (written by hand on purpose)
# ----------------------------------------------------------------------------------------------
def mean(values):
    return sum(values) / len(values)


def variance(values):
    m = mean(values)
    return sum((v - m) ** 2 for v in values) / len(values)


def percentile(values, q):
    s = sorted(values)
    pos = (len(s) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def gaussian_pdf(x, mu, var):
    """N(x | mu, var) = 1/sqrt(2*pi*var) * exp(-(x-mu)^2 / (2*var))"""
    return math.exp(-((x - mu) ** 2) / (2.0 * var)) / math.sqrt(2.0 * math.pi * var)


def log_likelihood(data, p):
    """log L = sum_i log( pi1*N(x_i|mu1,var1) + pi2*N(x_i|mu2,var2) )"""
    return sum(
        math.log(p["pi1"] * gaussian_pdf(x, p["mu1"], p["var1"])
                 + p["pi2"] * gaussian_pdf(x, p["mu2"], p["var2"]))
        for x in data
    )


# ----------------------------------------------------------------------------------------------
# 3. EM steps
# ----------------------------------------------------------------------------------------------
def initialise(data):
    """Initialisation that does NOT use labels and does NOT split at the global mean:
    - means start at the 25th and 75th percentiles (lower bump = children, upper = fathers)
    - both variances start at the overall variance (deliberately too wide)
    - mixing weights start at 50/50 (we pretend we know nothing)"""
    overall_var = variance(data)
    return {"mu1": percentile(data, 0.25), "mu2": percentile(data, 0.75),
            "var1": overall_var, "var2": overall_var, "pi1": 0.5, "pi2": 0.5}


def e_step(data, p):
    """Responsibility r_i = P(child | x_i) =
         pi1*N(x_i|mu1,var1) / ( pi1*N(x_i|mu1,var1) + pi2*N(x_i|mu2,var2) )"""
    resp = []
    for x in data:
        a = p["pi1"] * gaussian_pdf(x, p["mu1"], p["var1"])
        b = p["pi2"] * gaussian_pdf(x, p["mu2"], p["var2"])
        resp.append(a / (a + b))
    return resp


def m_step(data, resp):
    """Weighted maximum-likelihood re-estimates:
         N1  = sum r_i                N2 = sum (1 - r_i)
         mu1 = sum r_i x_i / N1       mu2 = sum (1-r_i) x_i / N2
         var1 = sum r_i (x_i-mu1)^2 / N1  (same for var2 with 1-r_i)
         pi1 = N1 / n                 pi2 = N2 / n"""
    n = len(data)
    n1 = sum(resp)
    n2 = n - n1
    mu1 = sum(r * x for r, x in zip(resp, data)) / n1
    mu2 = sum((1 - r) * x for r, x in zip(resp, data)) / n2
    var1 = sum(r * (x - mu1) ** 2 for r, x in zip(resp, data)) / n1
    var2 = sum((1 - r) * (x - mu2) ** 2 for r, x in zip(resp, data)) / n2
    return {"mu1": mu1, "mu2": mu2, "var1": var1, "var2": var2, "pi1": n1 / n, "pi2": n2 / n}


def run_em(data, max_iter=500, tol=1e-6):
    p = initialise(data)
    history = [dict(p, ll=log_likelihood(data, p))]
    for _ in range(max_iter):
        resp = e_step(data, p)          # E: how much does each point belong to each group?
        p = m_step(data, resp)          # M: refit each Gaussian using those soft memberships
        ll = log_likelihood(data, p)
        history.append(dict(p, ll=ll))
        if abs(ll - history[-2]["ll"]) < tol:
            break
    return p, history


# ----------------------------------------------------------------------------------------------
# 4. Classification of a new height (Bayes' rule with the fitted mixture)
# ----------------------------------------------------------------------------------------------
def classify(height, p):
    like_child = gaussian_pdf(height, p["mu1"], p["var1"])
    like_parent = gaussian_pdf(height, p["mu2"], p["var2"])
    joint_child = p["pi1"] * like_child
    joint_parent = p["pi2"] * like_parent
    evidence = joint_child + joint_parent
    return {
        "likelihood_child": like_child, "likelihood_parent": like_parent,
        "joint_child": joint_child, "joint_parent": joint_parent, "evidence": evidence,
        "p_child": joint_child / evidence, "p_parent": joint_parent / evidence,
    }


# ----------------------------------------------------------------------------------------------
# 5. The "just split at the global mean" baseline (for the presentation question)
# ----------------------------------------------------------------------------------------------
def naive_mean_split(data):
    g = mean(data)
    low = [x for x in data if x < g]
    high = [x for x in data if x >= g]
    return g, low, high


# ----------------------------------------------------------------------------------------------
# 6. Printing / plotting
# ----------------------------------------------------------------------------------------------
def print_table(history, rows=None):
    header = (f"{'Iteration':<20}{'mu1 (Children)':>15}{'mu2 (Parents)':>15}"
              f"{'sigma1^2':>11}{'sigma2^2':>11}{'pi1':>9}{'pi2':>9}{'Log-Likelihood':>17}")
    print(header)
    print("-" * len(header))
    rows = rows if rows is not None else range(len(history))
    for i in rows:
        h = history[i]
        label = "0 (Initialization)" if i == 0 else str(i)
        print(f"{label:<20}{h['mu1']:>15.4f}{h['mu2']:>15.4f}{h['var1']:>11.4f}{h['var2']:>11.4f}"
              f"{h['pi1']:>9.4f}{h['pi2']:>9.4f}{h['ll']:>17.4f}")


def save_markdown_table(history, path):
    lines = ["| Iteration | μ1 (Children) | μ2 (Parents/Fathers) | σ1² | σ2² | π1 | π2 | Log-Likelihood |",
             "|---|---|---|---|---|---|---|---|"]
    for i, h in enumerate(history):
        label = "0 (Initialization)" if i == 0 else str(i)
        lines.append(f"| {label} | {h['mu1']:.4f} | {h['mu2']:.4f} | {h['var1']:.4f} | {h['var2']:.4f} "
                     f"| {h['pi1']:.4f} | {h['pi2']:.4f} | {h['ll']:.4f} |")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def plot(data, history, fathers, children, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = history[-1]
    xs = [55 + i * 0.05 for i in range(int((82 - 55) / 0.05))]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    ax = axes[0]
    ax.hist(data, bins=40, density=True, color="#cfd8dc", edgecolor="white", label="Pooled heights (unlabeled)")
    ax.plot(xs, [p["pi1"] * gaussian_pdf(x, p["mu1"], p["var1"]) for x in xs], color="#1e88e5", lw=2,
            label=f"Children  μ={p['mu1']:.2f}")
    ax.plot(xs, [p["pi2"] * gaussian_pdf(x, p["mu2"], p["var2"]) for x in xs], color="#e53935", lw=2,
            label=f"Fathers  μ={p['mu2']:.2f}")
    ax.plot(xs, [p["pi1"] * gaussian_pdf(x, p["mu1"], p["var1"]) + p["pi2"] * gaussian_pdf(x, p["mu2"], p["var2"])
                 for x in xs], color="black", lw=1.2, ls="--", label="Mixture")
    g = mean(data)
    ax.axvline(g, color="gray", ls=":", label=f"Global mean {g:.2f}")
    ax.set_title("EM fit: two hidden Gaussians in the pooled heights")
    ax.set_xlabel("Height (inches)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(range(len(history)), [h["ll"] for h in history], marker="o", ms=3, color="#43a047")
    ax.set_title("Log-likelihood never decreases")
    ax.set_xlabel("EM iteration")
    ax.set_ylabel("Log-likelihood")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=float, help="test height to classify")
    ap.add_argument("--units", choices=["in", "cm"], default="in", help="units of --height (data is inches)")
    args = ap.parse_args()

    fathers, children = load_heights()
    data = fathers + children                     # labels thrown away from here on
    print(f"Pooled {len(data)} unlabeled heights ({len(fathers)} fathers + {len(children)} children), inches\n")

    final, history = run_em(data)

    print("=== Tracking table: initial state + first two iterations ===")
    print_table(history, rows=[0, 1, 2])
    print(f"\n=== Converged after {len(history) - 1} iterations ===")
    print_table(history, rows=[len(history) - 1])

    # Ground truth check (labels used ONLY here)
    print("\n=== Sanity check against the true labels (not used for fitting) ===")
    print(f"True children: mean={mean(children):.4f}  var={variance(children):.4f}  share={len(children)/len(data):.4f}")
    print(f"True fathers : mean={mean(fathers):.4f}  var={variance(fathers):.4f}  share={len(fathers)/len(data):.4f}")

    # Naive split
    g, low, high = naive_mean_split(data)
    print("\n=== 'Draw a line at the global mean' baseline ===")
    print(f"Global mean = {g:.4f}")
    print(f"Pile below: n={len(low)}  mean={mean(low):.4f}  var={variance(low):.4f}")
    print(f"Pile above: n={len(high)}  mean={mean(high):.4f}  var={variance(high):.4f}")
    fathers_below = sum(1 for f in fathers if f < g)
    children_above = sum(1 for c in children if c >= g)
    print(f"Fathers wrongly put in the 'low' pile : {fathers_below}/{len(fathers)}")
    print(f"Children wrongly put in the 'high' pile: {children_above}/{len(children)}")

    os.makedirs(OUT, exist_ok=True)
    save_markdown_table(history, os.path.join(OUT, "part1_em_tracking_table.md"))
    plot(data, history, fathers, children, os.path.join(OUT, "part1_em_fit.png"))
    print(f"\nSaved outputs/part1_em_tracking_table.md and outputs/part1_em_fit.png")

    if args.height is not None:
        h = args.height / 2.54 if args.units == "cm" else args.height
        r = classify(h, final)
        print(f"\n=== Classifying test height {args.height} {args.units} ({h:.2f} in) ===")
        print(f"Likelihood  N(x|Children) = {r['likelihood_child']:.6f}   N(x|Fathers) = {r['likelihood_parent']:.6f}")
        print(f"Prior x Likelihood         = {r['joint_child']:.6f}             = {r['joint_parent']:.6f}")
        print(f"Evidence p(x)              = {r['evidence']:.6f}")
        print(f"P(Child  | x) = {r['p_child']:.6f}  ({r['p_child']*100:.2f}%)")
        print(f"P(Parent | x) = {r['p_parent']:.6f}  ({r['p_parent']*100:.2f}%)")
        print("Prediction:", "CHILD" if r["p_child"] > r["p_parent"] else "PARENT (Father)")


if __name__ == "__main__":
    main()
