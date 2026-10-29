"""
Part 4 - Gradient descent for  y = m1*x1 + m2*x2 + b,  written with matrices and checked with SciPy.

Same problem as the Part 3 manual calculation:
    X = [[1, 3],      y = [5, 6]      m0 = [-1, 2]      b0 = [1, 1]
         [4, 10]]

Matrix form
    y_hat = X @ m + b
    e     = y_hat - y
    J     = (1/n) * e.T @ e                      (Mean Squared Error)
    dJ/dm = (2/n) * X.T @ e
    dJ/db = (2/n) * e
    m     = m - alpha * dJ/dm
    b     = b - alpha * dJ/db

SciPy part: `derivative_of(equation, point)` accepts ANY equation (a Python function of the
parameter vector) and returns its partial derivatives using scipy.differentiate.derivative.
We pass it the MSE equation and compare its answer with the hand-derived matrix formula
at every update, so the numbers in Part 3 are verified twice.

Usage:
    python gradient_descent.py                  # 3 updates (one per group member), alpha = 0.01
    python gradient_descent.py --members 4      # if the group has 4 members
    python gradient_descent.py --extra 50       # keep going after the members' updates to show the long-run trend
"""

import argparse
import os

import numpy as np
from scipy.differentiate import derivative

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs")

# ---------------------------------------------------------------- data and starting values
X = np.array([[1.0, 3.0],
              [4.0, 10.0]])
Y = np.array([5.0, 6.0])
M0 = np.array([-1.0, 2.0])
B0 = np.array([1.0, 1.0])
ALPHA = 0.01
N = X.shape[0]


# ---------------------------------------------------------------- the equation (cost function)
def mse_equation(params):
    """J(m1, m2, b1, b2) = (1/n) * || X @ m + b - y ||^2.
    `params` is the flat vector [m1, m2, b1, b2]."""
    m = params[:2]
    b = params[2:]
    e = X @ m + b - Y
    return (e @ e) / N


# ---------------------------------------------------------------- SciPy: equation -> derivative
def derivative_of(equation, point):
    """Partial derivative of `equation` w.r.t. every coordinate of `point`, using
    scipy.differentiate.derivative. For coordinate k we freeze all other coordinates and
    differentiate the 1-D slice  t -> equation(point with point[k] = t)."""
    point = np.asarray(point, dtype=float)
    grads = np.zeros_like(point)
    for k in range(point.size):
        def slice_k(t, k=k):
            t = np.asarray(t, dtype=float)
            out = np.empty(t.shape)
            for idx in np.ndindex(t.shape):        # SciPy evaluates at several t at once
                p = point.copy()
                p[k] = t[idx]
                out[idx] = equation(p)
            return out
        grads[k] = derivative(slice_k, point[k]).df
    return grads


def fmt(v):
    return "[" + ", ".join(f"{x:.6f}" for x in np.atleast_1d(v)) + "]"


# ---------------------------------------------------------------- gradient descent, every step visible
def run(updates, alpha=ALPHA, verbose=True):
    m = M0.copy()
    b = B0.copy()
    hist = {"m1": [m[0]], "m2": [m[1]], "b1": [b[0]], "b2": [b[1]], "mse": []}
    lines = []

    def log(s=""):
        lines.append(s)
        if verbose:
            print(s)

    for it in range(1, updates + 1):
        # Step 1: predictions
        y_hat = X @ m + b
        # Step 2: errors
        e = y_hat - Y
        # Step 3: cost
        J = (e @ e) / N
        # Step 4: gradients with the matrix formula
        dJ_dm = (2 / N) * (X.T @ e)
        dJ_db = (2 / N) * e
        # Step 5: same gradients with SciPy (numerical) for verification
        scipy_grad = derivative_of(mse_equation, np.concatenate([m, b]))
        # Step 6: update
        m_new = m - alpha * dJ_dm
        b_new = b - alpha * dJ_db

        if it == 1:
            hist["mse"].append(J)
        log(f"--- Update {it} ---")
        log(f"m = {fmt(m)}   b = {fmt(b)}")
        log(f"y_hat = X@m + b     = {fmt(y_hat)}")
        log(f"e = y_hat - y       = {fmt(e)}")
        log(f"J (MSE)             = {J:.6f}")
        log(f"dJ/dm = (2/n)X^T e  = {fmt(dJ_dm)}")
        log(f"dJ/db = (2/n)e      = {fmt(dJ_db)}")
        log(f"SciPy gradient      = {fmt(scipy_grad)}   (max diff {np.max(np.abs(scipy_grad - np.concatenate([dJ_dm, dJ_db]))):.2e})")
        log(f"m_new = m - {alpha}*dJ/dm = {fmt(m_new)}")
        log(f"b_new = b - {alpha}*dJ/db = {fmt(b_new)}")
        m, b = m_new, b_new
        y_hat_after = X @ m + b
        J_after = ((y_hat_after - Y) @ (y_hat_after - Y)) / N
        log(f"MSE after update    = {J_after:.6f}")
        log()
        for key, val in (("m1", m[0]), ("m2", m[1]), ("b1", b[0]), ("b2", b[1]), ("mse", J_after)):
            hist[key].append(val)
    return m, b, hist, lines


def plot(hist, members, path_params, path_error):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    its = list(range(len(hist["mse"])))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key, color in (("m1", "#1e88e5"), ("m2", "#e53935"), ("b1", "#43a047"), ("b2", "#fb8c00")):
        ax.plot(its, hist[key], marker="o", ms=3, label=key, color=color)
    ax.axvline(members, color="gray", ls=":", label=f"end of group's {members} manual updates")
    ax.set_title("How m and b change over iterations")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Parameter value")
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(path_params, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(its, hist["mse"], marker="o", ms=3, color="#6a1b9a")
    ax.axvline(members, color="gray", ls=":", label=f"end of group's {members} manual updates")
    ax.set_yscale("log")
    ax.set_title("How the error (MSE) changes over iterations")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSE (log scale)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path_error, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", type=int, default=3, help="number of group members = number of manual updates")
    ap.add_argument("--extra", type=int, default=100, help="extra updates shown only in the plots")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    args = ap.parse_args()

    print(f"X =\n{X}\ny = {Y}\nm0 = {M0}  b0 = {B0}  alpha = {args.alpha}\n")
    m, b, _, lines = run(args.members, args.alpha)

    print("=== Final values after the group's updates ===")
    print(f"m = {fmt(m)}   b = {fmt(b)}")
    final_pred = X @ m + b
    print(f"Predictions with final m, b: y_hat = {fmt(final_pred)}   (true y = {fmt(Y)})")

    # longer run, only for the trend plots
    _, _, hist, _ = run(args.members + args.extra, args.alpha, verbose=False)
    os.makedirs(OUT, exist_ok=True)
    plot(hist, args.members,
         os.path.join(OUT, "part4_params_over_iterations.png"),
         os.path.join(OUT, "part4_error_over_iterations.png"))
    with open(os.path.join(OUT, "part4_gd_log.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + f"\nFinal predictions: {fmt(final_pred)}\n")
    print("\nSaved outputs/part4_params_over_iterations.png, outputs/part4_error_over_iterations.png, outputs/part4_gd_log.txt")


if __name__ == "__main__":
    main()
