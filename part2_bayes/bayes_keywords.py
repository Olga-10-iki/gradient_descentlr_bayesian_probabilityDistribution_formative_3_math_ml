"""
Part 2 - Bayes' theorem on the IMDb 50K Movie Reviews dataset.

Conditional probability chosen by the group: P(Positive | keyword)   (we do NOT compute P(Negative | keyword))

    P(Positive | kw) = P(kw | Positive) * P(Positive) / P(kw)

    Prior      P(Positive)        = #positive reviews / #reviews
    Likelihood P(kw | Positive)   = #positive reviews containing kw / #positive reviews
    Marginal   P(kw)              = #reviews containing kw / #reviews
                                  = P(kw|Pos)P(Pos) + P(kw|Neg)P(Neg)   (law of total probability, used as a check)

"Containing kw" = the keyword appears at least once as a whole word (document-level presence,
so a review that says "boring" five times still counts once).

Only the Python standard library is used (csv, re, os, urllib). No pandas, no sklearn, no nltk.

Usage:
    python bayes_keywords.py
    python bayes_keywords.py --pos excellent superb --neg waste awful    # try other keywords
"""

import argparse
import csv
import os
import re
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "IMDB-Dataset.csv")
OUT = os.path.join(HERE, "..", "outputs")
SOURCE_URL = "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv"

# Group's keyword choice -------------------------------------------------------------------------
POSITIVE_KEYWORDS = ["excellent", "wonderful", "masterpiece", "superb"]
NEGATIVE_KEYWORDS = ["worst", "waste", "boring", "awful"]

TAG_RE = re.compile(r"<[^>]+>")       # removes the "<br /><br />" HTML tags in the reviews
WORD_RE = re.compile(r"[a-z']+")


def ensure_dataset(path=DATA):
    if not os.path.exists(path):
        print(f"Dataset not found, downloading the IMDb 50K reviews CSV (~66 MB) to {path} ...")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(SOURCE_URL, path)


def tokenize(text):
    """lower-case, drop HTML tags, return the SET of words in the review"""
    text = TAG_RE.sub(" ", text.lower())
    return set(WORD_RE.findall(text))


def count_everything(path, keywords):
    """One pass over the file. Returns totals and, for each keyword, how many
    positive / negative reviews contain it."""
    total = 0
    total_pos = 0
    kw_in_pos = {k: 0 for k in keywords}
    kw_in_neg = {k: 0 for k in keywords}

    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader)                                  # skip header: review,sentiment
        for review, sentiment in reader:
            words = tokenize(review)
            is_pos = sentiment.strip().lower() == "positive"
            total += 1
            if is_pos:
                total_pos += 1
            for k in keywords:
                if k in words:
                    if is_pos:
                        kw_in_pos[k] += 1
                    else:
                        kw_in_neg[k] += 1
    return total, total_pos, kw_in_pos, kw_in_neg


def bayes_for_keyword(k, total, total_pos, kw_in_pos, kw_in_neg):
    total_neg = total - total_pos
    prior_pos = total_pos / total                              # P(Positive)
    prior_neg = total_neg / total                              # P(Negative)
    likelihood = kw_in_pos[k] / total_pos                      # P(kw | Positive)
    likelihood_neg = kw_in_neg[k] / total_neg                  # P(kw | Negative)  (only used for the marginal check)
    marginal = (kw_in_pos[k] + kw_in_neg[k]) / total           # P(kw)
    marginal_check = likelihood * prior_pos + likelihood_neg * prior_neg
    posterior = likelihood * prior_pos / marginal              # Bayes' theorem
    return {
        "keyword": k, "pos_with_kw": kw_in_pos[k], "neg_with_kw": kw_in_neg[k],
        "prior": prior_pos, "likelihood": likelihood, "marginal": marginal,
        "marginal_check": marginal_check, "posterior": posterior,
    }


def keyword_block(r, total, total_pos):
    n_kw = r["pos_with_kw"] + r["neg_with_kw"]
    return "\n".join([
        f"### Keyword: `{r['keyword']}`",
        "",
        "| Quantity | Formula | Counts | Value |",
        "|---|---|---|---|",
        f"| Prior P(Positive) | positive reviews / all reviews | {total_pos} / {total} | {r['prior']:.4f} |",
        f"| Likelihood P({r['keyword']} \\| Positive) | positive reviews with word / positive reviews | {r['pos_with_kw']} / {total_pos} | {r['likelihood']:.4f} |",
        f"| Marginal P({r['keyword']}) | reviews with word / all reviews | {n_kw} / {total} | {r['marginal']:.4f} |",
        f"| **Posterior P(Positive \\| {r['keyword']})** | likelihood × prior / marginal | {r['likelihood']:.4f} × {r['prior']:.4f} / {r['marginal']:.4f} | **{r['posterior']:.4f}** |",
        "",
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", nargs="+", default=POSITIVE_KEYWORDS)
    ap.add_argument("--neg", nargs="+", default=NEGATIVE_KEYWORDS)
    args = ap.parse_args()
    pos_kw = [k.lower() for k in args.pos]
    neg_kw = [k.lower() for k in args.neg]

    ensure_dataset()
    total, total_pos, kw_in_pos, kw_in_neg = count_everything(DATA, pos_kw + neg_kw)
    print(f"Reviews: {total}  (positive {total_pos}, negative {total - total_pos})")
    print("Computing P(Positive | keyword) for every keyword\n")

    header = f"{'keyword':<13}{'group':<10}{'P(Pos)':>8}{'P(kw|Pos)':>11}{'P(kw)':>9}{'P(Pos|kw)':>11}   check"
    print(header)
    print("-" * len(header))
    results = []
    for group, kws in (("positive", pos_kw), ("negative", neg_kw)):
        for k in kws:
            r = bayes_for_keyword(k, total, total_pos, kw_in_pos, kw_in_neg)
            r["group"] = group
            results.append(r)
            ok = "OK" if abs(r["marginal"] - r["marginal_check"]) < 1e-12 else "MISMATCH"
            print(f"{k:<13}{group:<10}{r['prior']:>8.4f}{r['likelihood']:>11.4f}{r['marginal']:>9.4f}"
                  f"{r['posterior']:>11.4f}   {ok}")

    print("\nReading the posterior: above 0.50 the word pushes a review towards POSITIVE "
          "(prior is 0.50), below 0.50 it pushes towards NEGATIVE.")

    os.makedirs(OUT, exist_ok=True)
    md = ["# Part 2 results: P(Positive | keyword) on IMDb 50K reviews", ""]
    md.append("## Positive-sentiment keywords\n")
    md += [keyword_block(r, total, total_pos) for r in results if r["group"] == "positive"]
    md.append("## Negative-sentiment keywords\n")
    md += [keyword_block(r, total, total_pos) for r in results if r["group"] == "negative"]
    with open(os.path.join(OUT, "part2_bayes_results.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    print("Saved outputs/part2_bayes_results.md")


if __name__ == "__main__":
    main()
