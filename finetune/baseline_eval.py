"""Zero-shot store-matching baseline: off-the-shelf embeddinggemma logits.

This is step 3 of store_matching.md -- the number a fine-tune has to beat before
it is worth shipping. No training, no fitting: embed each store description and
each ingredient with the two prompt prefixes and score every pair the way the
fine-tune will, with the unnormalized dot product:

    logit = E_store(store) . E_ing(ingredient)      P(stocked) = sigmoid(logit)

Unnormalized matters: a well-stocked supermarket should be able to say so with a
larger norm, which cosine throws away. That needs `embd-normalize = -1` in the
preset -- llama-server L2-normalizes embeddings by default, and with normalized
vectors this is cosine again.

    uv run python ingest_eval_pairs.py               # once
    uv run python ingredient_expand.py               # once, glosses them
    llama serve --models-preset probes/models.ini    # in another terminal
    uv run python baseline_eval.py

Hand labels come from eval_pairs.csv, store descriptions from
grocery_store_examples.md, ingredient glosses from the db. Everything downstream
of the embeddings is one (ingredient, store) matrix. Cells where the baseline
disagrees with the hand label are written to baseline_predictions.csv; cells it
got right are blank, so the file reads as a map of where the baseline is wrong.
"""
import argparse
import csv
import re

import duckdb
import numpy as np
import requests

import settings

PAIRS = "manually_tagged/eval_pairs.csv"
STORES = "manually_tagged/grocery_store_examples.md"
OUT = "baseline_predictions.csv"

# store_matching.md: embeddinggemma is prompt-sensitive and trained with
# asymmetric query/document roles, so use the roles rather than bare text.
ING_PREFIX = "task: search result | query: {}"
STORE_PREFIX = "title: none | text: {}"

# Ordinal labels, descending stockedness, and the probabilities they stand for.
LABELS = np.array(["always", "usually", "sometimes", "never"])
P = np.array([0.95, 0.80, 0.20, 0.05])


def embed(texts):
    r = requests.post(f"{settings.SERVER}/v1/embeddings", timeout=600,
                      json={"model": settings.EMBED_MODEL, "input": texts})
    r.raise_for_status()
    data = sorted(r.json()["data"], key=lambda d: d["index"])
    return np.array([d["embedding"] for d in data])


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -700, 700)))


def load_stores(path, names):
    """Pull each store's free-text description out of the markdown.

    Keyed off the csv header rather than a split on ':' -- one entry in the file
    separates name from description with a period instead.
    """
    text = open(path).read()
    out = {}
    for name in names:
        m = re.search(rf"^{re.escape(name)}\s*[:.]\s*(.+?)(?:\n\s*\n|\Z)",
                      text, re.M | re.S)
        if not m:
            raise SystemExit(f"{path}: no description for {name!r}")
        out[name] = " ".join(m.group(1).split())
    return out


def expansions(ingredients):
    """ingredient -> gloss, read from the cache ingredient_expand.py wrote."""
    con = duckdb.connect(settings.DB, read_only=True)
    rows = dict(con.execute(
        "SELECT i.description, e.expansion FROM ingredients i "
        "JOIN ingredient_expansions e ON e.ingredient_id = i.id").fetchall())
    con.close()
    missing = [i for i in ingredients if i.strip().lower() not in rows]
    if missing:
        raise SystemExit(f"{settings.DB}: no expansion for {len(missing)} ingredients, "
                         f"e.g. {missing[:3]}. Run ingest_eval_pairs.py then "
                         f"ingredient_expand.py.")
    return {i: rows[i.strip().lower()] for i in ingredients}


def calibrate(logits, truth):
    """Map raw logits onto the four labels by rank. Both args are (ing, store).

    sigmoid(logit) is only a probability once the model has been trained to make
    it one; untrained, the logits are dominated by embedding norm and sigmoid
    saturates at 1.0 for every pair. So rank instead: cut the sorted logits at
    the hand labels' own frequencies, which asks the only question the baseline
    can answer -- does it *order* the pairs the way a human does?

    ponytail: this hands the baseline the true label distribution, which the
    fine-tune will not get -- deliberately generous, so beating it means
    something.
    """
    counts = np.bincount(truth.ravel(), minlength=len(LABELS))
    pred = np.empty(truth.size, dtype=int)
    pred[np.argsort(-logits, axis=None)] = np.repeat(np.arange(len(LABELS)), counts)
    return pred.reshape(truth.shape)


def auc(scores, positive):
    """Rank AUC for "is this actually stocked".

    ponytail: plain ranks, no tie correction -- two float dot products landing
    on the same bits does not happen here.
    """
    ranks = np.empty(scores.size, dtype=float)
    ranks[np.argsort(scores, axis=None)] = np.arange(1, scores.size + 1)
    pos, n_pos = positive.ravel(), positive.sum()
    n_neg = positive.size - n_pos
    if not n_pos or not n_neg:
        return np.nan
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def load_eval(pairs=PAIRS, stores=STORES):
    """The hand-labelled eval set: glossed ingredients, store blurbs, truth.

    Returns (ingredients, glosses, store names, store descriptions, truth), with
    truth an (ingredient, store) matrix of indices into LABELS. train.py scores
    the same set with the fine-tuned model, so this lives here rather than in
    main() -- one definition of what the eval set is.
    """
    rows = list(csv.DictReader(open(pairs)))
    names = [c for c in rows[0] if c != "ingredient"]
    descs = load_stores(stores, names)
    ingredients = [r["ingredient"] for r in rows]
    gloss = expansions(ingredients)

    labels = np.array([[r[n].strip() for n in names] for r in rows])
    bad = np.isin(labels, LABELS, invert=True)
    if bad.any():
        i, j = np.argwhere(bad)[0]
        raise SystemExit(f"{pairs}: bad label {labels[i, j]!r} "
                         f"for {ingredients[i]}/{names[j]}")
    truth = np.argmax(labels[..., None] == LABELS, axis=-1)
    return (ingredients, [gloss[i] for i in ingredients],
            names, [descs[n] for n in names], truth)


def report(logits, truth, n_stores):
    """Print the scoreboard for an (ingredient, store) logit matrix.

    Returns the rank-calibrated predictions. Baseline and fine-tune both go
    through here so the numbers being compared are computed identically.
    """
    pred = calibrate(logits, truth)
    stocked = P[truth] >= 0.5

    print(f"{truth.size} pairs, {n_stores} stores, {len(truth)} ingredients")
    print(f"  exact 4-way accuracy   {(pred == truth).mean():6.1%}   (chance 25.0%)")
    print(f"  stocked/not accuracy   {((P[pred] >= 0.5) == stocked).mean():6.1%}")
    print(f"  AUC (stocked vs not)   {auc(logits, stocked):6.3f}")
    print(f"  mean |p_pred - p_true| {np.abs(P[pred] - P[truth]).mean():6.3f}")
    # If the sigmoids are all 1.000 the embeddings came back L2-normalized, or
    # the norms are simply large: either way sigmoid is useless untrained and
    # the ranking above is the real result.
    print(f"  logits {logits.min():.2f}..{logits.max():.2f}, "
          f"sigmoid {sigmoid(logits.min()):.3f}..{sigmoid(logits.max()):.3f}")

    # The answer that actually matters: "no store carries this". Wrong there and
    # the user abandons a recipe they could have cooked.
    unavailable = ~stocked.any(axis=1)
    best = logits.max(axis=1)
    print(f"\n  truly unavailable anywhere: {unavailable.sum()}/{len(truth)} ingredients")
    print('  threshold sweep on max-logit-over-stores ("no store carries this"):')
    print("    cutoff  said-none  correct  recall-of-has-it")
    for cut in np.quantile(logits, np.arange(0, 1.0, 0.1)):
        said_none = best < cut
        recall = (~said_none & ~unavailable).sum() / (~unavailable).sum()
        print(f"    {cut:6.2f}  {said_none.sum():9d}  "
              f"{(said_none & unavailable).sum():7d}  {recall:16.1%}")
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=PAIRS)
    ap.add_argument("--stores", default=STORES)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    ingredients, gloss, names, descs, truth = load_eval(args.pairs, args.stores)
    logits = embed([ING_PREFIX.format(g) for g in gloss]) @ \
        embed([STORE_PREFIX.format(d) for d in descs]).T
    pred = report(logits, truth, len(names))

    wrong = pred != truth
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ingredient"] + names)
        w.writerows([[ing] + list(np.where(wrong_row, LABELS[pred_row], ""))
                     for ing, wrong_row, pred_row in zip(ingredients, wrong, pred)])
    print(f"\n{wrong.sum()} disagreements -> {args.out}")


if __name__ == "__main__":
    main()
