"""Find duplicate ingredients using the corpus itself as the oracle.

A is an alias of B when B's tokens are a subsequence of A's and B is used on its
own often enough to look canonical:

    "crispy bacon"      -> "bacon"     (bacon stands alone constantly)
    "fresh mint leaves" -> "mint"
    "olive oil"         -> no merge    ("oil" never stands alone)
    "chicken broth"     -> no merge    ("broth" never stands alone)

Writes the proposed merges to CSV. Eyeball them, then apply.
"""

import csv
from functools import cache
from itertools import combinations

import duckdb
import fire
import spacy

# The knob. Higher = fewer, safer merges. Tune against the CSV.
MIN_CANONICAL_USES = 3
MAX_TOKENS = 8  # ponytail: candidate search is 2^n; longer names are rare, skip them


@cache
def _nlp():
    # Tagger + attribute_ruler feed the rule-based lemmatizer; nothing here needs
    # a parse tree or entities.
    return spacy.load("en_core_web_sm", exclude=["parser", "ner"])


@cache
def tokens(description):
    """Lemmatized alpha tokens, stop words dropped."""
    doc = _nlp()(description.lower())
    return tuple(t.lemma_ for t in doc if t.is_alpha and not t.is_stop)


def find_merges(rows, min_uses=MIN_CANONICAL_USES):
    """rows: (id, description, uses). Returns (alias_row, canonical_row) pairs."""
    canonical = {}  # token tuple -> row, for names used often enough to absorb others
    for row in rows:
        key = tokens(row[1])
        if not key:
            continue
        prior = canonical.get(key)
        # Same tokens twice ("egg" and "eggs") is itself a duplicate; keep the
        # more-used spelling as canonical and let the other fall out as an alias.
        if prior is None or row[2] > prior[2]:
            canonical[key] = row

    merges = []
    for row in rows:
        key = tokens(row[1])
        if not key or len(key) > MAX_TOKENS:
            continue
        # Most specific match first: "red bell pepper" should land on "bell
        # pepper", not on "pepper". ponytail: one hop only, no transitive
        # chasing -- if the chain should collapse further, a second pass over
        # the applied merges will find it.
        for size in range(len(key), 0, -1):
            hit = None
            for sub in combinations(key, size):
                cand = canonical.get(sub)
                if cand and cand[0] != row[0] and cand[2] >= min_uses:
                    if hit is None or cand[2] > hit[2]:
                        hit = cand
            if hit is not None:
                merges.append((row, hit))
                break
    return merges


def resolve(merges):
    """alias id -> final canonical id, following one-hop merges to their end.

    find_merges can hand back "sharp cheddar -> cheddar" and "cheddar -> cheese"
    in the same batch; applying those literally would leave rows pointing at a
    deleted id.
    """
    target = {a[0]: c[0] for a, c in merges}
    out = {}
    for start in target:
        seen, end = {start}, target[start]
        while end in target and end not in seen:
            seen.add(end)
            end = target[end]
        out[start] = end
    return out


def apply_merges(con, merges):
    pairs = list(resolve(merges).items())
    con.executemany("update recipe_ingredients set ingredient_id = ? where ingredient_id = ?",
                    [(canon, alias) for alias, canon in pairs])
    # A recipe that listed both "bacon" and "crispy bacon" now has the row twice.
    con.execute("""
        create or replace temp table ri_dedup as select distinct * from recipe_ingredients;
        delete from recipe_ingredients;
        insert into recipe_ingredients select * from ri_dedup;
    """)
    # ponytail: no FK cascade in DuckDB, so clear the child row first.
    con.executemany("delete from ingredient_expansions where ingredient_id = ?",
                    [(alias,) for alias, _ in pairs])
    con.executemany("delete from ingredients where id = ?", [(alias,) for alias, _ in pairs])
    return len(pairs)


def main(db="recipes.db", out="ingredient_merges.csv", min_uses=MIN_CANONICAL_USES,
         apply=False):
    """Write the proposed merges to `out`, or --apply them to the database."""
    con = duckdb.connect(db, read_only=not apply)
    rows = con.execute("""
        select i.id, i.description, count(ri.recipe_id) as uses
        from ingredients i
        left join recipe_ingredients ri on ri.ingredient_id = i.id
        group by i.id, i.description
    """).fetchall()

    merges = find_merges(rows, min_uses)
    merges.sort(key=lambda m: -m[0][2])

    if apply:
        print(f"{apply_merges(con, merges)} ingredients merged away in {db}")
        return

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["alias_id", "alias", "alias_uses", "canonical_id", "canonical", "canonical_uses"])
        for alias, canon in merges:
            w.writerow([alias[0], alias[1], alias[2], canon[0], canon[1], canon[2]])

    print(f"{len(merges)} merges out of {len(rows)} ingredients -> {out}")


if __name__ == "__main__":
    fire.Fire(main)
