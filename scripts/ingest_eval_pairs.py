"""Put the hand-labelled eval ingredients into the ingredients table.

They have to live there so ingredient_expand.py glosses them like any other
ingredient, and so baseline_eval.py can read those glosses back out.

    uv run python ingest_eval_pairs.py
    uv run python ingredient_expand.py     # then gloss them
"""
import argparse

import extract
import settings

# Normalized the same way extract.store() does it, so an eval ingredient that
# also appears in a cookbook lands on the same row rather than a near-duplicate.
SQL = """
INSERT INTO ingredients (description)
SELECT DISTINCT lower(trim(regexp_replace(ingredient, '\\s+', ' ', 'g')))
FROM read_csv(?)
ON CONFLICT (description) DO NOTHING
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="eval_pairs.csv")
    args = ap.parse_args()

    con = extract.connect(settings.DB)
    before = con.execute("SELECT count(*) FROM ingredients").fetchone()[0]
    con.execute(SQL, (args.pairs,))
    con.commit()
    after = con.execute("SELECT count(*) FROM ingredients").fetchone()[0]
    con.close()
    print(f"{after - before} new ingredients from {args.pairs} "
          f"({after} total in {settings.DB})")


if __name__ == "__main__":
    main()
