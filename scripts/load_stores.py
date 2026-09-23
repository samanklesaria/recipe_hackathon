"""Load grocery_store_examples.md into training_stores, one store per line.

Blank lines are separators, not entries. Re-runnable: description is UNIQUE, so
already-loaded stores are skipped rather than duplicated.
"""

import argparse

import duckdb


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="recipes.db")
    ap.add_argument("--src", default="finetune/grocery_store_examples.md")
    args = ap.parse_args()

    with open(args.src) as f:
        stores = [line.strip() for line in f if line.strip()]

    con = duckdb.connect(args.db)
    con.executemany(
        "insert into training_stores (description) values (?) on conflict do nothing",
        [(s,) for s in stores],
    )
    con.close()
    print(f"{len(stores)} lines from {args.src} -> {args.db}")


if __name__ == "__main__":
    main()
