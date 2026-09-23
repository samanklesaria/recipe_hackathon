"""Load grocery_store_examples.md into training_stores, one store per line.

Blank lines are separators, not entries. Re-runnable: description is UNIQUE, so
already-loaded stores are skipped rather than duplicated.
"""

import duckdb
import fire


def main(db="recipes.db", src="finetune/grocery_store_examples.md"):
    with open(src) as f:
        stores = [line.strip() for line in f if line.strip()]

    con = duckdb.connect(db)
    con.executemany(
        "insert into training_stores (description) values (?) on conflict do nothing",
        [(s,) for s in stores],
    )
    con.close()
    print(f"{len(stores)} lines from {src} -> {db}")


if __name__ == "__main__":
    fire.Fire(main)
