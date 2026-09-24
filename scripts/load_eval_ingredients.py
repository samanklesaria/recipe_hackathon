"""Add the ingredients named in the fine-tune CSVs to the ingredients table.

Both grids name ingredients that may not appear in any loaded recipe, and
without a row here they never get a gloss from ingredient_expand.py -- so those
pairs train on the raw string while everything else trains on the gloss.
Re-runnable: description is UNIQUE, so existing ingredients are skipped.
"""

import csv

import duckdb
import fire

# csv -> the column holding the ingredient
SRCS = {"finetune/eval_pairs.csv": "ingredient",
        "finetune/explicit_store_examples.csv": "item"}


def main(db="recipes.db", srcs=SRCS):
    names = []
    for src, col in srcs.items():
        with open(src, newline="") as f:
            names += [r[col].strip() for r in csv.DictReader(f)]

    con = duckdb.connect(db)
    before = con.execute("select count(*) from ingredients").fetchone()[0]
    con.executemany(
        "insert into ingredients (description) values (?) on conflict do nothing",
        [(n,) for n in names],
    )
    after = con.execute("select count(*) from ingredients").fetchone()[0]
    con.close()
    print(f"{len(names)} named across {len(srcs)} files: {after - before} new, "
          f"{len(names) - (after - before)} already present or duplicated")


if __name__ == "__main__":
    fire.Fire(main)
