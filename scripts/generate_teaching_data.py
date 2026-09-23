"""Label every (ingredient gloss, training store) pair with the LLM.

The chat model's ordinal guess is the teaching signal the embedding fine-tune is
distilled from (see the README). Labels go into
embed_training_data as the word ('always'|'usually'|'sometimes'|'never'); the
word -> probability mapping belongs to finetune/train.py.

    llama serve --models-preset probes/models.ini     # in another terminal
    uv run python scripts/generate_teaching_data.py

One store at a time, every ingredient it is missing, committed per row. The
store-major order is the point: the store paragraph is the shared prompt prefix
that llama-server keeps in its KV cache (see store_llm.label_store). Resume is
per pair, so an interrupted run costs nothing twice.
"""
import duckdb

import settings
from store_llm import label_store

MISSING = """
SELECT e.ingredient_id, e.expansion FROM ingredient_expansions e
ANTI JOIN embed_training_data d
  ON d.ingredient_id = e.ingredient_id AND d.training_store_id = ?
ORDER BY e.expansion
"""


def main():
    con = duckdb.connect(settings.DB)
    stores = con.execute(
        "SELECT id, description FROM training_stores ORDER BY id").fetchall()
    done = 0
    for store_id, store in stores:
        todo = con.execute(MISSING, (store_id,)).fetchall()
        print(f"\n[{store_id}] {store[:70]} -- {len(todo)} to label", flush=True)
        for (ing_id, gloss), label in zip(todo, label_store(
                store, [g for _, g in todo])):
            con.execute("INSERT INTO embed_training_data VALUES (?, ?, ?)",
                        (ing_id, store_id, label))
            con.commit()
            done += 1
            print(f"  {label:9} {gloss}", flush=True)

    con.close()
    print(f"\n{done} pairs -> {settings.DB}:embed_training_data")


if __name__ == "__main__":
    main()
