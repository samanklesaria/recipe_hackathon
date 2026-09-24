"""Expand a bare ingredient name into a short gloss for the store-matching encoder.

    urad dal -> urad dal - split black lentil, South Asian pulse, sold dried

A 300m encoder cannot know what "urad dal" is; the gloss is cheaper than any
amount of fine-tuning (see the README). Expansions are cached in
ingredients.expansion, so a rerun costs nothing.

    llama serve --models-preset probes/models.ini     # in another terminal
    uv run python scripts/ingredient_expand.py

Every uncached row in the ingredients table, committed one at a time so an
interrupted run keeps what it already paid for.
"""
import sys

import duckdb
import requests

import settings

# One line, no quotes, bounded length. Bounding it in the grammar rather than
# with max_tokens means a run-on answer is impossible, not merely truncated.
GRAMMAR = r"""
root ::= [^"\n]{10,160}
"""

PROMPT = """Describe a cooking ingredient in one line, for someone deciding \
which grocery store sells it.

Give the form it is sold in, what it is, and the cuisine or aisle it belongs to. \
No amounts, no preparation instructions, no sentences.

urad dal -> split black lentil, South Asian pulse, sold dried
gochujang -> fermented Korean chili paste, sold in a tub, Korean aisle
scallion -> green onion, fresh produce, sold in bunches
cream of tartar -> powdered tartaric acid, baking aisle, small jar

{ingredient} ->"""

def expand(ingredient):
    """Return "<ingredient> - <gloss>", the string that gets embedded."""
    body = {
        "model": settings.CHAT_MODEL,
        "messages": [{"role": "user", "content": PROMPT.format(ingredient=ingredient)}],
        "grammar": GRAMMAR,
        "temperature": 0,
        "max_tokens": 128,
    }
    r = requests.post(f"{settings.SERVER}/v1/chat/completions", json=body, timeout=600)
    r.raise_for_status()
    gloss = r.json()["choices"][0]["message"]["content"].strip().strip("-").strip()
    return f"{ingredient} - {gloss}"


def main():
    con = duckdb.connect(settings.DB)
    rows = con.execute(
        "SELECT id, description FROM ingredients WHERE expansion IS NULL "
        "ORDER BY description").fetchall()

    done = 0
    for ing_id, desc in rows:
        try:
            text = expand(desc)
        except Exception as e:
            print(f"  {desc}: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            continue
        con.execute("UPDATE ingredients SET expansion = ? WHERE id = ?", (text, ing_id))
        con.commit()
        done += 1
        print(text, flush=True)

    con.close()
    print(f"\n{done}/{len(rows)} expanded -> {settings.DB}:ingredients.expansion")


if __name__ == "__main__":
    main()
