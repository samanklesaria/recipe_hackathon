"""A week of dinners, plus the shopping list split up by which store to visit.

Library only: the GUI calls shopping_plan() and renders what comes back. The
recipe half is recipes.sql's plan_week macro; the store half is the two-tower
dot product from finetune/train.py, each ingredient going to whichever store
scores highest.
"""
from collections import defaultdict
from typing import NamedTuple

import duckdb

import settings
import store_llm


class Recipe(NamedTuple):
    id: int
    name: str
    cooktime: int | None
    is_side: bool
    for_recipe: int | None   # the recipe that pulled this one in; None if a main pick
    ingredients: list[str]   # as the book writes them, quantities included


class Plan(NamedTuple):
    recipes: list[Recipe]
    # store description -> ingredient names. The None key holds what no store
    # scored above the threshold on: order it, or go find it yourself.
    by_store: dict[str | None, list[str]]


# The dot product is a logit trained against BCE, so 0 is p = 0.5: no store is
# more likely to stock it than not. ponytail: a fixed cut, not a calibration --
# the labels are an LLM's guess to begin with, so there is nothing to calibrate
# against until someone comes home empty-handed and says so.
THRESHOLD = 0.0


# The gloss is what gets embedded, the same string the fine-tune trained on.
# An ingredient with no gloss yet falls back to its bare name rather than
# dropping off the shopping list.
INGREDIENTS = """
SELECT DISTINCT i.description, coalesce(e.expansion, i.description)
FROM recipe_ingredients ri
JOIN ingredients i ON i.id = ri.ingredient_id
LEFT JOIN ingredient_expansions e ON e.ingredient_id = i.id
WHERE ri.recipe_id IN (SELECT unnest(?))
ORDER BY 1
"""


def shopping_plan(n=5, db=None, stores=None, threshold=THRESHOLD,
                  logits=store_llm.logits_from_embedding):
    """Plan n dinners and assign every ingredient they need to one store.

    An ingredient whose best store still scores below threshold lands under
    None rather than being sent to the least-bad option.

    stores defaults to the training_stores table; pass your own descriptions
    once you have a real list. logits is injectable for tests.
    """
    con = duckdb.connect(db or settings.DB, read_only=True)
    try:
        recipes = [Recipe(rid, name, cooktime, is_side, for_recipe, ingredients or [])
                   for rid, name, cooktime, is_side, for_recipe, ingredients
                   in con.execute("SELECT * FROM plan_week(?)", [n]).fetchall()]
        if not recipes:
            return Plan([], {})
        if stores is None:
            stores = [s for (s,) in con.execute(
                "SELECT description FROM training_stores ORDER BY id").fetchall()]
        rows = con.execute(INGREDIENTS, [[r.id for r in recipes]]).fetchall()
    finally:
        con.close()

    if not rows or not stores:
        return Plan(recipes, {})

    scores = logits([r[1] for r in rows], stores)
    best = scores.argmax(axis=1)

    by_store = defaultdict(list)
    for (name, _), j, row in zip(rows, best, scores):
        by_store[stores[j] if row[j] >= threshold else None].append(name)
    return Plan(recipes, dict(by_store))
