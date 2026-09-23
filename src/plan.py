"""A week of dinners, plus the shopping list split up by which store to visit.

Library only: the GUI calls shopping_plan() and renders what comes back. The
recipe half is recipes.sql's plan_week macro; the store half is the two-tower
dot product from finetune/train.py, each ingredient going to whichever store
scores highest.
"""
from collections import defaultdict
from typing import NamedTuple

import duckdb
import numpy as np

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
SELECT DISTINCT i.id, i.description, coalesce(e.expansion, i.description),
       v.embedding
FROM recipe_ingredients ri
JOIN ingredients i ON i.id = ri.ingredient_id
LEFT JOIN ingredient_expansions e ON e.ingredient_id = i.id
LEFT JOIN ingredient_embeddings v ON v.ingredient_id = i.id
WHERE ri.recipe_id IN (SELECT unnest(?))
ORDER BY 2
"""


def shopping_plan(n=5, db=None, threshold=THRESHOLD, embed=store_llm.embed):
    """Plan n dinners and assign every ingredient they need to one store.

    An ingredient whose best store still scores below threshold lands under
    None rather than being sent to the least-bad option.

    Stores and their embeddings come from the stores table, written by the GUI.
    Ingredient embeddings are filled in here, once, the first time an
    ingredient turns up. embed is injectable for tests.
    """
    con = duckdb.connect(db or settings.DB)
    try:
        recipes = [Recipe(rid, name, cooktime, is_side, for_recipe, ingredients or [])
                   for rid, name, cooktime, is_side, for_recipe, ingredients
                   in con.execute("SELECT * FROM plan_week(?)", [n]).fetchall()]
        if not recipes:
            return Plan([], {})
        # a store with no embedding was never saved through the GUI; it cannot
        # be scored, so it is not offered
        shops = con.execute(
            "SELECT description, embedding FROM stores"
            " WHERE embedding IS NOT NULL ORDER BY priority").fetchall()
        rows = con.execute(INGREDIENTS, [[r.id for r in recipes]]).fetchall()

        missing = [i for i, r in enumerate(rows) if r[3] is None]
        if missing:
            fresh = embed([store_llm.ING_PREFIX + rows[i][2] for i in missing])
            con.executemany("INSERT INTO ingredient_embeddings VALUES (?, ?)",
                            [(rows[i][0], [float(x) for x in v])
                             for i, v in zip(missing, fresh)])
            for i, v in zip(missing, fresh):
                rows[i] = (*rows[i][:3], v)
    finally:
        con.close()

    if not rows or not shops:
        return Plan(recipes, {})

    stores = [s for s, _ in shops]
    scores = np.array([r[3] for r in rows]) @ np.array([v for _, v in shops]).T

    by_store = defaultdict(list)
    for (_, name, _, _), row in zip(rows, scores):
        # stores is already in priority order, so the first one that can stock
        # it takes it -- fewer trips beats a slightly better match.
        pick = next((s for s, v in zip(stores, row) if v >= threshold), None)
        by_store[pick].append(name)
    return Plan(recipes, dict(by_store))
