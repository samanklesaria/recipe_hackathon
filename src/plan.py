"""A week of dinners, plus the shopping list split up by which store to visit.
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



def shopping_plan(n=5, db=None, threshold=0.6899744811276125, embed=store_llm.embed):
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
        rows = con.execute("SELECT * FROM plan_ingredients(?)",
                           [[r.id for r in recipes]]).fetchall()

        missing = [i for i, r in enumerate(rows) if r[3] is None]
        if missing:
            fresh = embed([store_llm.ING_PREFIX + rows[i][2] for i in missing])
            con.executemany("UPDATE ingredients SET embedding = ? WHERE id = ?",
                            [([float(x) for x in v], rows[i][0])
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
