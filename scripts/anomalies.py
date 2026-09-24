"""Find rows the epub parser got wrong, by the length of them and nothing else.

Three things go wrong often enough to look for:

  * a recipe with no ingredients at all -- a heading the parser mistook for a
    recipe, or a body it failed to read
  * a recipe whose "name" is a paragraph: a book title plus the first sentence
    of the blurb, or a run-on list of component recipes
  * an ingredient whose "description" is a cooking instruction

All three are the same symptom, a chunk boundary in the wrong place, and all
three are visible from the character count. No model required.

Prints by default:

    uv run python scripts/anomalies.py
    uv run python scripts/anomalies.py --delete-them
"""

import duckdb
import fire

# Both cuts sit well past the real 99th percentile: recipe names run to ~70
# characters, ingredients to ~85. Anything longer is prose that leaked in.
MAX_RECIPE_NAME = 90
MAX_INGREDIENT = 90

EMPTY = """
SELECT r.id, r.name FROM recipes r
WHERE NOT EXISTS (SELECT 1 FROM recipe_ingredients ri WHERE ri.recipe_id = r.id)
ORDER BY r.id
"""

LONG_NAME = """
SELECT id, name FROM recipes WHERE length(name) > ? ORDER BY length(name) DESC
"""

LONG_INGREDIENT = """
SELECT id, description FROM ingredients WHERE length(description) > ?
ORDER BY length(description) DESC
"""

# ponytail: no ON DELETE CASCADE in DuckDB, so every child table is named here.
# Add a table that references recipes or ingredients and you must add it below.
RECIPE_CHILDREN = [
    "DELETE FROM recipe_ingredients WHERE recipe_id = ?",
    "DELETE FROM recipe_requires WHERE recipe_id = ? OR requires_id = ?",
    "DELETE FROM recipe_pairings WHERE recipe_id = ? OR paired_id = ?",
]
INGREDIENT_CHILDREN = [
    "DELETE FROM recipe_ingredients WHERE ingredient_id = ?",
    "DELETE FROM embed_training_data WHERE ingredient_id = ?",
]


def find(con, max_name=MAX_RECIPE_NAME, max_ingredient=MAX_INGREDIENT):
    """[(table, label, [(id, text)])], empty lists kept so the report says "0"."""
    return [
        ("recipes", "recipes with no ingredients", con.execute(EMPTY).fetchall()),
        ("recipes", f"recipe names over {max_name} chars",
         con.execute(LONG_NAME, [max_name]).fetchall()),
        ("ingredients", f"ingredients over {max_ingredient} chars",
         con.execute(LONG_INGREDIENT, [max_ingredient]).fetchall()),
    ]


def delete(con, found):
    """Drop the flagged recipes and ingredients, children first."""
    recipes = {i for t, _, rows in found if t == "recipes" for i, _ in rows}
    ingredients = {i for t, _, rows in found if t == "ingredients" for i, _ in rows}
    # a recipe stripped of its junk ingredient may now have none left, but that
    # is the next run's problem -- deciding it here would delete two rows for
    # one flag and surprise whoever read the report
    for sql in RECIPE_CHILDREN:
        con.executemany(sql, [(i,) * sql.count("?") for i in recipes])
    con.executemany("DELETE FROM recipes WHERE id = ?", [(i,) for i in recipes])
    for sql in INGREDIENT_CHILDREN:
        con.executemany(sql, [(i,) for i in ingredients])
    con.executemany("DELETE FROM ingredients WHERE id = ?", [(i,) for i in ingredients])
    return len(recipes), len(ingredients)


def main(db="recipes.db", max_name=MAX_RECIPE_NAME, max_ingredient=MAX_INGREDIENT,
         limit=20, delete_them=False):
    """Report the flagged rows.

    limit is rows printed per category; delete_them (--delete-them) removes
    them instead.
    """
    con = duckdb.connect(db, read_only=not delete_them)
    found = find(con, max_name, max_ingredient)

    for _, label, rows in found:
        print(f"\n{len(rows)} {label}")
        for rid, text in rows[:limit]:
            print(f"  {rid:>6}  {' '.join(text.split())[:100]}")
        if len(rows) > limit:
            print(f"  ... {len(rows) - limit} more")

    if delete_them:
        r, i = delete(con, found)
        print(f"\ndeleted {r} recipes and {i} ingredients from {db}")
    else:
        print("\nnothing changed; pass --delete-them to remove these")
    con.close()


if __name__ == "__main__":
    fire.Fire(main)
