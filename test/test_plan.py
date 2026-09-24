import pathlib

import duckdb
import numpy as np

from plan import shopping_plan

SCHEMA = pathlib.Path(__file__).resolve().parent.parent / "recipes.sql"

STORES = ["Little India: lentils and spices", "Cub Foods: ordinary supermarket"]

DIM = 768


def vec(a, b):
    """A 768-wide vector that only uses its first two dimensions."""
    return [float(a), float(b)] + [0.0] * (DIM - 2)


def db(tmp_path, store_vecs=(vec(1, -1), vec(-1, 1))):
    path = str(tmp_path / "t.db")
    con = duckdb.connect(path)
    con.execute(SCHEMA.read_text())
    con.execute("""
        INSERT INTO recipes (id, name, upvotes, is_side) VALUES
            (1, 'dal', 10, FALSE), (2, 'raita', 5, TRUE);
        INSERT INTO recipe_pairings VALUES (1, 2);
        INSERT INTO ingredients (id, description, expansion) VALUES
            (1, 'urad dal', 'urad dal - split black lentil'), (2, 'yogurt', NULL);
        INSERT INTO recipe_ingredients VALUES (1, 1, '1 cup'), (2, 2, NULL);
    """)
    con.executemany(
        "INSERT INTO stores (description, priority, embedding) VALUES (?, ?, ?)",
        [(s, i, v) for i, (s, v) in enumerate(zip(STORES, store_vecs))])
    con.close()
    return path


def fake_embed(glosses):
    """Dimension 0 is "is a lentil", dimension 1 is "is anything else"."""
    return np.array([vec(1, 0) if "lentil" in g else vec(0, 1) for g in glosses])


def test_splits_the_list_by_store(tmp_path):
    plan = shopping_plan(2, db=db(tmp_path), embed=fake_embed)
    assert [r.name for r in plan.recipes] == ["dal", "raita"]
    assert plan.recipes[1].for_recipe == 1, "raita rides along with the dal"
    assert plan.recipes[0].ingredients == ["1 cup urad dal"]
    assert plan.by_store == {STORES[0]: ["urad dal"], STORES[1]: ["yogurt"]}


def test_ungloss_ingredient_still_gets_a_store(tmp_path):
    # yogurt has no expansion; it must not vanish
    seen = []
    plan = shopping_plan(2, db=db(tmp_path),
                         embed=lambda g: seen.extend(g) or fake_embed(g))
    assert any("yogurt" in s for s in seen), "bare name is used when there is no gloss"
    assert sum(len(v) for v in plan.by_store.values()) == 2


def test_embeddings_are_computed_once(tmp_path):
    path = db(tmp_path)
    calls = []
    embed = lambda g: calls.append(len(g)) or fake_embed(g)
    shopping_plan(2, db=path, embed=embed)
    plan = shopping_plan(2, db=path, embed=embed)
    assert calls == [2], "second run reads the cached vectors"
    assert plan.by_store == {STORES[0]: ["urad dal"], STORES[1]: ["yogurt"]}


def test_nothing_stocks_it(tmp_path):
    # both stores point the other way, so nobody gets sent anywhere
    plan = shopping_plan(2, db=db(tmp_path, [vec(-1, -1), vec(-1, -1)]), embed=fake_embed)
    assert plan.by_store == {None: ["urad dal", "yogurt"]}


def test_first_store_that_clears_the_bar_wins(tmp_path):
    # both stores stock everything; India is listed first, so it gets the order
    plan = shopping_plan(2, db=db(tmp_path, [vec(1, 1), vec(2, 2)]), embed=fake_embed)
    assert plan.by_store == {STORES[0]: ["urad dal", "yogurt"]}
