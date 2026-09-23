import pathlib

import duckdb

SCHEMA = pathlib.Path(__file__).resolve().parent.parent / "recipes.sql"


def fixture():
    con = duckdb.connect()
    con.execute(SCHEMA.read_text())
    con.execute("""
        INSERT INTO recipes (id, name, last_cooked, upvotes, downvotes, is_side) VALUES
            (1, 'beloved stew',   NULL,         40, 0, FALSE),
            (2, 'fine noodles',   '2000-01-01', 30, 1, FALSE),
            (3, 'cooked monday',  current_date, 99, 0, FALSE),  -- too recent
            (4, 'green sauce',    NULL,         10, 0, TRUE),   -- a side
            (5, 'hated thing',    NULL,          0, 50, FALSE);
        INSERT INTO recipes (id, name, upvotes, is_side) VALUES
            (6, 'chili oil',  5, TRUE),   -- the noodles need it
            (7, 'fried garlic', 5, TRUE); -- and the chili oil needs that
        INSERT INTO recipe_pairings VALUES (1, 4);  -- stew pairs with the sauce
        INSERT INTO recipe_requires VALUES (2, 6), (6, 7);
        INSERT INTO ingredients (id, description) VALUES (1, 'beef'), (2, 'parsley');
        INSERT INTO recipe_ingredients VALUES (1, 1, '2 lb'), (4, 2, NULL);
    """)
    return con


def test_excludes_recent_and_sides():
    con = fixture()
    for _ in range(20):
        ids = {r[0] for r in con.execute("SELECT * FROM plan_week(2)").fetchall()}
        assert 3 not in ids, "cooked today"
        assert 4 not in ids or 1 in ids, "side only ever shows up as a pairing"


def test_pairings_and_ingredients():
    con = fixture()
    # ask for everything eligible, so the picks are deterministic
    rows = {r[0]: r for r in con.execute("SELECT * FROM plan_week(3)").fetchall()}
    assert set(rows) == {1, 2, 5, 4, 6, 7}, rows
    assert rows[4][4] == 1, "sauce comes along for the stew"
    assert rows[6][4] == 2, "chili oil comes along for the noodles"
    assert rows[7][4] == 6, "and its own component follows it"
    assert rows[1][4] is None, "stew is a main pick"
    assert rows[1][5] == ["2 lb beef"]
    assert rows[4][5] == ["parsley"]
    assert rows[2][5] is None, "no ingredients recorded"


def test_thompson_prefers_the_liked_recipe():
    con = fixture()
    first = [con.execute("SELECT recipe_id FROM plan_week(1)").fetchone()[0] for _ in range(50)]
    assert first.count(5) <= 2, f"50-0 downvoted recipe kept winning: {first}"
    assert len(set(first)) > 1, "sampling is not actually random"
