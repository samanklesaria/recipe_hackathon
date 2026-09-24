import pathlib
import sys

import duckdb

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
from anomalies import delete, find  # noqa: E402

SCHEMA = pathlib.Path(__file__).resolve().parent.parent / "recipes.sql"

PROSE = "x" * 200


def db(tmp_path):
    con = duckdb.connect(str(tmp_path / "t.db"))
    con.execute(SCHEMA.read_text())
    con.execute(f"""
        INSERT INTO recipes (id, name) VALUES
            (1, 'dal'), (2, 'heading with no body'), (3, '{PROSE}');
        INSERT INTO ingredients (id, description) VALUES
            (1, 'urad dal'), (2, '{PROSE}');
        INSERT INTO training_stores (id, description) VALUES (1, 'a shop');
        INSERT INTO embed_training_data VALUES (2, 1, 'never');
        INSERT INTO recipe_ingredients VALUES (1, 1, '1 cup'), (3, 2, NULL);
        INSERT INTO recipe_pairings VALUES (1, 3);
        INSERT INTO recipe_requires VALUES (3, 1);
    """)
    return con


def test_flags_the_three_shapes(tmp_path):
    found = db(tmp_path)
    assert [[i for i, _ in rows] for _, _, rows in find(found)] == [[2], [3], [2]]


def test_delete_clears_the_children_too(tmp_path):
    con = db(tmp_path)
    assert delete(con, find(con)) == (2, 1)
    assert con.execute("SELECT id FROM recipes").fetchall() == [(1,)]
    assert con.execute("SELECT id FROM ingredients").fetchall() == [(1,)]
    for table in ("recipe_ingredients", "recipe_pairings", "recipe_requires",
                  "embed_training_data"):
        left = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        assert left == (1 if table == "recipe_ingredients" else 0), table
