import duckdb
import pytest

import ingredient_expand as ie


class FakeResponse:
    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self.content}}]}


@pytest.mark.parametrize("raw", [
    " split black lentil, South Asian pulse",
    "- split black lentil, South Asian pulse",
])
def test_expand_strips_the_arrow_leftovers(monkeypatch, raw):
    monkeypatch.setattr(ie.requests, "post", lambda *a, **k: FakeResponse(raw))
    assert ie.expand("urad dal") == "urad dal - split black lentil, South Asian pulse"


def test_only_uncached_rows_are_selected():
    con = duckdb.connect()
    con.execute(open("recipes.sql").read())
    con.execute("INSERT INTO ingredients (description) VALUES ('urad dal'), ('garlic')")
    con.execute("INSERT OR REPLACE INTO ingredient_expansions VALUES (1, 'urad dal - x')")
    rows = con.execute(
        "SELECT i.description FROM ingredients i "
        "ANTI JOIN ingredient_expansions e ON e.ingredient_id = i.id").fetchall()
    assert rows == [("garlic",)]
