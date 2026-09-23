from gui import list_html, menu_html
from plan import Plan, Recipe

PLAN = Plan(
    recipes=[
        Recipe(1, "Dal", 40, False, None, ["1 cup urad dal"]),
        Recipe(2, "Raita", None, True, 1, ["yogurt"]),
        Recipe(3, "Fried <garlic>", 5, True, 2, []),
    ],
    by_store={"Little India: lentils": ["urad dal"], None: ["yogurt"]},
)


def test_menu_lists_titles_only():
    html = menu_html(PLAN)
    assert "<li><b>Dal</b> with <b>Raita</b></li>" in html
    assert "urad dal" not in html and "40 min" not in html, "titles only"


def test_shopping_list_names_the_orphan_bucket():
    html = list_html(PLAN)
    assert "Little India" in html and "lentils" not in html, "heading is the name only"
    assert "No store stocks this" in html
    assert html.index("Little India") < html.index("No store"), "orphans sort last"


def test_empty_plan_says_so():
    assert "empty" in menu_html(Plan([], {}))
    assert "Nothing to buy" in list_html(Plan([], {}))


def test_store_text_round_trips(tmp_path, monkeypatch):
    import pathlib

    import duckdb
    import gui
    db = str(tmp_path / "t.db")
    con = duckdb.connect(db)
    con.execute((pathlib.Path(__file__).resolve().parent.parent / "recipes.sql").read_text())
    con.close()
    monkeypatch.setattr(gui.settings, "DB", db)
    calls = []
    monkeypatch.setattr(gui.store_llm, "embed",
                        lambda ts: calls.append(ts) or [[0.0] * 768 for _ in ts])

    gui.save_stores("  Little India: lentils \n\n Co-op: everything else\n")
    assert gui.load_stores() == "Little India: lentils\nCo-op: everything else"
    gui.save_stores("Little India: lentils\nCo-op: everything else")
    assert len(calls) == 1, "unchanged text does not re-embed"
    gui.save_stores("Co-op: everything else")
    assert gui.load_stores() == "Co-op: everything else", "removals stick"


def test_unembedded_rows_are_backfilled(tmp_path, monkeypatch):
    import pathlib

    import duckdb
    import gui
    db = str(tmp_path / "t.db")
    con = duckdb.connect(db)
    con.execute((pathlib.Path(__file__).resolve().parent.parent / "recipes.sql").read_text())
    # a row written before stores had an embedding column
    con.execute("insert into stores (description, priority) values ('Co-op: all of it', 0)")
    con.close()
    monkeypatch.setattr(gui.settings, "DB", db)
    monkeypatch.setattr(gui.store_llm, "embed", lambda ts: [[0.0] * 768 for _ in ts])

    gui.save_stores("Co-op: all of it")
    with duckdb.connect(db, read_only=True) as con:
        assert con.execute("select count(*) from stores"
                           " where embedding is null").fetchone() == (0,)
