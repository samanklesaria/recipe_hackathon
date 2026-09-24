import duckdb

from dedup_ingredients import apply_merges, find_merges, resolve


def test_find_merges():
    rows = [
        (1, "bacon", 40), (2, "crispy bacon", 1), (3, "mint", 20),
        (4, "fresh mint leaves", 1), (5, "olive oil", 30), (6, "oil", 1),
        (7, "chicken broth", 25), (8, "pepper", 50), (9, "bell pepper", 15),
        (10, "red bell pepper", 4), (11, "egg", 60), (12, "eggs", 5),
    ]
    got = {a[1]: c[1] for a, c in find_merges(rows)}
    assert got["crispy bacon"] == "bacon", got
    assert got["fresh mint leaves"] == "mint", got
    assert "olive oil" not in got, got  # "oil" is too rare to be canonical
    assert "chicken broth" not in got, got
    assert got["red bell pepper"] == "bell pepper", got  # most specific, not "pepper"
    assert got["eggs"] == "egg", got  # plural spelling folds into the common one


def test_resolve_chain():
    # chains collapse to the far end, not to a row that is itself about to go
    chain = [((2, "sharp cheddar", 1), (3, "cheddar", 5)), ((3, "cheddar", 5), (4, "cheese", 30))]
    assert resolve(chain) == {2: 4, 3: 4}


def test_apply_merges():
    con = duckdb.connect()
    con.execute("""
        create table ingredients (id integer primary key, description varchar, expansion varchar);
        create table recipe_ingredients (recipe_id integer, ingredient_id integer, quantity varchar);
        insert into ingredients values (1, 'bacon', null), (2, 'crispy bacon', 'bacon, but crispy');
        insert into recipe_ingredients values (7, 1, '2 slices'), (7, 2, '2 slices'), (8, 2, '1 lb');
    """)
    apply_merges(con, [((2, "crispy bacon", 2), (1, "bacon", 40))])
    assert con.execute("select id, description from ingredients").fetchall() == [(1, "bacon")]
    assert sorted(con.execute("select * from recipe_ingredients").fetchall()) == [
        (7, 1, "2 slices"), (8, 1, "1 lb")  # the duplicated row collapsed
    ]
