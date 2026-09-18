"""Smoke check for the boundary finder. Run: uv run pytest -x -s"""
import glob

import pytest

import epub_recipes as E

BOOKS = sorted(glob.glob("example_cookbooks/*.epub"))


@pytest.mark.parametrize("path", BOOKS, ids=lambda p: p.rsplit("/", 1)[-1][:20])
def test_toc_boundaries(path):
    recipes = E.find_recipes(path)
    assert recipes, "no ToC boundaries found"
    assert all(t.strip() and h for t, h, _ in recipes)
    assert not any(E.NOT_RECIPE.match(t) for t, _, _ in recipes)


@pytest.mark.parametrize("path", BOOKS, ids=lambda p: p.rsplit("/", 1)[-1][:20])
def test_chunks_are_split_and_filtered(path):
    units = E.chunks(path)
    assert units and all(u.text.strip() for u in units)
    # The index is a list of recipe names with no method -- it must not reach the LLM.
    assert not any("ndex" in u.href for u in units)


@pytest.mark.parametrize("path", BOOKS, ids=lambda p: p.rsplit("/", 1)[-1][:20])
def test_links_are_marked_and_resolvable(path):
    units, resolve = E.parse(path)
    marked = [u for u in units if u.links]
    assert marked, "no cross-recipe links found"
    for u in marked:
        for n in u.links:
            assert f"[[R{n}]]" in u.text, "marker missing from the text sent to the model"
    # Markers are unique book-wide, so a model citing one identifies one target.
    seen = [n for u in units for n in u.links]
    assert len(seen) == len(set(seen))
    hit = sum(1 for u in units for t in u.links.values() if resolve(*t) is not None)
    assert hit / len(seen) > 0.9, f"only {hit}/{len(seen)} links resolved"


def test_anchors_split_a_multi_recipe_document():
    """Bittman keeps 404 recipes in 4 documents; ToC anchors must cut them apart."""
    path = next(p for p in BOOKS if "Bittman" in p)
    units = E.chunks(path)
    assert len(units) > 300, f"anchor splitting did not fire: {len(units)} chunks"
    assert max(len(u.text) for u in units) < 20_000
