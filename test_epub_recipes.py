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
    units = list(E.chunks(path))
    assert units and all(text.strip() for _, _, text in units)
    # The index is a list of recipe names with no method -- it must not reach the LLM.
    assert not any("ndex" in href for href, _, _ in units)


def test_anchors_split_a_multi_recipe_document():
    """Bittman keeps 404 recipes in 4 documents; ToC anchors must cut them apart."""
    path = next(p for p in BOOKS if "Bittman" in p)
    units = list(E.chunks(path))
    assert len(units) > 300, f"anchor splitting did not fire: {len(units)} chunks"
    assert max(len(t) for _, _, t in units) < 20_000
