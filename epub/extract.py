"""Extract recipes from cookbook epubs with a local llama.cpp model.

Boundaries come from epub_recipes.chunks(); this module turns each chunk into
rows in the SQLite database defined by recipes.sql.

    llama serve --models-preset ~/llamacpp/llama-models.ini   # in another terminal
    uv run python extract.py example_cookbooks/*.epub --limit=5

The server must run with LLAMA_CACHE=/Users/sam/llamacpp/models -- the directory
directly holding the models--<org>--<repo> folders. Point it at a parent and the
cache reads as empty and re-downloads instead of failing.

--limit caps chunks per book, which is how you iterate on the prompt cheaply.
"""
import argparse
import json
import os
import re
import sys

import duckdb
import requests

import epub_recipes
import settings

SCHEMA_FILE = "recipes.sql"

# A chunk may hold zero recipes (front matter, an essay) or several, so the
# grammar always returns a list. Nothing but this shape can be emitted.
#
# llama.cpp will not parse a rule body that spans lines unless it is wrapped in
# parentheses -- without them the server rejects the whole grammar with a 400
# "failed to parse grammar". Keep the parens on `recipe` and `ingredient`.
GRAMMAR = r"""
root        ::= "{" ws "\"recipes\":" ws "[" ws recipes? ws "]" ws "}"
recipes     ::= recipe (ws "," ws recipe)*
recipe      ::= (
                "{" ws
                "\"name\":" ws string ws "," ws
                "\"cooktime\":" ws (number | "null") ws "," ws
                "\"is_side\":" ws boolean ws "," ws
                "\"requires\":" ws numbers ws "," ws
                "\"pairs_with\":" ws numbers ws "," ws
                "\"ingredients\":" ws "[" ws ingredients? ws "]" ws
                "}"
                )
ingredients ::= ingredient (ws "," ws ingredient)*
ingredient  ::= (
                "{" ws
                "\"description\":" ws string ws "," ws
                "\"quantity\":" ws (string | "null") ws
                "}"
                )
numbers     ::= "[" ws (number (ws "," ws number)*)? ws "]"
boolean     ::= "true" | "false"
string      ::= "\"" char* "\""
char        ::= [^"\\] | "\\" ["\\bfnrt/] | "\\u" hex hex hex hex
hex         ::= [0-9a-fA-F]
number      ::= [0-9]+
ws          ::= [ \t\n]*
"""

PROMPT = """You are reading one section of a cookbook. Extract every recipe in it.

Rules:
- If this section contains no recipe (an introduction, an essay, a headnote \
with no method), return an empty list. Do not invent a recipe.
- "description" is the bare ingredient, canonical and singular: "garlic", \
"soy sauce", "chicken thigh". No amounts, no preparation.
- "quantity" holds everything else about the amount, verbatim from the text: \
"2 cloves, minced", "a splash", "1 pound, cut into 1-inch cubes". Use null if \
the text gives no amount.
- "cooktime" is total time in whole minutes. Use null if the text does not \
state a time. Never estimate a time yourself.

Example: "toss in a couple of smashed garlic cloves" becomes
{{"description": "garlic", "quantity": "a couple of cloves, smashed"}}

"is_side" is true only for something not eaten on its own: a sauce, dressing, \
spice blend, pickle, condiment or side dish. A recipe for a protein component \
-- roasted tofu, velveted chicken, seared steak -- is NOT a side, even when it \
is served as part of something else. A complete dish is not a side.

The text contains markers like [[R12]], each one a cross-reference to another \
recipe in this book. For every marker that appears, put its number in exactly \
one of these lists:
- "requires": the other recipe is a component of this one -- its sauce, stock, \
spice mix or dough. You must make it in order to make this recipe.
- "pairs_with": the other recipe is merely suggested alongside -- serve with, \
goes well with, try it over.
Leave out any marker that is neither, such as a reference to a technique, a \
photo, an ingredient note or a chapter. Never invent a number that is not in \
the text.

Section title: {title}

Section text:
{text}
"""


def call_model(text, title, max_chars):
    body = {
        "model": settings.CHAT_MODEL,
        "messages": [{"role": "user",
                      "content": PROMPT.format(title=title or "(untitled)",
                                               text=text[:max_chars])}],
        "grammar": GRAMMAR,
        "temperature": 0,
        "max_tokens": 4096,
    }
    r = requests.post(f"{settings.SERVER}/v1/chat/completions", json=body, timeout=600)
    r.raise_for_status()
    choice = r.json()["choices"][0]
    # The grammar guarantees the shape but not that generation finished: hitting
    # the token limit truncates mid-string and json.loads fails with a confusing
    # error. Say what actually happened instead.
    if choice.get("finish_reason") == "length":
        raise RuntimeError(f"hit max_tokens ({body['max_tokens']}); output truncated")
    recipes = json.loads(choice["message"]["content"])["recipes"]

    # Nothing in the grammar stops the model looping out the same recipe over
    # and over -- it satisfies the schema perfectly while being wrong. Keep the
    # first of each name.
    seen, unique = set(), []
    for rec in recipes:
        key = (rec.get("name") or "").strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(rec)
    return unique


def grounded(ingredient, text):
    """Did this ingredient actually appear in the source, or did the model invent it?

    ponytail: substring match on the bare description. Catches the common
    failure -- the model recognising a dish and reciting its standard
    ingredient list. Upgrade path: stem the words if plurals cause noise.
    """
    return ingredient.lower() in text.lower()


def connect(db_path):
    fresh = not os.path.exists(db_path)
    con = duckdb.connect(db_path)
    if fresh:
        con.execute(open(SCHEMA_FILE).read())
    return con


def store(con, recipe, source_filename, anchor=None):
    recipe_id = con.execute(
        "INSERT INTO recipes (cooktime, name, filename, anchor, is_side) "
        "VALUES (?, ?, ?, ?, ?) RETURNING id",
        (recipe.get("cooktime"), recipe["name"], source_filename, anchor,
         bool(recipe.get("is_side")))).fetchone()[0]
    for ing in recipe.get("ingredients", []):
        desc = re.sub(r"\s+", " ", (ing.get("description") or "")).strip().lower()
        if not desc:
            continue
        con.execute("INSERT INTO ingredients (description) VALUES (?) "
                    "ON CONFLICT (description) DO NOTHING", (desc,))
        ing_id = con.execute(
            "SELECT id FROM ingredients WHERE description = ?", (desc,)).fetchone()[0]
        con.execute(
            "INSERT INTO recipe_ingredients (recipe_id, ingredient_id, quantity) "
            "VALUES (?, ?, ?)", (recipe_id, ing_id, ing.get("quantity")))
    return recipe_id


TABLES = {"requires": ("recipe_requires", "requires_id"),
          "pairs_with": ("recipe_pairings", "paired_id")}


def link_up(con, pending, unit_recipes, markers, resolve):
    """Turn (recipe, marker, kind) references into join-table rows.

    Deferred to the end of a book because a link routinely points at a recipe
    that has not been extracted yet.
    """
    written = {"requires": 0, "pairs_with": 0}
    for recipe_id, marker, kind in pending:
        target = markers.get(marker)
        if target is None:
            continue
        unit = resolve(*target)
        table, column = TABLES[kind]
        for other_id in unit_recipes.get(unit, []):
            if other_id == recipe_id:
                continue  # a recipe referring to its own section
            con.execute(
                f"INSERT INTO {table} (recipe_id, {column}) VALUES (?, ?) "
                "ON CONFLICT DO NOTHING", (recipe_id, other_id))
            written[kind] += 1
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("books", nargs="+")
    ap.add_argument("--limit", type=int, default=0, help="chunks per book (0 = all)")
    ap.add_argument("--max-chars", type=int, default=24000,
                    help="truncate a chunk before sending; the intro essays are huge")
    args = ap.parse_args()

    con = connect(settings.DB)
    totals = {"chunks": 0, "recipes": 0, "ingredients": 0, "ungrounded": 0,
              "requires": 0, "pairs_with": 0}

    for book in args.books:
        name = os.path.basename(book)
        print(f"\n=== {name}", flush=True)
        units, resolve = epub_recipes.parse(book)
        markers = {n: target for u in units for n, target in u.links.items()}
        unit_recipes, pending = {}, []

        for unit in units[:args.limit] if args.limit else units:
            totals["chunks"] += 1
            try:
                recipes = call_model(unit.text, unit.title, args.max_chars)
            except Exception as e:
                print(f"  [{unit.id}] {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                continue

            for rec in recipes:
                if not (rec.get("name") or "").strip():
                    continue
                ings = rec.get("ingredients", [])
                bad = [g["description"] for g in ings
                       if not grounded(g.get("description") or "", unit.text)]
                recipe_id = store(con, rec, name, unit.anchor)
                unit_recipes.setdefault(unit.id, []).append(recipe_id)
                for kind in TABLES:
                    # Only markers that really occur in this unit; the model is
                    # told not to invent numbers, but it is cheap to enforce.
                    pending += [(recipe_id, n, kind) for n in rec.get(kind, [])
                                if n in unit.links]
                totals["recipes"] += 1
                totals["ingredients"] += len(ings)
                totals["ungrounded"] += len(bad)
                flag = f"  !! not in source: {bad}" if bad else ""
                side = " [side]" if rec.get("is_side") else ""
                print(f"  [{unit.id}] {rec['name']}{side} "
                      f"({len(ings)} ingredients, cooktime={rec.get('cooktime')}){flag}",
                      flush=True)
            con.commit()

        written = link_up(con, pending, unit_recipes, markers, resolve)
        con.commit()
        for kind in TABLES:
            totals[kind] += written[kind]
        print(f"  links: {written['requires']} requires, "
              f"{written['pairs_with']} pairings", flush=True)

    con.close()
    print(f"\n{totals['chunks']} chunks -> {totals['recipes']} recipes, "
          f"{totals['ingredients']} ingredients "
          f"({totals['ungrounded']} not found in source text), "
          f"{totals['requires']} requires, {totals['pairs_with']} pairings -> {settings.DB}")


if __name__ == "__main__":
    main()
