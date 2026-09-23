"""Turn cookbook epubs into rows in the recipe database.

Boundaries come from the epub's own table of contents: leaf entries are the
recipe boundaries, and the unit of text is the spine document (or the slice of
it between two ToC anchors). Books whose ToC is only chapter-level yield
chapter-sized units; the model pass splits those into individual recipes. No
per-book parsing. Only the ToC and the XHTML spine are read, never images.

    uv run python -m epub_extract example_cookbooks/*.epub --limit=5

--limit caps units per book, which is how you iterate on the prompt cheaply.
"""
import html as _html
import json
import os
import posixpath
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from typing import NamedTuple

import duckdb
import fire
import requests

import settings

# ---------------------------------------------------------------- epub layout

def tag(el):
    return el.tag.rsplit("}", 1)[-1]


def find_all(root, name):
    return [el for el in root.iter() if tag(el) == name]


def text_of(el):
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def strip_tags(markup):
    return re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]+>", " ", markup))).strip()


def resolve_href(base_href, src):
    """Resolve an href against the file containing it. Returns (path, fragment)."""
    src = src or ""
    frag = src.split("#")[1] if "#" in src else ""
    path = src.split("#")[0]
    if not path:
        return base_href, frag
    return posixpath.normpath(posixpath.join(posixpath.dirname(base_href), path)), frag


def load_book(path):
    """Return (zip, manifest{id: (href, media, props)}, spine[hrefs], ncx_id, opf_href)."""
    z = zipfile.ZipFile(path)
    container = ET.fromstring(z.read("META-INF/container.xml"))
    opf = find_all(container, "rootfile")[0].get("full-path") or ""
    base = posixpath.dirname(opf)
    root = ET.fromstring(z.read(opf))

    manifest = {}
    for it in find_all(root, "item"):
        href = posixpath.normpath(posixpath.join(base, it.get("href", "")))
        manifest[it.get("id")] = (href, it.get("media-type", ""), it.get("properties") or "")

    spine_el = find_all(root, "spine")[0]
    spine = [manifest[r.get("idref")][0] for r in find_all(spine_el, "itemref")
             if r.get("idref") in manifest]
    return z, manifest, spine, spine_el.get("toc"), opf


def nav_href(manifest):
    return next((h for h, _, p in manifest.values() if "nav" in p.split()), None)


def toc_entries(z, manifest, ncx_id):
    """(depth, title, href, fragment) for every ToC entry, in reading order."""
    nav = nav_href(manifest)
    if nav:
        root = ET.fromstring(z.read(nav))
        navs = find_all(root, "nav")
        toc = next((n for n in navs if (n.get("{http://www.idpf.org/2007/ops}type")
                                        or "").strip() == "toc"), navs[0] if navs else root)
        out = []

        def walk(ol, depth):
            for li in [c for c in ol if tag(c) == "li"]:
                a = next((e for e in li.iter() if tag(e) in ("a", "span")), None)
                if a is not None and text_of(a):
                    href, frag = resolve_href(nav, a.get("href"))
                    out.append((depth, text_of(a), href, frag))
                for sub in [c for c in li if tag(c) == "ol"]:
                    walk(sub, depth + 1)

        for ol in [c for c in toc if tag(c) == "ol"]:
            walk(ol, 0)
        return out

    if not (ncx_id and ncx_id in manifest):
        return []
    ncx = manifest[ncx_id][0]
    root = ET.fromstring(z.read(ncx))
    out = []

    def walk_ncx(el, depth):
        for np in [c for c in el if tag(c) == "navPoint"]:
            label = next((text_of(l) for l in np.iter() if tag(l) == "text"), "")
            src = next((c.get("src") for c in np.iter() if tag(c) == "content"), "")
            if label:
                href, frag = resolve_href(ncx, src)
                out.append((depth, label, href, frag))
            walk_ncx(np, depth + 1)

    walk_ncx(find_all(root, "navMap")[0], 0)
    return out


# Standard epub structural roles that never contain recipes. The index is the
# important one: a long list of recipe names with no ingredients or method,
# i.e. prime material for the model to hallucinate recipes from.
SKIP_TYPES = {"index", "toc", "cover", "titlepage", "copyright-page", "colophon",
              "dedication", "acknowledgments", "bibliography", "glossary", "loi", "lot"}
# ponytail: "backmatter" is here because that is how some books tag their index.
# A recipe in back matter would be dropped -- none of the sample books do that;
# narrow this to "index" alone if one turns up that does.
SKIP_DOC_TYPES = SKIP_TYPES | {"doc-index", "backmatter"}
DOC_TYPE = re.compile(r'(?:epub:type|role)="([^"]+)"', re.I)


def skip_hrefs(z, manifest, opf):
    """Documents the book's own metadata flags as front/back matter."""
    skip = set()
    # epub2: <guide><reference type="index" href="..."/>
    for ref in find_all(ET.fromstring(z.read(opf)), "reference"):
        if (ref.get("type") or "").lower() in SKIP_TYPES:
            skip.add(resolve_href(opf, ref.get("href"))[0])

    # epub3: <nav epub:type="landmarks"><li><a epub:type="index" href="..."/>
    nav = nav_href(manifest)
    if nav:
        for n in find_all(ET.fromstring(z.read(nav)), "nav"):
            if (n.get("{http://www.idpf.org/2007/ops}type") or "") != "landmarks":
                continue
            for a in find_all(n, "a"):
                t = (a.get("{http://www.idpf.org/2007/ops}type") or "").lower()
                if any(part in SKIP_TYPES for part in t.split()):
                    skip.add(resolve_href(nav, a.get("href"))[0])
        skip.add(nav)
    return skip


# ------------------------------------------------------------------ chunking

ANY_ID = re.compile(r'(?:id|name)="([^"]+)"')
LINK = re.compile(r'<a\b[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.S | re.I)


class Chunk(NamedTuple):
    id: int
    href: str
    anchor: str          # "doc.xhtml#frag", or just "doc.xhtml" for a whole document
    title: str
    text: str
    links: dict          # marker number -> (target href, target fragment)


def parse(path):
    """Return (units, resolve) for a book.

    A unit is a single recipe wherever the ToC provides in-document anchors,
    and the whole document otherwise. `resolve(href, frag)` maps a link target
    back to the id of the unit containing it, or None.
    """
    z, manifest, spine, ncx_id, opf = load_book(path)
    entries = toc_entries(z, manifest, ncx_id)
    skip = skip_hrefs(z, manifest, opf)

    anchored, doc_titles = {}, {}
    for _, title, href, frag in entries:
        if frag:
            anchored.setdefault(href, []).append((title, frag))
        else:
            doc_titles.setdefault(href, title)

    units, next_id = [], [0]
    ranges = {}      # href -> [(start, end, unit id)]
    id_offsets = {}  # href -> {anchor name: byte offset}

    def add(href, raw, start, end, title, frag=""):
        links = {}

        def mark(m):
            """<a> becomes [[R<n>]]: the marker carries the reference, the anchor
            text does not -- books overwhelmingly write "here" or "this page"."""
            target, inner = m.group(1), m.group(2)
            if target.startswith(("http:", "https:", "mailto:")):
                return inner
            n = next_id[0]
            next_id[0] += 1
            links[n] = resolve_href(href, target)
            return f" [[R{n}]] {inner} "

        text = strip_tags(LINK.sub(mark, raw[start:end]))
        if not text:
            return
        units.append(Chunk(len(units), href, f"{href}#{frag}" if frag else href,
                           title, text, links))
        ranges.setdefault(href, []).append((start, end, units[-1].id))

    for href in spine:
        if href in skip or not href.endswith((".xhtml", ".html", ".htm")):
            continue
        try:
            raw = z.read(href).decode("utf8", "replace")
        except Exception:
            continue
        if {t for m in DOC_TYPE.findall(raw[:4000]) for t in m.lower().split()} & SKIP_DOC_TYPES:
            continue
        id_offsets[href] = {m.group(1): m.start() for m in ANY_ID.finditer(raw)}

        # ponytail: split on the ToC's own anchors, never on byte counts -- a
        # size-based cut would slice a recipe in half. Cut at the start of the
        # enclosing tag, not at the id attribute, or the split lands mid-tag.
        cuts = sorted((raw.rfind("<", 0, m.start()), title, frag)
                      for title, frag in anchored.get(href, [])
                      for m in [re.search(r'(?:id|name)="%s"' % re.escape(frag), raw)] if m)
        if not cuts:
            add(href, raw, 0, len(raw), doc_titles.get(href, ""))
            continue
        add(href, raw, 0, cuts[0][0], doc_titles.get(href, ""))
        for i, (start, title, frag) in enumerate(cuts):
            end = cuts[i + 1][0] if i + 1 < len(cuts) else len(raw)
            add(href, raw, start, end, title, frag)

    def resolve(href, frag):
        spans = ranges.get(href)
        if not spans:
            return None
        offset = id_offsets.get(href, {}).get(frag) if frag else None
        for start, end, uid in spans if offset is not None else []:
            if start <= offset < end:
                return uid
        return spans[0][2]

    return units, resolve


# --------------------------------------------------------------------- model

# A unit may hold zero recipes (front matter, an essay) or several, so the
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

    # Nothing in the grammar stops the model looping out the same recipe over
    # and over -- it satisfies the schema perfectly while being wrong. Keep the
    # first of each name, and drop recipes the model failed to name at all.
    seen, unique = set(), []
    for rec in json.loads(choice["message"]["content"])["recipes"]:
        key = (rec.get("name") or "").strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(rec)
    return unique


# ------------------------------------------------------------------ database

LINK_TABLES = {"requires": ("recipe_requires", "requires_id"),
               "pairs_with": ("recipe_pairings", "paired_id")}


def store(con, recipe, source_filename, anchor, source_text):
    recipe_id = con.execute(
        "INSERT INTO recipes (cooktime, name, filename, anchor, is_side) "
        "VALUES (?, ?, ?, ?, ?) RETURNING id",
        (recipe.get("cooktime"), recipe["name"], source_filename, anchor,
         bool(recipe.get("is_side")))).fetchone()[0]
    kept = 0
    for ing in recipe.get("ingredients", []):
        desc = re.sub(r"\s+", " ", (ing.get("description") or "")).strip().lower()
        # ponytail: substring match against the source. Catches the common
        # failure -- the model recognising a dish and reciting its standard
        # ingredient list. Upgrade path: stem the words if plurals cause noise.
        if not desc or desc not in source_text.lower():
            continue
        con.execute("INSERT INTO ingredients (description) VALUES (?) "
                    "ON CONFLICT (description) DO NOTHING", (desc,))
        ing_id = con.execute(
            "SELECT id FROM ingredients WHERE description = ?", (desc,)).fetchone()[0]
        con.execute(
            "INSERT INTO recipe_ingredients (recipe_id, ingredient_id, quantity) "
            "VALUES (?, ?, ?)", (recipe_id, ing_id, ing.get("quantity")))
        kept += 1
    return recipe_id, kept


def main(*books, limit=0, max_chars=24000):
    """Extract recipes from cookbook epubs into the database.

    books:     epub paths
    limit:     units per book (0 = all)
    max_chars: truncate a unit before sending; the intro essays are huge
    """
    con = duckdb.connect(settings.DB)
    totals = dict.fromkeys(("units", "recipes", "ingredients", "links"), 0)

    for book in books:
        name = os.path.basename(book)
        print(f"\n=== {name}", flush=True)
        try:
            units, resolve = parse(book)
        except Exception as e:
            # One unreadable book must not abort a whole shelf.
            print(f"  {type(e).__name__}: {e}", file=sys.stderr, flush=True)
            continue
        markers = {n: target for u in units for n, target in u.links.items()}
        unit_recipes, pending = {}, []

        for unit in units[:limit] if limit else units:
            totals["units"] += 1
            try:
                recipes = call_model(unit.text, unit.title, max_chars)
            except Exception as e:
                print(f"  [{unit.id}] {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                continue

            for rec in recipes:
                recipe_id, kept = store(con, rec, name, unit.anchor, unit.text)
                unit_recipes.setdefault(unit.id, []).append(recipe_id)
                for kind in LINK_TABLES:
                    # Only markers that really occur in this unit; the model is
                    # told not to invent numbers, but it is cheap to enforce.
                    pending += [(recipe_id, n, kind) for n in rec.get(kind, [])
                                if n in unit.links]
                totals["recipes"] += 1
                totals["ingredients"] += kept
                print(f"  [{unit.id}] {rec['name']}{' [side]' if rec.get('is_side') else ''} "
                      f"({kept} ingredients, cooktime={rec.get('cooktime')})", flush=True)
            con.commit()

        # Links are written last: one routinely points at a recipe that had not
        # been extracted yet when it was seen.
        for recipe_id, marker, kind in pending:
            unit = resolve(*markers[marker]) if marker in markers else None
            table, column = LINK_TABLES[kind]
            for other_id in unit_recipes.get(unit, []):
                if other_id == recipe_id:
                    continue  # a recipe referring to its own section
                con.execute(f"INSERT INTO {table} (recipe_id, {column}) VALUES (?, ?) "
                            "ON CONFLICT DO NOTHING", (recipe_id, other_id))
                totals["links"] += 1
        con.commit()

    con.close()
    print(f"\n{totals['units']} units -> {totals['recipes']} recipes, "
          f"{totals['ingredients']} ingredients, {totals['links']} links -> {settings.DB}")


if __name__ == "__main__":
    fire.Fire(main)
