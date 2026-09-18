"""Find recipe boundaries in an epub cookbook.

Reads only the table of contents and the XHTML spine -- never decompresses images.

One rule for every book: the ToC's leaf entries are the recipe boundaries, and
the unit of text handed downstream is the spine document. Books whose ToC is
only chapter-level yield chapter-sized chunks; the LLM pass splits those into
individual recipes. No per-book parsing.

Usage:  uv run --no-project python epub_recipes.py <book.epub> [--limit=N]
"""
import html as _html
import posixpath
import re
import struct
import sys
import zipfile
import zlib
import xml.etree.ElementTree as ET
from typing import NamedTuple


# ---------------------------------------------------------------- zip access

class SalvageZip:
    """Reader for a zip whose central directory is truncated.

    ponytail: scans local file headers instead of the index. Only used when
    zipfile refuses the file (incomplete download). Slower to open, same API.
    """

    def __init__(self, path):
        self.data = open(path, "rb").read()
        end = self.data.find(b"PK\x01\x02")
        if end < 0:
            end = len(self.data)
        self.entries = {}
        for m in re.finditer(b"PK\x03\x04", self.data[:end]):
            o = m.start()
            try:
                fields = struct.unpack("<IHHHHHIIIHH", self.data[o:o + 30])
            except struct.error:
                continue
            comp, csize, nlen, elen = fields[3], fields[7], fields[9], fields[10]
            name = self.data[o + 30:o + 30 + nlen].decode("utf8", "replace")
            if not nlen or not name.isprintable():
                continue
            self.entries[name] = (o + 30 + nlen + elen, csize, comp)

    def namelist(self):
        return list(self.entries)

    def read(self, name):
        start, csize, comp = self.entries[name]
        # csize is 0 when a data descriptor is used; zlib stops at stream end anyway.
        blob = self.data[start:start + csize] if csize else self.data[start:]
        return blob if comp == 0 else zlib.decompressobj(-15).decompress(blob)


def open_epub(path):
    try:
        return zipfile.ZipFile(path)
    except zipfile.BadZipFile:
        return SalvageZip(path)


# ---------------------------------------------------------------- xml helpers

def tag(el):
    return el.tag.rsplit("}", 1)[-1]


def find_all(root, name):
    return [el for el in root.iter() if tag(el) == name]


def text_of(el):
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def strip_tags(markup):
    return re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]+>", " ", markup))).strip()


# ---------------------------------------------------------------- epub layout

def load_book(path):
    """Return (zip, manifest{id: (href, media, props)}, spine[hrefs], ncx_id, opf_href)."""
    z = open_epub(path)
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


def _resolve(base_href, src):
    """Resolve an href relative to the file that contains it. Keeps the fragment."""
    src = src or ""
    frag = src.split("#")[1] if "#" in src else ""
    path = src.split("#")[0]
    if not path:
        return base_href, frag
    return posixpath.normpath(posixpath.join(posixpath.dirname(base_href), path)), frag


def toc_entries(path):
    """Yield (depth, title, href, fragment) from the ToC, in reading order."""
    z, manifest, spine, ncx_id, _ = load_book(path)
    nav = next((h for h, _, p in manifest.values() if "nav" in p.split()), None)
    if nav:
        return z, spine, _parse_nav(z, nav)
    if ncx_id and ncx_id in manifest:
        return z, spine, _parse_ncx(z, manifest[ncx_id][0])
    return z, spine, []


def _parse_nav(z, nav_href):
    root = ET.fromstring(z.read(nav_href))
    navs = find_all(root, "nav")
    toc = next((n for n in navs
                if (n.get("{http://www.idpf.org/2007/ops}type") or "").strip() == "toc"), None)
    if toc is None:
        toc = navs[0] if navs else root

    out = []

    def walk(ol, depth):
        for li in [c for c in ol if tag(c) == "li"]:
            a = next((e for e in li.iter() if tag(e) in ("a", "span")), None)
            if a is not None and text_of(a):
                href, frag = _resolve(nav_href, a.get("href"))
                out.append((depth, text_of(a), href, frag))
            for sub in [c for c in li if tag(c) == "ol"]:
                walk(sub, depth + 1)

    for ol in [c for c in toc if tag(c) == "ol"]:
        walk(ol, 0)
    return out


def _parse_ncx(z, ncx_href):
    root = ET.fromstring(z.read(ncx_href))
    out = []

    def walk(el, depth):
        for np in [c for c in el if tag(c) == "navPoint"]:
            label = next((text_of(l) for l in np.iter() if tag(l) == "text"), "")
            src = next((c.get("src") for c in np.iter() if tag(c) == "content"), "")
            if label:
                href, frag = _resolve(ncx_href, src)
                out.append((depth, label, href, frag))
            walk(np, depth + 1)

    walk(find_all(root, "navMap")[0], 0)
    return out


# ------------------------------------------------------------ recipe filter

# Front/back matter and section headings that are never recipes.
NOT_RECIPE = re.compile(
    r"^(cover|title page|copyright|dedication|contents|table of contents|acknowledg|"
    r"about the author|index|introduction|foreword|preface|epigraph|notes?|"
    r"appendix|glossary|bibliography|resources|equipment|pantry|"
    r"chapter\s|part\s|\d+[\.\s]*$)", re.I)


def find_recipes(path):
    """ToC leaf entries, minus front/back matter. Returns [(title, href, frag)]."""
    z, spine, entries = toc_entries(path)
    keep = [e for e in entries if e[2] and not NOT_RECIPE.match(e[1]) and 3 <= len(e[1]) <= 90]
    if not keep:
        return []
    # Recipes sit at the deepest heavily-populated level; section headings are
    # far fewer and sit above them. Pure counting, no per-book configuration.
    counts = {}
    for depth, *_ in keep:
        counts[depth] = counts.get(depth, 0) + 1
    leaf = max(counts, key=lambda d: counts[d])
    return [(t, h, f) for d, t, h, f in keep if d == leaf]


# ------------------------------------------------------- non-recipe documents

# Standard epub structural roles that never contain recipes. The index is the
# important one: it is a long list of recipe names with no ingredients or
# method, i.e. prime material for the model to hallucinate recipes from.
SKIP_TYPES = {"index", "toc", "cover", "titlepage", "copyright-page", "colophon",
              "dedication", "acknowledgments", "bibliography", "glossary", "loi", "lot"}
# Landmarks often omit the index, so the document's own role is checked too.
# ponytail: "backmatter" is included because that is how some books tag their
# index. A recipe in back matter would be dropped -- none of the sample books
# do that; narrow this to "index" alone if a book turns up that does.
SKIP_DOC_TYPES = SKIP_TYPES | {"doc-index", "backmatter"}
DOC_TYPE = re.compile(r'(?:epub:type|role)="([^"]+)"', re.I)


def is_skippable_doc(raw):
    """True if the document tags itself as index/back matter in its opening markup."""
    tokens = {t for m in DOC_TYPE.findall(raw[:4000]) for t in m.lower().split()}
    return bool(tokens & SKIP_DOC_TYPES)


def skip_hrefs(path):
    """Documents flagged as front/back matter by the book's own metadata."""
    z, manifest, spine, ncx_id, opf = load_book(path)
    skip = set()

    # epub2: <guide><reference type="index" href="..."/>
    root = ET.fromstring(z.read(opf))
    for ref in find_all(root, "reference"):
        if (ref.get("type") or "").lower() in SKIP_TYPES:
            skip.add(_resolve(opf, ref.get("href"))[0])

    # epub3: <nav epub:type="landmarks"><li><a epub:type="index" href="..."/>
    nav = next((h for h, _, p in manifest.values() if "nav" in p.split()), None)
    if nav:
        navroot = ET.fromstring(z.read(nav))
        for n in find_all(navroot, "nav"):
            if (n.get("{http://www.idpf.org/2007/ops}type") or "") != "landmarks":
                continue
            for a in find_all(n, "a"):
                t = (a.get("{http://www.idpf.org/2007/ops}type") or "").lower()
                if any(part in SKIP_TYPES for part in t.split()):
                    skip.add(_resolve(nav, a.get("href"))[0])
        skip.add(nav)
    return skip


def _anchor_offsets(raw, frags):
    """Byte offset of each ToC fragment within the document, in document order."""
    found = []
    for title, frag in frags:
        m = re.search(r'(?:id|name)="%s"' % re.escape(frag), raw)
        if m:
            # Cut at the start of the enclosing tag, not at the id attribute
            # itself -- splitting mid-tag leaves 'id="frag">' in the text and a
            # dangling '<p' on the end of the previous chunk.
            start = raw.rfind("<", 0, m.start())
            found.append((start if start != -1 else m.start(), title, frag))
    return sorted(found)


ANY_ID = re.compile(r'(?:id|name)="([^"]+)"')
LINK = re.compile(r'<a\b[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.S | re.I)


class Chunk(NamedTuple):
    id: int
    href: str
    anchor: str          # "doc.xhtml#frag", or just "doc.xhtml" for a whole document
    title: str
    text: str
    links: dict          # marker number -> (target href, target fragment)


def _mark_links(seg, doc_href, next_id):
    """Replace <a> tags with [[R<n>]] markers so the model can cite them.

    ponytail: the marker carries the reference, not the anchor text -- books
    overwhelmingly write "here" or "this page", which tells the model nothing.
    """
    links = {}

    def repl(m):
        target, inner = m.group(1), m.group(2)
        if target.startswith(("http:", "https:", "mailto:")):
            return inner
        n = next_id[0]
        next_id[0] += 1
        links[n] = _resolve(doc_href, target)
        return f" [[R{n}]] {inner} "

    return LINK.sub(repl, seg), links


def parse(path):
    """Return (units, resolve) for a book.

    A unit is a single recipe wherever the ToC provides in-document anchors,
    and the whole document otherwise. Documents the book marks as index,
    copyright, etc. are dropped. `resolve(href, frag)` maps a link target back
    to the id of the unit containing it, or None.
    """
    z, spine, entries = toc_entries(path)
    skip = skip_hrefs(path)

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
        seg, links = _mark_links(raw[start:end], href, next_id)
        text = strip_tags(seg)
        if not text:
            return
        anchor = f"{href}#{frag}" if frag else href
        unit = Chunk(len(units), href, anchor, title, text, links)
        units.append(unit)
        ranges.setdefault(href, []).append((start, end, unit.id))

    for href in spine:
        if href in skip or not href.endswith((".xhtml", ".html", ".htm")):
            continue
        try:
            raw = z.read(href).decode("utf8", "replace")
        except Exception:
            continue
        if is_skippable_doc(raw):
            continue
        id_offsets[href] = {m.group(1): m.start() for m in ANY_ID.finditer(raw)}

        # ponytail: split on the ToC's own anchors, never on byte counts --
        # a size-based cut would slice a recipe in half.
        cuts = _anchor_offsets(raw, anchored.get(href, []))
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
        if not frag:
            return spans[0][2]
        offset = id_offsets.get(href, {}).get(frag)
        if offset is None:
            return spans[0][2]
        for start, end, uid in spans:
            if start <= offset < end:
                return uid
        return spans[0][2]

    return units, resolve


def chunks(path):
    """Units only, for callers that do not need link resolution."""
    return parse(path)[0]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    limit = 20
    for a in sys.argv[1:]:
        if a.startswith("--limit="):
            limit = int(a.split("=")[1])
    for path in args:
        print(f"\n=== {path.rsplit('/', 1)[-1]}")
        try:
            recipes = find_recipes(path)
        except Exception as e:
            print(f"    failed: {type(e).__name__}: {e}")
            continue
        units = list(chunks(path))
        sizes = sorted(len(t) for _, _, t in units)
        print(f"    {len(recipes)} recipe candidates from ToC | {len(units)} chunks "
              f"(median {sizes[len(sizes) // 2]}B, max {sizes[-1]}B)")
        for title, _href, _frag in recipes[:limit]:
            print(f"      {title}")


if __name__ == "__main__":
    main()
