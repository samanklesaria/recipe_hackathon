"""The week's menu and the shopping list, two tabs, nothing to click but Generate.

    uv run python -m gui

Both views are rendered as HTML into a read-only QTextBrowser: the content is a
document, not a form, and a document is one setHtml call instead of a tree of
widgets to keep in sync.
"""
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from html import escape

import duckdb
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPlainTextEdit,
                             QPushButton, QTabWidget, QTextBrowser,
                             QVBoxLayout, QWidget)

import plan as planner
import settings
import store_llm

CSS = """
<style>
  body { font-family: -apple-system, sans-serif; }
  h2 { margin-bottom: 2px; }
  .sub { color: #666; font-size: small; }
  ul { margin-top: 2px; }
</style>
"""


BOOKS = "example_cookbooks"


def menu_html(plan):
    """Just the titles: one bullet per main, its sides named alongside.

    Each title links to recipe:<id>; Window.open_recipe turns that into a
    Calibre viewer opened at the recipe's ToC entry.
    """
    kids = defaultdict(list)
    for r in plan.recipes:
        if r.for_recipe is not None:
            kids[r.for_recipe].append(r)

    def link(r):
        return f'<a href="recipe:{r.id}"><b>{escape(r.name)}</b></a>'

    def line(r):
        sides = " and ".join(link(k) for k in kids[r.id])
        return f"<li>{link(r)}" + (f" with {sides}" if sides else "") + "</li>"

    mains = [r for r in plan.recipes if r.for_recipe is None]
    if not mains:
        return CSS + "<p>Nothing to cook. Is the database empty?</p>"
    return CSS + "<ul>" + "".join(line(r) for r in mains) + "</ul>"


def list_html(plan):
    """One paragraph per store, one line per item."""
    out = []
    for store, items in sorted(plan.by_store.items(), key=lambda kv: kv[0] or "~"):
        # store descriptions read "Name: two sentences about it"
        name = store.split(":")[0] if store else "No store stocks this"
        out.append(f"<h2>{escape(name)}</h2><p>"
                   + "<br>".join(escape(i) for i in sorted(items)) + "</p>")
    return CSS + ("".join(out) or "<p>Nothing to buy.</p>")


def sources(ids):
    """id -> (epub path, ToC href), for the recipes we know where to find."""
    with duckdb.connect(settings.DB, read_only=True) as con:
        rows = con.execute(
            "select id, filename, anchor from recipes where id in (select unnest(?))"
            " and filename is not null and anchor is not null", [list(ids)]).fetchall()
    return {rid: (os.path.join(BOOKS, fn), anchor) for rid, fn, anchor in rows}


def _stored():
    """(descriptions in priority order, whether every one of them is embedded)."""
    with duckdb.connect(settings.DB, read_only=True) as con:
        rows = con.execute("select description, embedding is not null"
                           " from stores order by priority").fetchall()
    return [d for d, _ in rows], all(ok for _, ok in rows)


def load_stores():
    return "\n\n".join(_stored()[0])


def save_stores(text):
    """Same parse as scripts/load_stores.py: one store per non-blank line.

    Line order is the priority: the store listed first wins a tie. The store
    embeddings are computed here, once per edit, so planning a week is only
    the ingredient side of the dot product.
    """
    stores = [line.strip() for line in text.splitlines() if line.strip()]
    # unchanged text still needs a pass if a row predates the embedding column:
    # plan.py cannot score a store without one, and would quietly drop it
    known, embedded = _stored()
    if not stores or (stores == known and embedded):
        return
    vecs = store_llm.embed([store_llm.STORE_PREFIX + s for s in stores])
    with duckdb.connect(settings.DB) as con:
        con.execute("delete from stores")
        con.executemany(
            "insert into stores (description, priority, embedding) values (?, ?, ?)"
            " on conflict do nothing",
            [(s, i, [float(x) for x in v]) for i, (s, v) in enumerate(zip(stores, vecs))])


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dinner")
        self.menu = QTextBrowser()
        self.menu.setOpenLinks(False)
        self.menu.anchorClicked.connect(self.open_recipe)
        self.sources = {}
        self.shopping = QTextBrowser()

        button = QPushButton("Generate")
        button.clicked.connect(self.generate)
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(self.menu)
        layout.addWidget(button)

        self.stores = QPlainTextEdit(load_stores())

        tabs = QTabWidget()
        tabs.addTab(page, "Menu")
        tabs.addTab(self.shopping, "Shopping List")
        tabs.addTab(self.stores, "Grocery Stores")
        # the store list is only read when a plan is generated, so writing it
        # back on the way out of the tab is soon enough
        tabs.currentChanged.connect(lambda _: save_stores(self.stores.toPlainText()))
        self.setCentralWidget(tabs)
        self.generate()

    def closeEvent(self, event):
        save_stores(self.stores.toPlainText())
        super().closeEvent(event)

    def open_recipe(self, url):
        """toc-href is an exact match on the book's own ToC href, which is
        exactly what epub_extract stored as the anchor.

        Launched through `open`, not by running the ebook-viewer binary: a
        Chromium helper spawned into our process coalition aborts its sandbox
        and the viewer dies with "Render process crashed". `open` hands the job
        to launchd, which starts it in a coalition of its own. Hence the
        absolute book path too, since launchd has no working directory of ours.
        """
        found = self.sources.get(int(url.toString().removeprefix("recipe:")))
        viewer = shutil.which("ebook-viewer")
        if not (found and viewer):
            return
        path, anchor = found
        app = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(viewer))),
                           "ebook-viewer.app")
        subprocess.Popen(["open", "-a", app, "--args",
                          "--open-at", "toc-href:" + anchor, os.path.abspath(path)])

    def generate(self):
        save_stores(self.stores.toPlainText())
        plan = planner.shopping_plan()
        self.sources = sources([r.id for r in plan.recipes])
        self.menu.setHtml(menu_html(plan))
        self.shopping.setHtml(list_html(plan))


def main():
    app = QApplication(sys.argv)
    window = Window()
    window.resize(720, 640)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
