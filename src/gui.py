"""The week's menu and the shopping list, two tabs, nothing to click but Generate.

    uv run python -m gui

Both views are rendered as HTML into a read-only QTextBrowser: the content is a
document, not a form, and a document is one setHtml call instead of a tree of
widgets to keep in sync.
"""
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


def menu_html(plan):
    """Just the titles: one bullet per main, its sides named alongside."""
    kids = defaultdict(list)
    for r in plan.recipes:
        if r.for_recipe is not None:
            kids[r.for_recipe].append(r)

    def line(r):
        sides = " and ".join(f"<b>{escape(k.name)}</b>" for k in kids[r.id])
        return f"<li><b>{escape(r.name)}</b>" + (f" with {sides}" if sides else "") + "</li>"

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


def _stored():
    """(descriptions in priority order, whether every one of them is embedded)."""
    with duckdb.connect(settings.DB, read_only=True) as con:
        rows = con.execute("select description, embedding is not null"
                           " from stores order by priority").fetchall()
    return [d for d, _ in rows], all(ok for _, ok in rows)


def load_stores():
    return "\n".join(_stored()[0])


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

    def generate(self):
        save_stores(self.stores.toPlainText())
        plan = planner.shopping_plan()
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
