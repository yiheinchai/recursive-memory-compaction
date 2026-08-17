#!/usr/bin/env python3
"""Assemble the documentation site from content fragments.

The docs were one 1,100-line page. Everything worked and nothing was findable:
a reader looking for a single config key scrolled past the whole design
rationale, and there was no URL to send anyone that meant "how compression
decides what to drop" rather than "the docs".

So the content is split into pages that each answer one question, and this
script puts the shared chrome around them. A generator rather than eight
hand-maintained copies of the same header, because the alternative is eight
navigations that drift apart — and the one that drifts is always the one the
reader is looking at.

Deliberately no dependencies and no watch mode: `python3 docs/build.py`, commit
the output, GitHub Pages serves it. A docs toolchain that has to be installed
before a typo can be fixed is a docs toolchain that stops being used.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).parent
FRAGMENTS = HERE / "_sections"

VERSION = "v0.1"
REPO = "https://github.com/yiheinchai/rmc"


@dataclass
class Page:
    slug: str
    title: str
    blurb: str
    sections: list[str] = field(default_factory=list)
    lead: str = ""

    @property
    def href(self) -> str:
        return f"{self.slug}.html"


@dataclass
class Group:
    name: str
    pages: list[Page]


# The information architecture. Ordered by what a reader needs first, not by
# what is most interesting to explain: install, then the thing that goes wrong,
# then reference, and only then how it works. Someone reading the design notes
# is having a good day; someone whose hook is silent is not.
NAV: list[Group] = [
    Group("Get started", [
        Page("quickstart", "Quickstart",
             "Install RMC, wire the hooks, and see the first lesson land.",
             ["start"]),
        Page("troubleshooting", "Troubleshooting",
             "Symptoms, likely causes, and the command that settles each one.",
             ["trouble"]),
        Page("tasks", "Everyday tasks",
             "Reading what RMC knows, teaching it directly, and reading its own numbers.",
             ["tasks"]),
        Page("migrate", "Migrate from Claude skills",
             "Convert an existing SKILL.md library into lessons that recall automatically.",
             ["migrate"]),
    ]),
    Group("Reference", [
        Page("cli", "CLI reference",
             "Every command, what it does, and when you would reach for it.",
             ["commands"]),
        Page("configuration", "Configuration",
             "Every setting, its default, and the reasoning behind that default.",
             ["config"]),
        Page("data-model", "Data model",
             "What a lesson is on disk, and how the graph is shaped.",
             ["model"]),
    ]),
    Group("How it works", [
        Page("concepts", "The loop and the rule",
             "The seven stages, and the one rule that decides what belongs to code.",
             ["loop", "rule"]),
        Page("recall", "01 Recall",
             "Choosing which lessons enter your context, on every prompt.",
             ["recall"]),
        Page("reflection", "02 Reflection",
             "Noticing that a session contained something worth keeping.",
             ["reflect"]),
        Page("attribution", "03 Attribution",
             "Crediting the lessons that actually bore on the work.",
             ["attribute"]),
        Page("consolidation", "04 Consolidation",
             "Building abstraction from what gets used together.",
             ["consolidate"]),
        Page("compression", "05 Compression",
             "Making a lesson shorter without making it wrong.",
             ["compress"]),
        Page("descent", "06 Descent",
             "Recovering a dropped specific at the moment it turns out to matter.",
             ["descend"]),
        Page("dreaming", "07 Dreaming",
             "The offline pass that keeps the top of the store narrow.",
             ["dream"]),
    ]),
    Group("Measuring", [
        Page("evaluation", "Evaluation",
             "Scoring retrieval and compression against recorded outcomes.",
             ["eval"]),
        Page("tuning", "Self-tuning",
             "Letting RMC propose and validate its own retrieval improvements.",
             []),
        Page("limits", "Known limits",
             "What RMC still gets wrong, stated plainly.",
             ["gaps"]),
    ]),
]

PAGES: list[Page] = [p for g in NAV for p in g.pages]


# --------------------------------------------------------------------------- #
# chrome
# --------------------------------------------------------------------------- #

FAVICON = (
    "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'>"
    "<rect width='32' height='32' fill='%230a0a0a'/><rect x='6' y='7' width='20' height='3' fill='white'/>"
    "<rect x='9' y='14' width='14' height='3' fill='white'/>"
    "<rect x='12' y='21' width='8' height='3' fill='white'/></svg>"
)


def head(page: Page) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(page.title)} — RMC</title>
<meta name="description" content="{html.escape(page.blurb)}">
<meta property="og:title" content="{html.escape(page.title)} — RMC docs">
<meta property="og:description" content="{html.escape(page.blurb)}">
<link rel="icon" href="{FAVICON}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="style.css">
<link rel="stylesheet" href="docs.css">
<script>
// Applied before first paint: reading the stored theme in the body would show
// a white flash on every navigation for anyone who chose dark.
try {{
  var t = localStorage.getItem('rmc-theme');
  if (t) document.documentElement.dataset.theme = t;
  else if (matchMedia('(prefers-color-scheme: dark)').matches)
    document.documentElement.dataset.theme = 'dark';
}} catch (e) {{}}
</script>
</head>
<body>"""


def topbar() -> str:
    return f"""
<nav class="nav">
  <div class="wrap nav-in">
    <button class="burger" aria-label="Menu" aria-expanded="false"></button>
    <a class="logo" href="./">
      <svg width="22" height="22" viewBox="0 0 32 32" aria-hidden="true">
        <rect width="32" height="32" fill="currentColor"/>
        <rect x="6" y="7" width="20" height="3" fill="var(--plate)"/>
        <rect x="9" y="14" width="14" height="3" fill="var(--plate)"/>
        <rect x="12" y="21" width="8" height="3" fill="var(--plate)"/>
      </svg>
      RMC
    </a>
    <button class="search-open" aria-label="Search documentation">
      <span>Search</span><kbd>/</kbd>
    </button>
    <div class="nav-links">
      <a href="./">Overview</a>
      <a href="quickstart.html">Docs</a>
      <button class="theme" aria-label="Toggle colour scheme"></button>
      <a class="btn" href="{REPO}">GitHub&nbsp;↗</a>
    </div>
  </div>
</nav>"""


def sidebar(current: Page) -> str:
    out = ['<aside class="side" id="side"><nav>']
    for group in NAV:
        out.append(f'<div class="grp">{group.name}</div>')
        for page in group.pages:
            live = ' class="on" aria-current="page"' if page is current else ""
            out.append(f'<a href="{page.href}"{live}>{html.escape(page.title)}</a>')
    out.append("</nav></aside>")
    return "\n".join(out)


def slugs(body: str) -> str:
    """Give every heading an id, deriving one from its text where it has none.

    Only the h2s carried ids, because they were the anchors of a single long
    page. Splitting the docs makes the subheadings addressable too — and both
    the secondary navigation and the anchor links key off the id, so a heading
    without one is silently absent from the contents rather than visibly
    broken.
    """
    used: set[str] = set()

    def fix(match: re.Match) -> str:
        level, attrs, inner = match.group(1), match.group(2), match.group(3)
        if 'id="' in attrs:
            used.add(re.search(r'id="([^"]+)"', attrs).group(1))
            return match.group(0)
        text = re.sub(r"<[^>]+>", "", inner).replace("&nbsp;", " ")
        base = re.sub(r"[^a-z0-9]+", "-", html.unescape(text).lower()).strip("-")[:48]
        base = base or "section"
        ident, n = base, 2
        while ident in used:
            ident, n = f"{base}-{n}", n + 1
        used.add(ident)
        return f"<h{level}{attrs} id=\"{ident}\">{inner}</h{level}>"

    return re.sub(r"<h([234])([^>]*?)>(.*?)</h\1>", fix, body, flags=re.S)


def on_this_page(body: str) -> str:
    """Secondary navigation, built from the headings actually present.

    Generated rather than written by hand for the usual reason: a hand-kept
    contents list is correct on the day it is written.
    """
    items = re.findall(r'<h([23]) id="([^"]+)"[^>]*>(.*?)</h[23]>', body, re.S)
    if len(items) < 2:
        return ""
    out = ['<aside class="onpage"><div class="grp">On this page</div><nav>']
    for level, ident, label in items:
        text = re.sub(r"<[^>]+>", "", label).replace("&nbsp;", " ").strip()
        # A command signature is the right heading and the wrong nav entry:
        # `rmc report [--about "..."] [--expected "..."] [--days N]` wraps to
        # three lines in a 208px column and buries the name it exists to show.
        text = re.sub(r"\s*[\[<].*$", "", text).strip() or text
        out.append(f'<a class="l{level}" href="#{ident}">{html.escape(text)}</a>')
    out.append("</nav></aside>")
    return "\n".join(out)


def pager(current: Page) -> str:
    index = PAGES.index(current)
    previous = PAGES[index - 1] if index else None
    following = PAGES[index + 1] if index + 1 < len(PAGES) else None
    out = ['<nav class="pager">']
    if previous:
        out.append(
            f'<a class="prev" href="{previous.href}"><span>Previous</span>'
            f"<b>{html.escape(previous.title)}</b></a>"
        )
    else:
        out.append("<span></span>")
    if following:
        out.append(
            f'<a class="next" href="{following.href}"><span>Next</span>'
            f"<b>{html.escape(following.title)}</b></a>"
        )
    out.append("</nav>")
    return "\n".join(out)


def anchors(body: str) -> str:
    """Give every heading a clickable anchor.

    The point of splitting the docs was linkability; a heading you cannot get a
    URL for is only half split out.
    """
    def add(match: re.Match) -> str:
        level, attrs, inner = match.group(1), match.group(2), match.group(3)
        found = re.search(r'id="([^"]+)"', attrs)
        if not found:
            return match.group(0)
        # After the text, not before it. An anchor at opacity 0 still occupies
        # its box, so leading it indented every heading by the width of a
        # character nobody could see.
        return (
            f"<h{level}{attrs}>{inner}<a class=\"anchor\" href=\"#{found.group(1)}\" "
            f"aria-label=\"Link to this section\">#</a></h{level}>"
        )

    return re.sub(r"<h([234])([^>]*?)>(.*?)</h\1>", add, body, flags=re.S)


def search_index() -> str:
    """A tiny client-side index. No service, no request, no tracking.

    Docs search that calls out to a third party is a dependency on someone
    else's uptime for the page people read when something is broken.
    """
    def plain(fragment: str) -> str:
        return html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", fragment))).strip()

    records = []
    for page in PAGES:
        body = slugs(load(page))

        # One record per section, carrying that section's own text. Indexing
        # the page as a single truncated blob loses exactly the content people
        # search for: a config key or a flag sits three thousand characters
        # into a reference table, past any sane cut-off, and comes back "no
        # matches" from a page that documents it.
        pieces = re.split(r'(?=<h[23] id=")', body)
        records.append({"p": page.title, "u": page.href, "h": page.title,
                        "t": (page.blurb + " " + plain(pieces[0]))[:600]})
        for piece in pieces[1:]:
            found = re.match(r'<h[23] id="([^"]+)"[^>]*>(.*?)</h[23]>', piece, re.S)
            if not found:
                continue
            heading = re.sub(r"<[^>]+>", "", found.group(2)).replace("&nbsp;", " ").strip()
            records.append({
                "p": page.title,
                "u": f"{page.href}#{found.group(1)}",
                "h": heading,
                "t": plain(piece)[:1800],
            })
    return json.dumps(records, separators=(",", ":"))


def load(page: Page) -> str:
    chunks = []
    for name in page.sections:
        path = FRAGMENTS / f"{name}.html"
        if path.exists():
            chunks.append(path.read_text(encoding="utf-8"))
    extra = FRAGMENTS / f"_{page.slug}.html"
    if extra.exists():
        chunks.append(extra.read_text(encoding="utf-8"))
    return "\n\n".join(chunks)


def render(page: Page) -> str:
    body = slugs(load(page))
    # The first <h2> becomes the page title, so it must not repeat inside.
    body = re.sub(r"^\s*<h2 id=\"[^\"]+\"[^>]*>.*?</h2>", "", body, count=1, flags=re.S)
    return "\n".join([
        head(page),
        topbar(),
        '<div class="shell">',
        sidebar(page),
        '<main class="doc">',
        f'<div class="crumbs"><a href="quickstart.html">Docs</a><span>/</span>'
        f"<span>{html.escape(page.title)}</span></div>",
        f"<h1>{html.escape(page.title)}</h1>",
        f'<p class="lead">{page.blurb}</p>',
        '<div class="prose">',
        anchors(body),
        "</div>",
        pager(page),
        "</main>",
        on_this_page(body),
        "</div>",
        search_dialog(),
        '<script id="rmc-search" type="application/json">' + search_index() + "</script>",
        '<script src="docs.js"></script>',
        "</body>\n</html>",
    ])


def search_dialog() -> str:
    return """
<div class="searchbox" id="searchbox" hidden>
  <div class="searchbox-in" role="dialog" aria-modal="true" aria-label="Search documentation">
    <input type="search" id="q" placeholder="Search the documentation" autocomplete="off" spellcheck="false">
    <div id="results"></div>
    <div class="hint"><kbd>↑</kbd><kbd>↓</kbd> to navigate <kbd>↵</kbd> to open <kbd>esc</kbd> to close</div>
  </div>
</div>"""


REDIRECT = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<!-- docs.html was the whole documentation for as long as it was one page.
     Those URLs are in commit messages, issues and other people's notes, so it
     stays and forwards rather than 404s. -->
<link rel="canonical" href="quickstart.html">
<meta http-equiv="refresh" content="0; url=quickstart.html">
<title>RMC — Documentation</title>
</head>
<body>
<p>The documentation is now split by topic.
<a href="quickstart.html">Continue to the docs</a>.</p>
</body>
</html>
"""


def main() -> None:
    for page in PAGES:
        (HERE / page.href).write_text(render(page), encoding="utf-8")
        print(f"  {page.href:24} {len(load(page).splitlines()):4d} lines")
    (HERE / "docs.html").write_text(REDIRECT, encoding="utf-8")
    print(f"{len(PAGES)} pages + docs.html redirect")


if __name__ == "__main__":
    main()
