#!/usr/bin/env python3
"""Build the open Chinook Jargon lexicon from public-domain dictionaries.

    python3 scripts/build_lexicon.py                # download (cached) + build
    python3 scripts/build_lexicon.py --no-download  # use the cache only
    python3 scripts/build_lexicon.py --stats        # also print overlap stats

Inputs  : archive.org ``<id>_djvu.txt`` OCR text (and ``_djvu.xml`` word coordinates for the
          multi-column Le Jeune and Demers pages) of each source, cached under
          ``corpus/lexicon_src/`` (override with ``--src-dir`` or
          ``$CHINUKPIPA_LEXICON_SRC``).  The download step is part of this script.
Outputs : ``data/lexicon/lexicon.tsv``         one row per (headword, source)
          ``data/lexicon/lexicon_merged.tsv``  one row per normalized headword
          ``data/lexicon/sources.yaml``        per-source citation + counts

Every source has its own parser because every dictionary has its own layout
and its own OCR damage.  Parsers only *extract*: a headword is never made up,
and uncertain OCR is flagged (``conf=B``/``conf=C`` in ``notes``) rather than
silently repaired.  Per-source parsing notes are written to ``data/lexicon/sources.yaml``.

Only the Python standard library is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import os
import re
import ssl
import subprocess
import sys
import urllib.request
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "data" / "lexicon"
DEFAULT_SRC = Path(os.environ.get("CHINUKPIPA_LEXICON_SRC", "corpus/lexicon_src"))

# Load chinukpipa/lexicon.py by path (shares normalize_headword / skeleton / levenshtein
# without requiring the package to be installed or its __init__ to be importable).
_spec = importlib.util.spec_from_file_location("chinukpipa_lexicon", REPO / "chinukpipa" / "lexicon.py")
LX = importlib.util.module_from_spec(_spec)
sys.modules["chinukpipa_lexicon"] = LX
_spec.loader.exec_module(LX)  # type: ignore[union-attr]
normalize_headword = LX.normalize_headword
skeleton = LX.skeleton
levenshtein = LX.levenshtein

LEXICON_COLS = [
    "headword", "variants", "gloss_en", "gloss_fr", "pos",
    "source_key", "source_page", "kamloops", "notes",
]

# --------------------------------------------------------------------------
# Source registry
# --------------------------------------------------------------------------

PD_US = (
    "Published {year}, i.e. before 1931; public domain in the United States by "
    "publication date. {extra}"
)

SOURCES: "OrderedDict[str, dict]" = OrderedDict()


def _src(key, **kw):
    SOURCES[key] = dict(key=key, **kw)


_src(
    "rudiments1924",
    title="Chinook Rudiments",
    author="J. M. R. Le Jeune, O.M.I.",
    year=1924,
    citation=(
        "Le Jeune, J. M. R. (Jean Marie Raphael), 1855-1930. 1924. Chinook Rudiments. [Kamloops, B.C.]: s.n. "
        "36 p. (Newberry Library, Edward E. Ayer Collection, call no. PM843 .L45 1924; scanned by Indiana "
        "University for the Internet Archive)."
    ),
    ia_id="Ayer_PM843_L45_1924",
    kamloops=True,
    kind="vocabulary lists + grammar notes (Kamloops variety, Le Jeune's own Latin spelling; lithographed handwriting)",
    status="parsed (very noisy OCR; entries graded by validation)",
    license=PD_US.format(
        year=1924,
        extra="Author died 1930.",
    ),
)
_src(
    "rudiments1898",
    title="Chinook and Shorthand Rudiments, with which the Chinook Jargon and the Wawa shorthand can be mastered without a teacher in a few hours",
    author="J. M. R. Le Jeune, O.M.I.",
    year=1898,
    citation=(
        "Le Jeune, J. M. R. (Jean Marie Raphael), 1855-1930. 1898. Chinook and Shorthand Rudiments, with which the "
        "Chinook Jargon and the Wawa shorthand can be mastered without a teacher in a few hours. Kamloops, B.C.: "
        "[s.n.]. 43 p. (Canadian Institute for Historical Microreproductions no. 15465, filmed from the copy held by "
        "the Provincial Archives of British Columbia)."
    ),
    ia_id="cihm_15465",
    xml_ids=["cihm_15465"],
    kamloops=True,
    kind="alphabetical vocabulary of the commonest Chinook words (Kamloops variety, Le Jeune's Latin spelling, printed; each word also given in shorthand)",
    status="parsed (xml two-column reader, vocabulary pages only; shorthand-glyph noise between headword and gloss)",
    license=PD_US.format(year=1898, extra="Author died 1930."),
)
_src(
    "practical1886",
    title="Practical Chinook Vocabulary",
    author="J. M. R. Le Jeune, O.M.I.",
    year=1886,
    citation=(
        "Le Jeune, J. M. R. 1886. Practical Chinook Vocabulary, comprising all & the only usual words "
        "of that wonderful language arranged in a most advantageous order for the speedy learning of the same, "
        "after the plan of Right Rev. Bishop Durieu O.M.I. Kamloops: St. Louis' Mission."
    ),
    ia_id="Ayer_PM848_L4_1886",
    ia_alt=["cihm_15489"],
    xml_ids=["Ayer_PM848_L4_1886", "cihm_15489"],
    kamloops=True,
    kind="classified vocabulary (Chinook-English, topical lists, printed in two columns)",
    status="parsed (layout-aware two-column reader; OCR errors in glosses)",
    license=PD_US.format(year=1886, extra="Author died 1930."),
)
_src(
    "vocab1892",
    title="Chinook Vocabulary, Chinook-English",
    author="J. M. R. Le Jeune, O.M.I. (from the original of Rt. Rev. Bishop Durieu)",
    year=1892,
    citation=(
        "Le Jeune, J. M. R. 1892 (Oct.). Chinook Vocabulary, Chinook-English, from the original of "
        "Rt. Rev. Bishop Durieu O.M.I. Kamloops: St. Louis Mission. [lithographed manuscript]"
    ),
    ia_id="Ayer_PM848_L4_1892",
    ia_alt=["cihm_15474"],
    kamloops=True,
    kind="alphabetical Chinook-English list: hand-lettered Roman spelling, the word in phonography (Duployan shorthand), English gloss",
    status="NOT PARSED: OCR fails on the lithographed handwriting (232 rows were read by AI from the images instead; see data/gt/)",
    license=PD_US.format(year=1892, extra="Author died 1930."),
)
_src(
    "gibbs1863",
    title="A Dictionary of the Chinook Jargon, or Trade Language of Oregon",
    author="George Gibbs",
    year=1863,
    citation=(
        "Gibbs, George. 1863. A Dictionary of the Chinook Jargon, or Trade Language of Oregon. "
        "Smithsonian Miscellaneous Collections 161. Washington: Smithsonian Institution."
    ),
    ia_id="dictionaryofchin00gibbrich",
    kamloops=False,
    kind="Chinook-English dictionary with etymologies (Part I used; Part II English-Chinook index skipped)",
    status="parsed",
    license=PD_US.format(year=1863, extra="Author died 1873."),
)
_src(
    "hibben1889",
    title="Dictionary of the Chinook Jargon, or Indian Trade Language of the North Pacific Coast",
    author="T. N. Hibben & Co. (Victoria, B.C.)",
    year=1889,
    citation=(
        "Hibben, T. N. & Co. [1889 printing; © 1877]. Dictionary of the Chinook Jargon, or Indian Trade "
        "Language of the North Pacific Coast. Victoria, B.C.: T. N. Hibben & Co."
    ),
    ia_id="dictionaryofchin00unse_0",
    kamloops=False,
    kind="abridgement of Gibbs 1863 (Part I Chinook-English, etymologies dropped)",
    status="parsed",
    license=PD_US.format(year=1889, extra="Compiler anonymous; text is an abridgement of Gibbs (1863)."),
)
_src(
    "shaw1909",
    title="The Chinook Jargon and How to Use It: A Complete and Exhaustive Lexicon",
    author="George C. Shaw",
    year=1909,
    citation=(
        "Shaw, George C. 1909. The Chinook Jargon and How to Use It: a complete and exhaustive lexicon "
        "of the oldest trade language of the American continent. Seattle: Rainier Printing Co."
    ),
    ia_id="chinookjargonhow00shaw",
    ia_alt=["chinookjargonhow00shawuoft"],
    kamloops=False,
    kind="Chinook-English lexicon with origin marks (main Lexicon + Supplemental Vocabulary)",
    status="parsed",
    license=PD_US.format(year=1909, extra="Published in Seattle 1909."),
)
_src(
    "gill1909",
    title="Gill's Dictionary of the Chinook Jargon",
    author="John Kaye Gill",
    year=1909,
    citation=(
        "Gill, John Kaye. 1909 (14th ed.). Gill's Dictionary of the Chinook Jargon, with examples of use "
        "in conversation and notes upon tribes and tongues. Portland, Or.: J. K. Gill Co."
    ),
    ia_id="gillsdictionaryo00gill",
    kamloops=False,
    kind="Chinook-English dictionary with origin marks (Chinook-English part only)",
    status="parsed",
    license=PD_US.format(year=1909, extra=""),
)
_src(
    "gill1887",
    title="Dictionary of the Chinook Jargon, with Examples of Use in Conversation",
    author="John Kaye Gill",
    year=1887,
    citation=(
        "Gill, John Kaye. 1887. Dictionary of the Chinook Jargon, with examples of use in conversation "
        "[compiled from all vocabularies and greatly enlarged]. Portland, Or.: J. K. Gill & Co."
    ),
    ia_id="dictionaryofchin00gillrich",
    kamloops=False,
    kind="Chinook-English dictionary (Chinook-English part only)",
    status="parsed",
    license=PD_US.format(year=1887, extra=""),
)
_src(
    "demers1871",
    title="Chinook Dictionary, Catechism, Prayers and Hymns",
    author="M. Demers, F. N. Blanchet, L. N. St-Onge",
    year=1871,
    citation=(
        "Demers, Modeste, F. N. Blanchet and L. N. St-Onge. 1871. Chinook Dictionary, Catechism, Prayers "
        "and Hymns, revised, corrected and completed in 1867 by Most Rev. F. N. Blanchet, with modifications "
        "and additions by Rev. L. N. St. Onge. Montreal: Cie. d'Imprimerie."
    ),
    ia_id="cihm_04222",
    xml_ids=["cihm_04222"],
    kamloops=False,
    kind="Chinook-English word list in Blanchet/St-Onge phonetic spelling (two columns per page; ditto marks; noisy OCR)",
    status="parsed (xml two-column reader; entries graded by validation)",
    license=PD_US.format(year=1871, extra=""),
)
_src(
    "long1909",
    title="Dictionary of the Chinook Jargon",
    author="Frederick J. Long",
    year=1909,
    citation=(
        "Long, Frederick J. 1909. Dictionary of the Chinook Jargon [English-Chinook and Chinook-English "
        "pocket lexicon]. Seattle: Lowman & Hanford Co. (Google Books scan, Stanford copy)."
    ),
    ia_id="dictionarychino00jgoog",
    kamloops=False,
    kind="pocket lexicon; only the Chinook-English half is parsed (Chinook forms follow Gibbs's spelling, syllabified)",
    status="parsed partially (head block / gloss block pairing only where line counts agree)",
    license=PD_US.format(year=1909, extra="Seattle 1909 pocket lexicon."),
)

# --------------------------------------------------------------------------
# Download / cache
# --------------------------------------------------------------------------

IA_URL = "https://archive.org/download/{id}/{id}_djvu.txt"


_LAST_FETCH = [0.0]
MIN_INTERVAL = 1.0  # seconds between requests to archive.org (be polite)
USER_AGENT = "chinuk-pipa-lexicon/1.0 (open-source research; +https://github.com/gradyhnichols/chinuk-pipa)"


def _fetch(url: str, dest: Path) -> None:
    """Fetch *url* to *dest* (at most one request per MIN_INTERVAL seconds), falling back to curl."""
    import time
    wait = MIN_INTERVAL - (time.monotonic() - _LAST_FETCH[0])
    if wait > 0:
        time.sleep(wait)
    _LAST_FETCH[0] = time.monotonic()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        ctx = ssl.create_default_context()
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=180, context=ctx) as r, open(tmp, "wb") as fh:
            while True:
                chunk = r.read(1 << 16)
                if not chunk:
                    break
                fh.write(chunk)
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"  urllib failed ({exc}); trying curl", file=sys.stderr)
        subprocess.run(["curl", "-sS", "-L", "--fail", "-m", "300", "-A", USER_AGENT, "-o", str(tmp), url], check=True)
    tmp.replace(dest)


def ia_ids(spec: dict) -> List[str]:
    return [spec["ia_id"]] + list(spec.get("ia_alt", []))


def source_path(src_dir: Path, ia_id: str, ext: str = "txt") -> Path:
    return src_dir / f"{ia_id}_djvu.{ext}"


def ensure_sources(src_dir: Path, download: bool, keys: Iterable[str]) -> None:
    """Make sure every needed ``_djvu.txt`` (and ``_djvu.xml`` for layout-aware parsers) is cached."""
    for key in keys:
        spec = SOURCES[key]
        needs = [(ia, "txt") for ia in ia_ids(spec)]
        needs += [(ia, "xml") for ia in spec.get("xml_ids", [])]
        for ia, ext in needs:
            p = source_path(src_dir, ia, ext)
            if p.exists() and p.stat().st_size > 1000:
                continue
            if not download:
                if ia == spec["ia_id"] or ext == "xml":
                    raise SystemExit(f"missing {p} (run without --no-download)")
                continue
            url = IA_URL.format(id=ia).replace("_djvu.txt", f"_djvu.{ext}")
            print(f"  downloading {url}", file=sys.stderr)
            _fetch(url, p)


def read_lines(src_dir: Path, ia_id: str) -> List[str]:
    txt = source_path(src_dir, ia_id).read_text(encoding="utf-8", errors="replace")
    txt = txt.replace("\r", "").replace("\x0c", "\n")
    return txt.split("\n")


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def squash(s: str) -> str:
    return " ".join(s.split())


def fix_quotes(s: str) -> str:
    return (
        s.replace("’", "'").replace("‘", "'")
        .replace("“", '"').replace("”", '"')
        .replace("—", "-").replace("–", "-")
    )


POS_CANON = {
    "n": "n", "v": "v", "adv": "adv", "adj": "adj", "ad": "adj", "a": "adj", "udj": "adj",
    "interj": "interj", "inter j": "interj", "prep": "prep", "conj": "conj", "pron": "pron",
    "num": "num", "art": "art", "aux": "aux", "interrog": "interrog",
}


def canon_pos(raw: str) -> str:
    out: List[str] = []
    for t in re.split(r"[.,;\s]+", raw.lower()):
        t = t.strip()
        if t in POS_CANON and POS_CANON[t] not in out:
            out.append(POS_CANON[t])
    return ",".join(out)


def clean_gloss(s: str, limit: int = 240) -> str:
    s = squash(fix_quotes(s))
    s = re.sub(r"\s+([,;.:!?])", r"\1", s)
    s = re.sub(r"(?<=\w)- (?=[a-z])", "", s)  # re-join end-of-line hyphenation
    s = s.strip(" ;,.:-")
    if len(s) > limit:
        cut = s[:limit]
        k = max(cut.rfind(";"), cut.rfind(","), cut.rfind(" "))
        s = cut[:k] if k > limit // 2 else cut
        s = s.strip(" ;,.:-") + " ..."
    return s


def note_str(d: Dict[str, object]) -> str:
    parts = []
    for k, v in d.items():
        if v in (None, "", False):
            continue
        v = squash(str(v)).replace(";", ",").replace("\t", " ")
        parts.append(f"{k}={v}")
    return "; ".join(parts)


class Rec(dict):
    """A raw extracted entry before normalization."""


PARSE_INFO: Dict[str, dict] = {}   # side-channel for per-source parse statistics


def rec(source: str, head: str, variants: Iterable[str] = (), gloss_en: str = "", gloss_fr: str = "",
        pos: str = "", page: object = "", **notes) -> Rec:
    return Rec(source=source, head=head, variants=[v for v in variants if v], gloss_en=gloss_en,
               gloss_fr=gloss_fr, pos=pos, page=str(page or ""), notes=dict(notes))


class PageTracker:
    """Follow running heads / bare page-number lines through an OCR text.

    ``head_pats`` match running-head lines.  Each may capture the page number
    in group 1; when the number is missing or implausible (OCR) the page is
    inferred as previous + 1.  ``bare=True`` additionally treats a line that
    is only digits as a page number (Shaw).  ``is_head(line)`` lets parsers
    drop running heads from entry bodies.
    """

    def __init__(self, head_pats: List[str], bare: bool = False, lo: int = 0, hi: int = 999,
                 start: int = 1, bare_gap: bool = False):
        self.pats = [re.compile(p) for p in head_pats]
        self.bare = bare
        self.bare_gap = bare_gap
        self.lo, self.hi = lo, hi
        self.cur = "" if start is None else str(start)

    def is_head(self, line: str) -> bool:
        s = squash(line)
        return any(p.match(s) for p in self.pats)

    def feed(self, line: str) -> None:
        s = squash(line)
        for p in self.pats:
            m = p.match(s)
            if m:
                prev = int(self.cur) if self.cur.isdigit() else 0
                n = None
                if m.groups() and m.group(1) and m.group(1).isdigit():
                    n = int(m.group(1))
                if n is not None and prev < n <= prev + 3 and self.lo <= n <= self.hi:
                    self.cur = str(n)
                else:
                    self.cur = str(prev + 1)
                return
        if self.bare and re.fullmatch(r"\d{1,3}", s):
            n = int(s)
            prev = int(self.cur) if self.cur.isdigit() else 0
            if self.lo <= n <= self.hi and (not self.cur and self.bare_gap or prev < n <= prev + (12 if self.bare_gap else 3)):
                self.cur = str(n)


# --------------------------------------------------------------------------
# Layout-aware reading (word coordinates from ``*_djvu.xml``)
# --------------------------------------------------------------------------


class Ctx:
    """Gives parsers access to the cached source files."""

    def __init__(self, src_dir: Path):
        self.src_dir = src_dir
        self._lines: Dict[str, List[str]] = {}
        self._xml: Dict[str, list] = {}

    def lines(self, ia_id: str) -> List[str]:
        if ia_id not in self._lines:
            self._lines[ia_id] = read_lines(self.src_dir, ia_id)
        return self._lines[ia_id]

    def xml_pages(self, ia_id: str) -> list:
        if ia_id not in self._xml:
            self._xml[ia_id] = load_xml_pages(source_path(self.src_dir, ia_id, "xml"))
        return self._xml[ia_id]


def load_xml_pages(path: Path) -> List[Tuple[int, int, List[Tuple[str, int, int, int, int]]]]:
    """Return ``[(page_w, page_h, [(text, x0, ytop, x1, ybottom), ...]), ...]``."""
    import xml.etree.ElementTree as ET

    out = []
    root = ET.parse(str(path)).getroot()
    for obj in root.iter("OBJECT"):
        w, h = int(obj.get("width")), int(obj.get("height"))
        words = []
        for wd in obj.iter("WORD"):
            x0, yb, x1, yt = (int(v) for v in wd.get("coords").split(","))
            words.append(((wd.text or "").strip(), x0, yt, x1, yb))
        out.append((w, h, words))
    return out


def find_splits(words, w: int, ncols: int) -> List[int]:
    """x positions of the column gutters: widest empty run in the expected region."""
    cov = [0] * (w + 2)
    for t, x0, y0, x1, y1 in words:
        if x1 - x0 > 0.25 * w:  # titles / rules spanning the gutter
            continue
        for x in range(max(0, x0), min(w, x1) + 1):
            cov[x] += 1
    regions = [(0.38, 0.62)] if ncols == 2 else [(0.22, 0.45), (0.55, 0.80)]
    splits = []
    for a, b in regions:
        lo, hi = int(a * w), int(b * w)
        m = min(cov[lo:hi + 1])
        run: List[int] = []
        best: List[int] = []
        for x in range(lo, hi + 1):
            if cov[x] == m:
                if run and x == run[-1] + 1:
                    run.append(x)
                else:
                    run = [x]
                if len(run) > len(best):
                    best = list(run)
        splits.append(best[len(best) // 2] if best else (lo + hi) // 2)
    return splits


def column_word_lines(words, w: int, ncols: int, splits: Optional[List[int]] = None):
    """Split a page into columns; return ``([[line of word tuples, ...] per column], splits)``."""
    from statistics import median

    sp = splits if splits is not None else find_splits(words, w, ncols)
    cols: List[list] = [[] for _ in range(ncols)]
    for wd in words:
        cx = (wd[1] + wd[3]) / 2
        cols[sum(cx > x for x in sp)].append(wd)
    res: List[List[list]] = []
    for c in cols:
        if not c:
            res.append([])
            continue
        hs = [(t[4] - t[2]) for t in c if t[4] > t[2]]
        mh = median(hs) if hs else 40
        c.sort(key=lambda t: ((t[2] + t[4]) / 2, t[1]))
        lines: List[list] = []
        cur: list = []
        cy = None
        for wd in c:
            yc = (wd[2] + wd[4]) / 2
            if cy is None or abs(yc - cy) <= 0.6 * mh:
                cur.append(wd)
                cy = yc if cy is None else (cy * (len(cur) - 1) + yc) / len(cur)
            else:
                lines.append(cur)
                cur = [wd]
                cy = yc
        if cur:
            lines.append(cur)
        res.append([sorted(l, key=lambda t: t[1]) for l in lines])
    return res, sp


def column_lines(words, w: int, ncols: int, splits: Optional[List[int]] = None) -> Tuple[List[List[str]], List[int]]:
    """Split a page into columns and return each column as a list of text lines."""
    cols, sp = column_word_lines(words, w, ncols, splits)
    return [[" ".join(t[0] for t in l) for l in c] for c in cols], sp


# Language / origin vocabulary used by Gibbs and friends -------------------------------------

ORIGIN_WORDS = (
    r"Chinook|French|Can\.|Canadian|English|Nootka|Chihalis|Chehalis|Klikatat|Nisqually|Clatsop|Yakama|"
    r"Wasco|Jargon|Quaere|Probably|By onoma|Salish|Tokwaht|Clayoquot|Makah|Cowlitz|Sahaptin|Spanish|Cree|"
    r"Lummi|Kwantlen|Nittinat|Flathead|Haida|Tshimsian|Skwale|Wallawalla|Walla|Latin|Greek|Common to|"
    r"Identical|Corrupted|Possibly|Perhaps|From the|Contraction|Hale|Dutch|German|Russian|Chinese|"
    r"Hawaiian|Kanaka|Sandwich|Klamath|Tillamook|Kalapuya|Calapooya|Umpqua|Cayuse|Shahaptin|Kootenay|"
    r"Algonkin|Ojibwa|Siwash"
)

STRICT_POS = r"(?:n|v|adv|adj|ad|interj|inter\s*j|prep|conj|pron|art|num|aux|interrog)"


# --------------------------------------------------------------------------
# Gibbs 1863 and Hibben 1889 (same layout: ``Head, pos. [Origin, FORM.] Gloss. Ex. ...``)
# --------------------------------------------------------------------------

_tokhead = r"[A-Z][A-Za-z'’‘\-]*(?:\s[A-Za-z'’‘\-]+)?"
_gibbs_start = re.compile(
    r"^(?P<head>" + _tokhead + r")(?P<alts>(?:,?\s*or\s+[A-Za-z][A-Za-z'’‘\-]*(?:\s[A-Za-z'’‘\-]+)?)*)"
    r"\s*[,.]\s*(?P<rest>.*)$"
)
_gibbs_pos_strict = re.compile(r"^(?P<pos>" + STRICT_POS + r"\b\.?(?:\s*[,.]?\s*(?:" + STRICT_POS + r")\b\.?)*)\s*(?P<rest>.*)$")
_gibbs_origin = re.compile(r"^(?:" + ORIGIN_WORDS + r")\b")


def _gibbs_is_start(line: str):
    """Return (match, pos, rest, loose) when *line* opens an entry.

    ``loose`` is True for entries recognised without a clean ``n.``/``v.`` marker
    (cross-references, OCR-garbled pos, origin-first); the caller applies an
    alphabetical sanity check to those only.
    """
    m = _gibbs_start.match(line)
    if not m:
        return None
    rest = m.group("rest")
    if re.match(r"^See\b", rest):
        return m, "", rest, True
    pm = _gibbs_pos_strict.match(rest)
    if pm:
        return m, pm.group("pos"), pm.group("rest"), False
    # OCR-garbled pos token (e.g. ``\u00ab.``, ``\u00bb.``, ``11.``, ``71``) followed by an origin word
    gm = re.match(r"^([^\s]{1,7})\s+(.*)$", rest)
    if gm and _gibbs_origin.match(gm.group(2)) and not re.match(r"^[A-Za-z]{4,}", gm.group(1)):
        return m, "?", gm.group(2), True
    if _gibbs_origin.match(rest):
        return m, "", rest, True
    return None


_hib_start = re.compile(
    r"^(?P<head>[A-Z][A-Za-z'\u2019\u2018\.\-]{1,24})(?P<alts>(?:,?\s*or\s+[A-Za-z][A-Za-z'\u2019\u2018\-]{1,24})*)"
    r"(?:,\s*(?P<pos>" + STRICT_POS + r"(?:\s*[.,]\s*" + STRICT_POS + r")*)\.?|\.)[\s]+(?P<rest>\S.*)$"
)
_hib_start_np = re.compile(  # hyphenated head whose full stop was lost to OCR: ``Moo-la A mill.``
    r"^(?P<head>[A-Z][A-Za-z'\u2019\u2018]{0,10}-[A-Za-z'\u2019\u2018\-]{1,20})(?P<alts>)\s+(?P<rest>[<A-Z]\S*.*)$"
)
_NOT_HEAD = {"ex", "ques", "ans", "q", "a", "see", "the", "mr", "dr", "st", "no", "so"}


def _hibben_is_start(line: str):
    m = _hib_start.match(line)
    if not m:
        m = _hib_start_np.match(line)
        if m:
            h = re.sub(r"[^a-z]", "", m.group("head").lower())
            if len(h) < 4:
                return None
            return m, "", m.group("rest"), True
        return None
    h = re.sub(r"[^a-z]", "", m.group("head").lower())
    if h in _NOT_HEAD or len(h) < 2:
        return None
    return m, (m.group("pos") or ""), m.group("rest"), True


_ABBR = re.compile(r"\b(Can|Fr|imp|pl|sing|Mr|Dr|St|Capt|Col|lit|esp|etc|viz|cf|sc|no|vol|ibid|onoma|Prob|Chin|Eng|Fr\. Can)\.|\bu\. ?d\.|\bi\. ?e\.|\be\. ?g\.")


def _protect(s: str) -> str:
    return _ABBR.sub(lambda m: m.group(0).replace(".", "\x01"), s)


def _unprotect(s: str) -> str:
    return s.replace("\x01", ".")


def _etym_split(rest: str) -> Tuple[str, str]:
    """Split ``Origin, FORM. Gloss ...`` into (etymology, remainder) heuristically."""
    rest = _protect(rest.strip())
    etym_parts: List[str] = []
    while rest:
        m = re.match(r"^(.{1,160}?[.])(\s+|$)", rest)
        if not m:
            break
        sent = m.group(1)
        looks_etym = bool(_gibbs_origin.match(sent) or re.match(r"^\((?:Hale|Anderson|Shaw|Tolmie|Pandosy|Cook|Jewitt)", sent)
                          or re.match(r"^(?:idem|Not Jargon|Of local use)", sent))
        has_caps = bool(re.search(r"\b[A-Z][A-Z'\-]{2,}\b|\bidem\b", sent))
        short_lang = bool(re.fullmatch(r"(?:Can\x01 )?(?:" + ORIGIN_WORDS + r")(?: (?:and|or) \w+)?\.?", sent))
        if looks_etym and (has_caps or short_lang or len(sent) < 40):
            etym_parts.append(sent)
            rest = rest[m.end():].lstrip()
            continue
        break
    return _unprotect(" ".join(etym_parts)), _unprotect(rest)


def _gloss_from(rest: str) -> str:
    rest = re.split(r"\s(?:Ex\.|Example:|Ques\.|Q\.\s)", " " + rest)[0]
    rest = _protect(rest.strip())
    parts = re.split(r"(?<=[a-z\)])\.\s+(?=[A-Z])", rest)
    g = parts[0] if parts else rest
    if len(g) < 12 and len(parts) > 1:
        g = g + ". " + parts[1]
    return clean_gloss(_unprotect(g))


def _split_alts(head: str, alts: str) -> List[str]:
    out = [head.strip()]
    for a in re.split(r",?\s*or\s+", alts or ""):
        a = a.strip(" ,.")
        if a:
            out.append(a)
    return out


def parse_gibbs_like(lines: List[str], key: str, start_pat: str, end_pat: str, page_pats: List[str],
                     with_etym: bool, first_page: int = 1, is_start=None, window: bool = False) -> List[Rec]:
    s = next(i for i, l in enumerate(lines) if re.search(start_pat, squash(l)))
    e = next(i for i in range(s + 1, len(lines)) if re.search(end_pat, squash(lines[i])))
    is_start = is_start or _gibbs_is_start
    pt = PageTracker(page_pats, lo=1, hi=200, start=first_page)
    starts: List[Tuple[int, re.Match, str, str]] = []
    letter = "a"
    recent: List[str] = []
    for i in range(s, e):
        if i > s:
            pt.feed(lines[i])
        l = squash(lines[i])
        if not l or pt.is_head(l):
            continue
        hm = re.fullmatch(r"([A-Z])[.,]?", l)
        if hm:
            letter = hm.group(1).lower()
            continue
        r = is_start(l)
        if not r:
            continue
        m, pos, rest, loose = r
        h = re.sub(r"[^a-z]", "", m.group("head").lower())
        if len(h) < 2:
            continue
        if window:
            if recent and not (min(recent) <= h[0] <= chr(ord(max(recent)) + 4)):
                continue
            recent = (recent + [h[0]])[-5:]
        elif loose and not (letter <= h[0] <= chr(ord(letter) + 2)):
            continue
        starts.append((i, m, pos, pt.cur))
    out: List[Rec] = []
    for idx, (i, m, pos, page) in enumerate(starts):
        j = starts[idx + 1][0] if idx + 1 < len(starts) else e
        body_lines = []
        for k in range(i, j):
            t = squash(lines[k])
            if pt.is_head(t) or re.fullmatch(r"[A-Z]\.?", t):
                continue
            body_lines.append(t)
        text = " ".join(body_lines)
        mm = is_start(text)
        if not mm:
            continue
        m2, pos2, rest2, _loose = mm
        variants = _split_alts(m2.group("head"), m2.group("alts"))
        pos_c = canon_pos(pos2) if pos2 and pos2 != "?" else ""
        notes: Dict[str, object] = {}
        if pos2 == "?":
            notes["pos_ocr"] = "garbled"
        if re.match(r"^See\b", rest2):
            xref = re.match(r"^See\s+(.+?)\.?\s*$", rest2)
            r_ = rec(key, variants[0], variants, gloss_en="", pos=pos_c, page=page,
                     xref=xref.group(1) if xref else rest2, type="crossref")
            out.append(r_)
            continue
        if with_etym:
            etym, remainder = _etym_split(rest2)
        else:
            etym, remainder = "", rest2
        gloss = _gloss_from(remainder)
        notes["etym"] = clean_gloss(etym, 100) if etym else ""
        out.append(rec(key, variants[0], variants, gloss_en=gloss, pos=pos_c, page=page, **notes))
    return out


_GIBBS_HEADS = [
    r"^(?:(?:\d{1,3}|\S{1,3})[:;.]?\s+)?DICTIONARY\s+OF\s+TH\S*\s+\S*CHINOOK\s+\S*\s*\W*$",
    r"^PA[ER]T\s+[IiJl1]\W{0,3}\s*CHINOOK\W*\s*\w{0,10}\W{0,3}\s*(?:(\d{1,3}|[A-Z]|\d\)))?\W*$",
]


def parse_gibbs(lines: List[str]) -> List[Rec]:
    return parse_gibbs_like(
        lines, "gibbs1863",
        start_pat=r"^PART I\.?\s*CHINOOK[-\u2014 ]+ENGLISH\.?$",
        end_pat=r"(?i)^ENGLISH[-\u2014 ]CHINOOK\.?$",
        page_pats=_GIBBS_HEADS, with_etym=True, first_page=1,
    )


def parse_hibben(lines: List[str]) -> List[Rec]:
    return parse_gibbs_like(
        lines, "hibben1889",
        start_pat=r"^PART I\s*\W?\s*CHINOOK\W*ENGLISH\.?$",
        end_pat=r"(?i)^PART\s+I[ILT1]\W*\s*ENGLISH\W+CHINOOK\.?$",
        page_pats=_GIBBS_HEADS, with_etym=False, first_page=3, is_start=_hibben_is_start, window=True,
    )


# --------------------------------------------------------------------------
# Shaw 1909
# --------------------------------------------------------------------------

_SHAW_TOK = r"[A-Z][A-Za-z'\u2019\u2018\-]*(?:\s[A-Za-z'\u2019\u2018\-]+)?"
_SHAW_POS = r"(?:n|v|adv|adj|ad|interj|inter\s*j|prep|conj|pron|art|num|aux|interrog)"
_shaw_start = re.compile(
    r"^(?P<head>" + _SHAW_TOK + r")(?P<alts>(?:,?\s+or\s+[A-Za-z][A-Za-z'\u2019\u2018\-]*(?:\s[A-Za-z'\u2019\u2018\-]+)?)*)"
    r"\s*[,.]?\s+(?P<pos>" + _SHAW_POS + r"\.(?:\s*[,.]?\s*" + _SHAW_POS + r"\.)*)\s*(?P<rest>.*)$"
)
_shaw_see = re.compile(r"^(?P<head>" + _SHAW_TOK + r")\.\s+\(See\s+(?P<tgt>[^)]+)\)")
_ORIGIN_TAG = re.compile(r"^\(\s*(?P<tag>[CEFNSJPQ0]|Canadian F|Fr\. ?Canadian|Quaere[^)]{0,30}|\?|Wasco\.?)\s*(?:[\)\]]|[I1l](?=[.\s(]))\.?\s*")
SHAW_ORIGIN = {"C": "Chinook", "E": "English", "F": "French", "N": "Nootka", "S": "Salish", "J": "Jargon",
               "P": "French", "0": "Chinook", "Q": "uncertain"}


def _balanced_paren(text: str, maxlen: int = 400) -> Optional[int]:
    """Index just past the ``)`` closing the ``(`` at text[0], or None."""
    depth = 0
    for i, ch in enumerate(text[:maxlen]):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def _shaw_body(rest: str) -> Tuple[str, str, str]:
    """Return (origin_tag, etymology, gloss) from the text after the pos list."""
    rest = rest.strip()
    origin = ""
    etym = ""
    for _ in range(3):
        if not rest.startswith("("):
            # opening paren lost to OCR: ``Chihalis,-Yaiem.) A story``
            um = re.match(r"^((?:[A-Z][a-z]+|Fr\.|Eng\.)[,.]\s*-?[^()]{0,90}?)\)\.?\s*", rest)
            if um and not etym:
                etym = um.group(1)
                rest = rest[um.end():]
                continue
            break
        m = _ORIGIN_TAG.match(rest)
        if m and not origin:
            origin = SHAW_ORIGIN.get(m.group("tag"), m.group("tag"))
            rest = rest[m.end():].lstrip()
            continue
        end = _balanced_paren(rest)
        if end is None:
            # etymology wrapped past the end of what we saw: cut at first ``.)`` or ``)``
            k = rest.find(")")
            end = k + 1 if 0 < k < 200 else None
        if end is None:
            break
        etym = etym or rest[1:end - 1]
        rest = rest[end:].lstrip(" .")
    rest = re.sub(r"^[.\s]+", "", rest)
    g = re.split(r"\s(?:Examples?:|Ex\.:?|Ex\.\s|Note\b)", " " + rest)[0]
    g = _protect(g.strip())
    parts = re.split(r"(?<=[a-z\)])\.\s+(?=[A-Z\(\"])", g)
    first = parts[0] if parts else g
    if len(first) < 14 and len(parts) > 1:
        first = first + ". " + parts[1]
    return origin, clean_gloss(etym, 100), clean_gloss(_unprotect(first))


def _dehyphenate(text: str) -> str:
    return re.sub(r"(?<=[a-z])- (?=[a-z])", "", text)


def parse_shaw(lines: List[str]) -> List[Rec]:
    key = "shaw1909"
    s = next(i for i, l in enumerate(lines) if re.fullmatch(r'A"?\s+LEXICON', squash(l)))
    e = next(i for i, l in enumerate(lines) if squash(l).startswith("SUPPLEMENTAL") and "VOCAB" in squash(l))
    e2 = next(i for i in range(e + 1, len(lines)) if squash(lines[i]).startswith("PRONOUNCING") and "VOCAB" in squash(lines[i]))
    out: List[Rec] = []

    # ---- main Lexicon ----
    pt = PageTracker([], bare=True, lo=1, hi=60, start=None, bare_gap=True)
    starts = []
    for i in range(s, e):
        pt.feed(lines[i])
        l = squash(lines[i])
        if not l:
            continue
        m = _shaw_start.match(l)
        if m:
            starts.append((i, "entry", pt.cur))
            continue
        m = _shaw_see.match(l)
        if m:
            starts.append((i, "see", pt.cur))
    for idx, (i, kind, page) in enumerate(starts):
        j = starts[idx + 1][0] if idx + 1 < len(starts) else e
        body = []
        for k in range(i, j):
            t = squash(lines[k])
            if re.fullmatch(r"\d{1,3}", t) or re.fullmatch(r"THE CHINOOK JARGON\.?", t) or re.fullmatch(r"AND HOW TO USE IT\.?", t):
                continue
            body.append(t)
        text = _dehyphenate(" ".join(body))
        if kind == "see":
            m = _shaw_see.match(text)
            if m:
                out.append(rec(key, m.group("head"), [m.group("head")], page=page, xref=m.group("tgt"), type="crossref",
                               section="lexicon"))
            continue
        m = _shaw_start.match(text)
        if not m:
            continue
        variants = _split_alts(m.group("head"), m.group("alts"))
        origin, etym, gloss = _shaw_body(m.group("rest"))
        out.append(rec(key, variants[0], variants, gloss_en=gloss, pos=canon_pos(m.group("pos")), page=page,
                       origin=origin, etym=etym, section="lexicon"))

    # ---- Supplemental vocabulary ----
    sup = re.compile(
        r"^(?P<head>" + _SHAW_TOK + r")(?P<alts>(?:,?\s*or\s+[A-Za-z][A-Za-z'\u2019\u2018\-]*)*)\s*,\s*"
        r"\(\s*(?P<tag>[^)]{1,24})\)[.,]?\s*(?P<rest>.*)$"
    )
    pt = PageTracker([], bare=True, lo=1, hi=60, start=int(out[-1]["page"] or 30) if out else 30, bare_gap=True)
    cur: Optional[dict] = None
    sup_recs: List[dict] = []
    for i in range(e + 1, e2):
        pt.feed(lines[i])
        l = squash(lines[i])
        if not l or re.fullmatch(r"[A-Z]", l) or re.fullmatch(r"\d{1,3}", l) or l.startswith("Less Familiar"):
            continue
        m = sup.match(l)
        if m:
            cur = dict(head=m.group("head"), alts=m.group("alts"), tag=m.group("tag"), gloss=m.group("rest"), page=pt.cur)
            sup_recs.append(cur)
        elif cur is not None:
            cur["gloss"] += " " + l
    for d in sup_recs:
        variants = _split_alts(d["head"], d["alts"])
        tag = d["tag"].strip(" .")
        origin = SHAW_ORIGIN.get(tag, "Wasco" if tag.startswith("Wasco") else ("French (Canadian)" if tag.startswith("Canadian") else tag))
        g = clean_gloss(_dehyphenate(d["gloss"]))
        out.append(rec(key, variants[0], variants, gloss_en=g, page=d["page"], origin=origin, section="supplemental"))
    return out


# --------------------------------------------------------------------------
# Gill 1909 / Gill 1887   ``Head (tag). Gloss. "Example"``
# --------------------------------------------------------------------------

_GILL_TAG = {"J": "Jargon", "S": "Salish", "N": "Nootka", "F": "French", "O. C": "Old Chinook", "OC": "Old Chinook",
             "Ch": "Chinook", "C": "Chinook", "W": "Wasco", "E": "English", "?": "uncertain", "Q": "uncertain"}
_gill_tok = r"[A-Z][A-Za-z0-9'\u2019\u2018\-\u00c0-\u017f\.\^\"]{0,30}"
_gill_start = re.compile(
    r"^(?P<head>" + _gill_tok + r"(?:\s[a-z][A-Za-z0-9'\u2019\u2018\-\u00c0-\u017f]{1,20}){0,3}?)"
    r"(?P<alts>(?:\s*,\s*(?:or\s+)?(?:more commonly\s+)?[A-Za-z][A-Za-z0-9'\u2019\u2018\-\u00c0-\u017f]{1,30}"
    r"(?:\s[a-z][A-Za-z0-9'\u2019\u2018\-\u00c0-\u017f]{1,20}){0,2})*)"
    r"\s*(?:\((?P<tag>[^)]{1,12})\)\s*[.!]?|[.!])\s+(?P<rest>\S.*)$"
)
# In the 1909 text OCR sometimes renders accented vowels as digits / capitals inside a word.
_GILL_OCR = str.maketrans({"4": "a", "6": "o", "0": "o", "1": "l"})


def _gill_clean_head(h: str) -> Tuple[str, bool]:
    fixed = False
    out = []
    for i, ch in enumerate(h):
        if ch in "4609" and 0 < i:
            out.append({"4": "a", "6": "o", "0": "o", "9": "g"}.get(ch, ch))
            fixed = True
        elif ch == "S" and 0 < i < len(h) - 1 and h[i - 1].islower():
            out.append("a")
            fixed = True
        else:
            out.append(ch)
    return "".join(out), fixed


def parse_gill(lines: List[str], key: str, start_pat: str, end_pat: str, bare_lo: int, bare_hi: int,
               first_page: Optional[int], allow_phrase: bool) -> List[Rec]:
    s = next(i for i, l in enumerate(lines) if re.search(start_pat, squash(l)))
    e = next(i for i in range(s + 1, len(lines)) if re.search(end_pat, squash(lines[i])))
    pt = PageTracker([r"^(\d{1,3})\s+CHINOOK\s+DICTIONARY\.?$", r"^CHINOOK\s+DICTIONARY\.?\s+(\d{1,3})$"],
                     bare=True, lo=bare_lo, hi=bare_hi, start=first_page, bare_gap=True)
    starts: List[Tuple[int, str, str]] = []
    recent: List[str] = []
    for i in range(s + 1, e):
        pt.feed(lines[i])
        l = squash(fix_quotes(lines[i]))
        if not l or pt.is_head(l) or re.fullmatch(r"\d{1,3}", l) or re.fullmatch(r"CHINOOK\s+DICTIONARY\.?", l):
            continue
        m = _gill_start.match(l)
        if not m:
            continue
        head = m.group("head")
        if " " in head.strip() and not allow_phrase:
            continue
        h = re.sub(r"[^a-z]", "", _gill_clean_head(head)[0].lower())
        if len(h) < 2 or h in _NOT_HEAD:
            continue
        if recent and not (min(recent) <= h[0] <= chr(ord(max(recent)) + 4)):
            continue
        recent = (recent + [h[0]])[-5:]
        starts.append((i, pt.cur, l))
    out: List[Rec] = []
    for idx, (i, page, _first) in enumerate(starts):
        j = starts[idx + 1][0] if idx + 1 < len(starts) else e
        body = []
        for k in range(i, j):
            t = squash(fix_quotes(lines[k]))
            if not t or pt.is_head(t) or re.fullmatch(r"\d{1,3}", t) or re.fullmatch(r"CHINOOK\s+DICTIONARY\.?", t) \
                    or re.fullmatch(r"[A-Z]{1,2}", t):
                continue
            body.append(t)
        text = _dehyphenate(" ".join(body))
        m = _gill_start.match(text)
        if not m:
            continue
        head_raw = m.group("head").strip()
        head_fix, fixed = _gill_clean_head(head_raw)
        variants = [head_raw]
        for a_ in re.split(r"\s*,\s*(?:or\s+)?(?:more commonly\s+)?", m.group("alts") or ""):
            a_ = a_.strip(" ,")
            if a_:
                variants.append(a_)
        tag = (m.group("tag") or "").strip(" .")
        rest = m.group("rest")
        notes: Dict[str, object] = {}
        if tag:
            notes["origin"] = _GILL_TAG.get(tag, tag)
        if fixed:
            notes["ocr"] = "head digit/capital read as accented vowel"
            variants = [head_fix] + variants
        xr = re.match(r"^See\s+(.+?)\.?\s*$", rest)
        if xr:
            notes.update(xref=xr.group(1), type="crossref")
            out.append(rec(key, head_fix, variants, page=page, **notes))
            continue
        # drop a second leading tag ``(S.)`` that follows the head's period
        tm = re.match(r"^\(([A-Za-z?. ]{1,6})\)\s*[.:]?\s*", rest)
        if tm and not tag:
            notes["origin"] = _GILL_TAG.get(tm.group(1).strip(" ."), tm.group(1))
            rest = rest[tm.end():]
        if " " in head_fix.strip():
            notes["type"] = "phrase"
        g = _protect(rest)
        g = re.split(r'\s(?=")', g)[0]  # examples start with a quote mark
        parts = re.split(r"(?<=[a-z\)])\.\s+(?=[A-Z\"])", g)
        first = parts[0]
        if len(first) < 14 and len(parts) > 1:
            first += ". " + parts[1]
        out.append(rec(key, head_fix, variants, gloss_en=clean_gloss(_unprotect(first)), page=page, **notes))
    return out


def parse_gill1909(lines: List[str]) -> List[Rec]:
    return parse_gill(lines, "gill1909", start_pat=r"^CHINOOK[-\u2014 ]ENGLISH$", end_pat=r"^CONVERSATIONAL PHRASES$",
                      bare_lo=40, bare_hi=140, first_page=47, allow_phrase=False)


def parse_gill1887(lines: List[str]) -> List[Rec]:
    return parse_gill(lines, "gill1887", start_pat=r"^\W?HINOOK[-\u2014 ]ENGLISH\W*$", end_pat=r"^(?:Yoot-skut|Yoot-skut\.) +Short",
                      bare_lo=30, bare_hi=60, first_page=31, allow_phrase=True)


# --------------------------------------------------------------------------
# Le Jeune 1886, Practical Chinook Vocabulary (two-column pages -> read by column)
# --------------------------------------------------------------------------

_P86_NOISE = re.compile(r"[|_~\\\u2014\u2013=\[\]{}<>\u00b7]+")
_P86_SECT = re.compile(r"^[\s\-\u2014\u2013_.|:]*(?:[IVXLil1]{1,7}[.,]?|\d{1,2}|[\-\u2014\u2013]+\d+[\-\u2014\u2013]+)\s*$")
_P86_HYPH_END = re.compile(r"[-\u2010\u2011\u2012\u2013\u2014*]\s*$")
_P86_START = re.compile(r"^(?:[\u201d\"']{1,3}\s*)?[A-Za-z][A-Za-z'\u2018\u2019\u2018\-\. ]{0,26}?,\s*\S")
def _p86_entries(col_lines: List[str]) -> List[str]:
    entries: List[List[str]] = []
    cur: Optional[List[str]] = None
    for raw in col_lines:
        t = squash(_P86_NOISE.sub(" ", fix_quotes(raw).replace("\u201d", '"')))
        t = t.strip(" .'`\u2018")
        if not t or len(t) < 2:
            continue
        if _P86_SECT.match(t):
            cur = None
            continue
        incomplete = cur is not None and (_P86_HYPH_END.search(cur[-1]) or not re.search(r"[.!?]\s*$", cur[-1]))
        starts = bool(_P86_START.match(t))
        if starts and not (incomplete and not re.match(r"^[A-Z\"]", t)):
            cur = [t]
            entries.append(cur)
        elif cur is not None:
            cur.append(t)
    out = []
    for e in entries:
        txt = ""
        for piece in e:
            if txt and _P86_HYPH_END.search(txt):
                txt = _P86_HYPH_END.sub("", txt) + piece
            else:
                txt = (txt + " " + piece).strip()
        out.append(txt)
    return out


_P86_FUSE = re.compile(r"(?<=[a-z0-9)\.,\u00e9\u00e8])\s+(?=[A-Z][\w'\u2019\-]*(?:[ \-][\w'\u2019\-]+){0,2},\s+\S)")
_P86_PROPER = {"God", "Jesus", "Christ", "Indian", "Indians", "American", "English", "French", "Frenchman", "Englishman",
               "Canadian", "Mary", "Virgin", "Holy", "Lord", "Sunday", "Boston", "Bishop", "Pope", "Father", "King"}


def _p86_split(txt: str):
    """Return a list of ``(ditto, head, gloss)``; a column line may fuse several entries."""
    m = re.match(r"^(?P<ditto>[\"']{1,3}\s*)?(?P<head>[^,]+?),\s*(?P<gloss>.+)$", txt)
    if not m:
        return []
    head = squash(m.group("head")).strip(" .*'\"")
    pieces = _P86_FUSE.split(m.group("gloss"))
    out = [(bool(m.group("ditto")), head, pieces[0])]
    rejoin = False
    for pc in pieces[1:]:
        mm = re.match(r"^(?P<h>[^,]+?),\s*(?P<g>.+)$", pc)
        if not mm or mm.group("h").split()[0] in _P86_PROPER:
            out[-1] = (out[-1][0], out[-1][1], out[-1][2] + " " + pc)
            continue
        out.append((False, squash(mm.group("h")).strip(" .*'\""), mm.group("g")))
    return out


def _p86_page_entries(pages, first: int, last: int, numbers_page: Optional[int]) -> List[dict]:
    res: List[dict] = []
    for pi in range(first, last + 1):
        w, h, words = pages[pi]
        cols, sp = column_lines(words, w, 2)
        if pi in (18, 19):  # religious words: left column = Chinook, right column = English, aligned rows
            continue
        for ci, c in enumerate(cols):
            for txt in _p86_entries(c):
                for ditto, head, gloss in _p86_split(txt):
                    res.append(dict(page=pi, ditto=ditto, head=head, gloss=gloss, raw=txt, col=ci))
    return res


def parse_practical1886(ctx: "Ctx") -> List[Rec]:
    key = "practical1886"
    pages = ctx.xml_pages("Ayer_PM848_L4_1886")
    book_page = {i: i - 2 for i in range(3, 20)}  # xml page 4 = printed p.1 (vocabulary begins)
    recs: List[Rec] = []
    # ---- standard pages (4 contains the numbers block - handled separately, but its lower half is standard)
    ents = _p86_page_entries(pages, 5, 17, None)
    # page 4: split numbers block (y < line of ``Moon``) from the rest
    w, h, words = pages[4]
    ymoon = min((wd[2] for wd in words if re.match(r"(?i)^(moon|stars|sky)\W*$", wd[0])), default=None)
    ytop = min((wd[2] for wd in words if re.match(r"(?i)^(sa.?hale|hlehe|elehe)", wd[0])), default=ymoon)
    if ytop is None:
        ytop = int(h * 0.74)
    num_words = [wd for wd in words if 1100 < wd[2] < ytop - 40]
    rest_words = [wd for wd in words if wd[2] >= ytop - 40]
    sp4 = find_splits([wd for wd in words if wd[2] > 1100], w, 2)
    ncols, _ = column_lines(num_words, w, 2, splits=sp4)
    rcols, _ = column_lines(rest_words, w, 2, splits=sp4)
    for ci, c in enumerate(rcols):
        for txt in _p86_entries(c):
            for ditto, head, gloss in _p86_split(txt):
                ents.append(dict(page=4, ditto=ditto, head=head, gloss=gloss, raw=txt, col=ci))
    # numbers block: ``English, Chinook[, or alt].`` (English first), 15 items in fixed order
    num_items: List[str] = []
    for c in ncols:
        cur = None
        for raw in c:
            t = squash(_P86_NOISE.sub(" ", fix_quotes(raw))).strip(" .:*")
            if not t or t in ("?", "....", "y;", "4;", "b]", "Ly"):
                continue
            if re.match(r"^[|i]?\s*[A-Za-z0-9]{2,9}(?: hundred)?,\s", t) or re.match(r"^[A-Za-z]{3,9},\s*$", t):
                cur = [t]
                num_items.append(cur)
            elif cur is not None:
                cur.append(t.lstrip("- "))
    num_gloss = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
                 "twelve", "twenty", "thirty", "one hundred"]
    if len(num_items) == len(num_gloss):
        for it, eng in zip(num_items, num_gloss):
            txt = " ".join(it)
            txt = re.sub(r"(?<=\w)-\s+", "", txt)
            m = re.match(r"^[|i]?\s*[A-Za-z0-9]{2,9}(?: hundred)?,\s*(.+)$", txt)
            if not m:
                continue
            chin = " ".join(tok for tok in m.group(1).split()
                            if not re.fullmatch(r"(?:Ly|y;|b\]|4;|\*\d\)|\d+\)?|[A-Za-z]\W?)", tok) or tok in ("pi", "or"))
            chin = re.sub(r"[.:;*]+", " ", chin)
            chin = re.sub(r"\bor\b", ",", chin)
            alts = [squash(a).strip(" ,") for a in chin.split(",") if squash(a).strip(" ,")]
            alts = [a for a in alts if re.fullmatch(r"[A-Za-z][A-Za-z' ]{1,22}", a)]
            if not alts:
                continue
            recs.append(rec(key, alts[0], alts, gloss_en=eng, page=2, type="number",
                            conf="B", ocr="numbers block read English-first, glosses assigned in order"))
    # ---- convert entries
    for e in ents:
        head, gloss = e["head"], e["gloss"]
        bp = str(book_page.get(e["page"], ""))
        g = clean_gloss(_dehyphenate(gloss), 120)
        if e["ditto"]:
            head = "Mamook " + head
        if not re.fullmatch(r"[A-Za-z][A-Za-z'\u2018\u2019\-\. ]{1,36}", head) or not g:
            continue
        notes = {"raw": e["raw"] if len(e["raw"]) < 90 else ""}
        if e["ditto"]:
            notes["ditto"] = "mamook (from ditto mark)"
        if " " in head.strip():
            notes["type"] = "phrase"
        recs.append(rec(key, head, [head], gloss_en=g, page=bp, **notes))
    # ---- pages 18-19: religious words, rows aligned by y
    for pi in (18, 19):
        w, h, words = pages[pi]
        cols, sp = column_lines(words, w, 2)
        lw = [wd for wd in words if (wd[1] + wd[3]) / 2 <= sp[0]]
        rw = [wd for wd in words if (wd[1] + wd[3]) / 2 > sp[0]]
        def rows(ws):
            ws = sorted(ws, key=lambda t: ((t[2] + t[4]) / 2))
            out_, cur_, cy = [], [], None
            hh = [t[4] - t[2] for t in ws if t[4] > t[2]]
            mh = sorted(hh)[len(hh) // 2] if hh else 40
            for t in ws:
                yc = (t[2] + t[4]) / 2
                if cy is None or abs(yc - cy) <= 0.7 * mh:
                    cur_.append(t); cy = yc if cy is None else (cy + yc) / 2
                else:
                    out_.append((cy, " ".join(x[0] for x in sorted(cur_, key=lambda x: x[1])))); cur_ = [t]; cy = yc
            if cur_:
                out_.append((cy, " ".join(x[0] for x in sorted(cur_, key=lambda x: x[1]))))
            return out_
        L_, R_ = rows(lw), rows(rw)
        for yl, tl in L_:
            near = [(abs(yr - yl), tr) for yr, tr in R_ if abs(yr - yl) < 90]
            if not near:
                continue
            tr = min(near)[1]
            head = squash(re.sub(r"[^A-Za-z'\u2019\- ,]", "", tl)).strip(" ,-")
            gl = clean_gloss(re.sub(r"[^A-Za-z.'\u2019\- ,]", "", tr))
            if len(head) < 3 or len(gl) < 3 or not re.match(r"^[A-Z]", head):
                continue
            hv = [x.strip() for x in head.split(",") if x.strip()]
            recs.append(rec(key, hv[0], hv, gloss_en=gl, page=str(book_page.get(pi, "")), type="religious",
                            conf="B", ocr="row-aligned two-column table"))
    return recs


# --------------------------------------------------------------------------
# Le Jeune 1924, Chinook Rudiments (lithographed handwriting -> very noisy OCR)
# --------------------------------------------------------------------------
#
# The vocabulary lists are lines like ``Kah, \u00ab shorthand-OCR junk \u00bb, where,``: Latin headword,
# glyph noise, English gloss.  The parser only extracts candidates; the build's validation pass
# (``validate_noisy``) grades each one against the clean dictionaries.

_RUD_SECTIONS = [
    # (key, start regex, end regex, strict)  - strict sections keep validated (A/B) entries only
    ("grammar-lists", r"(?i)^\s*Phonetic Alphabet", r"(?i)The 163 Original words", True),
    ("163-original-words", r"(?i)The 163 Original words", r"(?i)22 Chim\w* words", False),
    ("22-more-words", r"(?i)22 Chim\w* words", r"(?i)french words", False),
    ("french-words", r"(?i)french words", r"(?i)Reli\w+ Words", False),
    ("religious-words", r"(?i)Reli\w+ Words", r"(?i)(?:Conon|Common|Comm?en)\s+\w*\s*words", False),
    ("other-districts", r"(?i)Lust of words used", r"(?i)More eX\s*erc", False),
    ("sound-words-districts", r"(?i)^\s*Used im othe districts", r"(?i)^\s*Remarks\.", False),
    ("durieu-1879-vocabulary", r"(?i)Practical Vocabulon", r"(?i)^\s*Notice\.", False),
]
_RUD_ENG_STOP = {
    "the", "and", "of", "to", "is", "a", "an", "in", "on", "it", "he", "she", "we", "you", "i", "as", "at", "by",
    "for", "or", "if", "so", "no", "not", "are", "was", "be", "this", "that", "with", "from", "his", "her",
    "my", "me", "our", "your", "their", "its", "all", "any", "but", "can", "has", "have", "had", "do", "did",
    "will", "shall", "one", "two", "ex", "see", "note", "then", "when", "where", "who", "what", "how", "why",
}
_RUD_LINE = re.compile(r"^[^A-Za-z]{0,3}(?P<h>[A-Za-z][A-Za-z'\u2019\u2018\-\. ]{0,26}?)\s*[,;]\s*(?P<rest>.*)$")
_RUD_GLOSS = re.compile(r"^(?:to |a |an |the |of )?[A-Za-z][A-Za-z '\-/]{1,38}$")


def _rud_gloss(rest: str) -> str:
    segs = [squash(x).strip(" .;:!?'\"-_") for x in re.split(r"[,;]", fix_quotes(rest))]
    for g in reversed(segs):
        letters = re.sub(r"[^A-Za-z]", "", g)
        if len(letters) >= 3 and re.search(r"[aeiouy]", letters.lower()) and _RUD_GLOSS.match(g):
            if len(g.split()) <= 4 and not re.search(r"[A-Z].*[A-Z]", g[1:]):
                return g
    return ""


def parse_rudiments1924(lines: List[str]) -> List[Rec]:
    key = "rudiments1924"
    recs: List[Rec] = []
    for sec, sp, ep, strict in _RUD_SECTIONS:
        sre, ere = re.compile(sp), re.compile(ep)
        i0 = next((i for i, l in enumerate(lines) if sre.search(l)), None)
        if i0 is None:
            continue
        i1 = next((i for i in range(i0 + 1, len(lines)) if ere.search(lines[i])), len(lines))
        for ln in range(i0 + 1, i1):
            raw = squash(fix_quotes(lines[ln]))
            if len(raw) < 4 or len(raw) > 90:
                continue
            m = _RUD_LINE.match(raw)
            if not m:
                continue
            head = squash(m.group("h")).strip(" .-'")
            toks = head.split()
            if not (1 <= len(toks) <= 3) or len(re.sub(r"[^A-Za-z]", "", head)) < 2:
                continue
            if len(toks) == 1 and toks[0].lower() in _RUD_ENG_STOP:
                continue
            if any(len(t) > 18 for t in toks):
                continue
            gloss = _rud_gloss(m.group("rest"))
            recs.append(rec(key, head, [head], gloss_en=gloss, page="", section=sec, line=ln + 1,
                            raw=raw if len(raw) < 80 else "", strict="yes" if strict else "",
                            gloss_state="" if gloss else "unrecovered"))
    return recs


# --------------------------------------------------------------------------
# Demers & Blanchet 1871 (St-Onge ed.), Chinook Dictionary - two text columns per page
# --------------------------------------------------------------------------

_DEM_HEAD = re.compile(
    r"(?:(?<=\s)|^)(?:(?P<dit>[\u201d\"\u00bb%]\s*)(?P<dh>[a-z][a-z'\u2019\-]{1,17})|(?P<h>[A-Z][A-Za-z'\u2019\-]{1,17}))"
    r"(?P<alt>(?:\s+or\s+[A-Za-z][A-Za-z'\u2019\-]{1,17})*)\s*,\s+")
_DEM_STOP = {"Bible", "Indian", "Indians", "English", "French", "Sunday", "Catholic", "Christmas", "God", "Lord", "See",
             "Jesus", "Mary", "Holy", "Ember", "Latin", "Canadian", "American"}


def parse_demers1871(ctx: "Ctx") -> List[Rec]:
    key = "demers1871"
    pages = ctx.xml_pages("cihm_04222")
    recs: List[Rec] = []
    for pi in range(21, 39):
        w, h, words = pages[pi]
        if not words:
            continue
        cols, sp = column_lines(words, w, 2)
        printed = pi - 13  # xml page 21 = printed page 15? (running numbers are OCR'd inconsistently)
        for ci, c in enumerate(cols):
            txt = ""
            for ln in c:
                ln = squash(fix_quotes(ln).replace("\u201d", '"'))
                if re.fullmatch(r"[\d\W]{0,4}", ln):
                    continue
                if re.fullmatch(r"(?:ADJECTIVES|VERBS|ADVERBS|CONJUNCTIONS|PREPOSITIONS|NOUNS|NUMBERS|PRONOUNS|INTERJECTIONS)[,.]?", ln.strip(" .")):
                    continue
                if txt.endswith("-") and ln and ln[0].islower():
                    txt = txt[:-1] + ln
                else:
                    txt += " " + ln
            txt = txt.strip()
            ms = list(_DEM_HEAD.finditer(txt))
            last_head = ""
            for k, m in enumerate(ms):
                end = ms[k + 1].start() if k + 1 < len(ms) else len(txt)
                gl = txt[m.end():end]
                if m.group("h"):
                    head = m.group("h")
                    if head in _DEM_STOP:
                        continue
                    last_head = head
                    ditto = False
                else:
                    head = (last_head + " " + m.group("dh")) if last_head else m.group("dh")
                    ditto = True
                alts = [a.strip() for a in re.split(r"\s+or\s+", m.group("alt")) if a.strip()]
                gl = re.split(r"(?<=[a-z\)])\.(?:\s|$)", gl)[0]
                g = clean_gloss(gl, 90)
                if len(re.sub(r"[^A-Za-z]", "", g)) < 2:
                    continue
                recs.append(rec(key, head, [head] + [a if not ditto else last_head + " " + a for a in alts],
                                gloss_en=g, page="", ocr_page_index=pi, **({"ditto": "head repeated from previous entry"} if ditto else {})))
    return recs


# --------------------------------------------------------------------------
# Long 1909, Dictionary of the Chinook Jargon - Chinook-English half
# (OCR emits a block of headwords, then a block of glosses, in the same order)
# --------------------------------------------------------------------------

_LONG_POS = r"(?:n|v|adv|adj|ad|interj|inter\s*j|prep|conj|pron|art|num)"
_LONG_HEAD = re.compile(
    r"^(?P<h>[A-Za-z^\u00a3\u00ab\u00bb~'\u2019\-\. ,\u00e9]{1,36}?)"
    r"(?:\s*[,.]\s*(?P<pos>" + _LONG_POS + r"\.?(?:\s*,\s*" + _LONG_POS + r"\.?)*))?\s*[.,;]?$")
_LONG_OCR = str.maketrans({"^": "a", "\u00a3": "E", "~": "", "\u00ab": "", "\u00bb": ""})


_LONG_POS_TAIL = re.compile(r",\s*(?:n|v|w|ii|adv|adj|ad|cuf|conj|prep|pron|interj|inter\s*j|art|num|N|V)\b\W*$", re.I)
_LONG_FURNITURE = re.compile(r"^(?:[A-Z0-9]{1,3}|[^A-Za-z]{0,6}\d{0,3}[^A-Za-z]{0,6}|.*CHINOOK.{0,3}\w{0,3}LISH|.*[EB]N\w{0,2}LISH)$", re.I)


def _long_label(t: str) -> str:
    if _LONG_POS_TAIL.search(t):
        return "H"
    if re.search(r"[A-Za-z]-[A-Za-z]", t) and not re.search(r";", t) and len(t) < 30 and not re.search(r"\s[a-z]{3,}\s[a-z]{3,}", t):
        return "H"
    if re.search(r"[A-Za-z]'-?[A-Za-z]?", t) and not re.search(r";", t) and len(t) < 24 and re.match(r"^[A-Z\u00a3(]", t):
        return "H"
    return "G"


def parse_long1909(lines: List[str]) -> List[Rec]:
    key = "long1909"
    i0 = next(i for i, l in enumerate(lines) if re.fullmatch(r"\s*CHINOOK-ENGLISH\s*", l))
    i1 = next((i for i in range(i0, len(lines)) if "EXAMPLES IN CONVERSATION" in lines[i]), len(lines))
    seq: List[List] = []   # [label, text]
    for raw in lines[i0:i1]:
        t = squash(raw)
        if not t or (len(t) <= 4 and re.fullmatch(r"[^a-z]*", t)) or _LONG_FURNITURE.match(t) and len(t) < 22 and not _LONG_POS_TAIL.search(t) and re.search(r"CHINOOK|LISH|^\W*\d+\W*$|^[A-Z]$", t):
            continue
        seq.append([_long_label(t), t])
    for k in range(1, len(seq) - 1):          # smooth isolated labels
        if seq[k][0] != seq[k - 1][0] and seq[k - 1][0] == seq[k + 1][0]:
            seq[k][0] = seq[k - 1][0]
    runs: List[Tuple[str, List[str]]] = []
    for lab, t in seq:
        if runs and runs[-1][0] == lab:
            runs[-1][1].append(t)
        else:
            runs.append((lab, [t]))
    recs: List[Rec] = []
    skipped = 0
    total_heads = sum(len(r) for lab, r in runs if lab == "H")
    pending: List[str] = []
    for lab, run in runs:
        if lab == "H":
            pending.extend(run)
            continue
        if not pending:
            continue
        g = run
        if len(g) != len(pending):  # join wrapped gloss lines (lower-case continuation)
            g2: List[str] = []
            for t in g:
                if g2 and re.match(r"^[a-z]", t) and not g2[-1].endswith(";"):
                    g2[-1] += " " + t
                else:
                    g2.append(t)
            g = g2
        if len(g) != len(pending):
            if len(run) >= 3:
                skipped += len(pending)
                pending = []
            continue
        for ht, gt in zip(pending, g):
            m = _LONG_HEAD.match(ht)
            if not m:
                continue
            h = squash(m.group("h").translate(_LONG_OCR)).strip(" ,.-'")
            h = re.sub(r"\s*,\s*", " ", h)
            if len(re.sub(r"[^A-Za-z]", "", h)) < 2 or len(h.split()) > 4:
                continue
            gl = clean_gloss(gt.replace(" ;", ";"), 150)
            if not gl:
                continue
            recs.append(rec(key, h, [h], gloss_en=gl, pos=canon_pos(m.group("pos") or ""), page="",
                            align="head/gloss runs paired by line count"))
        pending = []
    PARSE_INFO["long1909"] = {"headword_lines_seen": total_heads, "headwords_aligned": len(recs),
                              "headword_lines_skipped_unaligned": skipped}
    return recs


# --------------------------------------------------------------------------
# Le Jeune 1898, Chinook and Shorthand Rudiments (printed vocabulary, 2 columns)
# --------------------------------------------------------------------------

_R98_ART = {"la", "le", "les", "l", "lo", "li"}


def parse_rudiments1898(ctx: "Ctx") -> List[Rec]:
    key = "rudiments1898"
    pages = ctx.xml_pages("cihm_15465")
    recs: List[Rec] = []
    for pi in range(7, 11):
        w, h, words = pages[pi]
        cols, sp = column_word_lines(words, w, 2)
        for ci, c in enumerate(cols):
            if pi == 9 and ci == 1:      # right column of this page is the English-first word list
                continue
            # left edge of the headword column = most common x0 of line-initial alphabetic words
            xs = sorted(l[0][1] for l in c if l and re.fullmatch(r"[A-Za-z][A-Za-z']{2,}.?", l[0][0]))
            if not xs:
                continue
            bins = Counter(round(x / 40) for x in xs)
            edge = bins.most_common(1)[0][0] * 40
            section = "chinook-vocabulary"
            for l in c:
                l = [t for t in l if t[3] >= edge - 30]            # drop margin noise left of the column
                t = squash(fix_quotes(" ".join(x[0] for x in l)))
                if re.match(r"(?i)^English words|^The above vocabulary", t):
                    break
                if re.match(r"(?i)^words from french", t):
                    section = "words-from-french"
                    continue
                if re.search(r"(?i)vocabulary|^chinook\W*$|^\W*\d{1,3}\W*$", t) and len(t) < 28:
                    continue
                toks = t.split()
                if len(toks) < 2:
                    continue
                clean = lambda x: x.strip("),.;:|!*\u2018'`-").replace("\u2019", "'")
                first = clean(toks[0])
                shown = toks[0].strip(",.;:|)")
                k = 1
                if first.lower() in _R98_ART and len(toks) > 2:
                    first = first + " " + clean(toks[1])
                    shown = first
                    k = 2
                if not re.fullmatch(r"[A-Za-z][A-Za-z' ]{1,24}", first):
                    continue
                gl: List[str] = []
                for tok in reversed(toks[k:]):
                    tt = tok.strip(",.;:!?")
                    if re.fullmatch(r"[A-Za-z][A-Za-z'\-]*", tt) and len(gl) < 4:
                        gl.insert(0, tt)
                    else:
                        break
                if not gl:
                    continue
                recs.append(rec(key, first, [shown], gloss_en=" ".join(gl), page=pi, section=section,
                                raw=t if len(t) < 70 else ""))
    return recs


CTX_PARSERS: Dict[str, Callable[["Ctx"], List[Rec]]] = {
    "practical1886": parse_practical1886,
    "rudiments1898": parse_rudiments1898,
    "demers1871": parse_demers1871,
}
PARSERS: Dict[str, Callable[[List[str]], List[Rec]]] = {
    "rudiments1924": parse_rudiments1924,
    "gibbs1863": parse_gibbs,
    "hibben1889": parse_hibben,
    "shaw1909": parse_shaw,
    "gill1909": parse_gill1909,
    "gill1887": parse_gill1887,
    "long1909": parse_long1909,
}


# --------------------------------------------------------------------------
# Post-processing: normalization, per-source merge, validation, confidence
# --------------------------------------------------------------------------

CLEAN_SOURCES = ["gibbs1863", "hibben1889", "shaw1909", "gill1909", "gill1887"]
NOISY_SOURCES = ["rudiments1924", "rudiments1898", "practical1886", "demers1871", "long1909"]
PARSED_SOURCES = ["rudiments1924", "rudiments1898", "practical1886", "gibbs1863", "hibben1889", "shaw1909", "gill1909",
                  "gill1887", "demers1871", "long1909"]
LE_JEUNE = {k for k, v in SOURCES.items() if v.get("kamloops")}
CONF_RANK = {"A": 0, "B": 1, "C": 2}

_STOP_GLOSS = {
    "a", "an", "the", "of", "to", "in", "on", "or", "and", "is", "it", "for", "with", "by", "at", "as", "be",
    "from", "that", "this", "one", "any", "some", "not", "also", "very", "used", "see", "same", "like", "make",
    "made", "being", "are", "was", "has", "have", "his", "her", "its", "their", "than", "up", "out",
}


def _stem(t: str) -> str:
    for suf in ("ing", "ed", "es", "s", "ly"):
        if len(t) > len(suf) + 2 and t.endswith(suf):
            return t[: -len(suf)]
    return t


def gloss_tokens(g: str) -> set:
    out = set()
    for t in re.findall(r"[a-z]+", g.lower()):
        if len(t) >= 3 and t not in _STOP_GLOSS:
            out.add(_stem(t))
    return out


_CONS_RUN = re.compile(r"[bcdfgjkmnpqrstvxz]{4,}")


def well_formed(head: str) -> bool:
    """Cheap OCR sanity test for a headword (a necessary, not sufficient, condition)."""
    head = re.sub(r"\s*-\s*", "-", head)          # OCR splits ``Chik -a-min`` / ``Coop '-coop``
    head = re.sub(r"\s+(?=['\u2019])", "", head)
    toks = head.split()
    if not (1 <= len(toks) <= 4):
        return False
    for t in toks:
        a = re.sub(r"[^a-z]", "", t.lower())
        if not (2 <= len(a) <= 18) or not re.fullmatch(r"[A-Za-z][A-Za-z'’‘\-\.]*", t):
            return False
        if not re.search(r"[aeiouy]", a) or _CONS_RUN.search(a):
            return False
        if a[0] == "x":
            return False        # initial X: OCR confusion for K (Shaw), no Chinook Jargon word starts with x
    return True


class RefIndex:
    """Headwords of the cleanly OCR'd dictionaries, used to grade the noisy sources."""

    def __init__(self, rows: List[dict]):
        self.exact: Dict[str, List[Tuple[str, set]]] = defaultdict(list)
        self.skel: Dict[str, set] = defaultdict(set)
        self.gt: Dict[str, set] = defaultdict(set)
        for r in rows:
            n = r["headword"]
            gt = gloss_tokens(r["gloss_en"])
            self.exact[n].append((r["source_key"], gt))
            self.gt[n] |= gt
            sk = skeleton(n)
            if " " not in n and len(sk) >= 2:   # single-token headwords only
                self.skel[sk].add(n)
        self.bk_norm = LX._BKTree([n for n in self.exact if " " not in n])
        self.bk_skel = LX._BKTree([k for k in self.skel if " " not in k and len(k) >= 3])
        self.vocab = set()
        for r in rows:
            self.vocab |= set(re.findall(r"[a-z]+", r["gloss_en"].lower()))

    def _agree(self, head_gt: set, norms: Iterable[str]) -> bool:
        return bool(head_gt) and any(head_gt & self.gt.get(n, set()) for n in norms)

    def match(self, norm: str, gloss: str) -> dict:
        """Return {level, nearest, agree, via} (level 0 exact, 1 skeleton, 2 edit<=1/2, None)."""
        hg = gloss_tokens(gloss)
        if norm in self.exact:
            return dict(level=0, nearest=norm, agree=self._agree(hg, [norm]), via="exact")
        sk = skeleton(norm)
        if " " in norm:  # multi-word: compositional check
            parts = norm.split()
            if all(p in self.exact or skeleton(p) in self.skel for p in parts):
                return dict(level=1, nearest=" ".join(parts), agree=False, via="all-words-attested")
            return dict(level=None, nearest="", agree=False, via="")
        if len(sk) >= 2 and sk in self.skel and (len(sk) >= 3 or self._agree(hg, sorted(self.skel[sk]))):
            cands = sorted(self.skel[sk])
            ag = self._agree(hg, cands)
            return dict(level=1, nearest=cands[0] if len(cands) == 1 or not ag else
                        next(c for c in cands if hg & self.gt.get(c, set())), agree=ag, via="skeleton")
        best = None
        if len(norm) >= 4:
            md = 2 if len(norm) >= 8 else 1
            for d, w in sorted(self.bk_norm.query(norm, md)):
                ag = self._agree(hg, [w])
                best = dict(level=2, nearest=w, agree=ag, via=f"edit{d}")
                if ag:
                    break
        if len(sk) >= 4:
            for d, w in sorted(self.bk_skel.query(sk, 1)):
                cands = sorted(self.skel[w])
                ag = self._agree(hg, cands)
                if best is None or (ag and not best["agree"]):
                    best = dict(level=2, nearest=cands[0], agree=ag, via=f"skeleton-edit{d}")
        return best or dict(level=None, nearest="", agree=False, via="")


def _display_head(h: str) -> str:
    h = squash(fix_quotes(h)).strip(" .,;:-'\"*")
    return h


def to_rows(raw: List[Rec], key: str) -> List[dict]:
    """Normalize raw records of one source and merge duplicates of the same normalized headword."""
    spec = SOURCES[key]
    by: "OrderedDict[str, dict]" = OrderedDict()
    for r in raw:
        head = _display_head(r["head"])
        norm = normalize_headword(head)
        if not norm or len(norm.replace(" ", "")) < 2 or len(norm) > 44 or len(norm.split()) > 4:
            continue
        variants: List[str] = []
        for v in [head] + [_display_head(x) for x in r["variants"]]:
            if v and v not in variants and normalize_headword(v):
                variants.append(v)
        row = by.get(norm)
        if row is None:
            row = dict(headword=norm, variants=variants, gloss_en=[], gloss_fr=[], pos=[], source_key=key,
                       source_page=r["page"], kamloops="yes" if spec.get("kamloops") else "no", notes=dict(r["notes"]),
                       n_raw=0)
            by[norm] = row
        else:
            for v in variants:
                if v not in row["variants"]:
                    row["variants"].append(v)
            for k, v in r["notes"].items():
                row["notes"].setdefault(k, v)
        row["n_raw"] += 1
        for fld, val in (("gloss_en", r["gloss_en"]), ("gloss_fr", r["gloss_fr"])):
            val = (val or "").strip()
            if val and val.lower() not in [x.lower() for x in row[fld]]:
                row[fld].append(val)
        for pp in filter(None, (r["pos"] or "").split(",")):
            if pp not in row["pos"]:
                row["pos"].append(pp)
        if not row["source_page"] and r["page"]:
            row["source_page"] = r["page"]
    rows = []
    for row in by.values():
        row["gloss_en"] = " | ".join(row["gloss_en"])
        row["gloss_fr"] = " | ".join(row["gloss_fr"])
        row["pos"] = ",".join(row["pos"])
        rows.append(row)
    return rows


def grade_noisy(key: str, rows: List[dict], ref: RefIndex) -> List[dict]:
    """Assign conf=A/B/C to rows of an OCR-noisy source by comparing with the clean dictionaries."""
    out = []
    needs_gloss = key == "long1909"
    for r in rows:
        n = r["headword"]
        notes = r["notes"]
        gl = r["gloss_en"]
        # --- gloss sanity: tokens must be English words known from the clean glosses
        toks = re.findall(r"[A-Za-z]+", gl)
        if toks:
            low = [t.lower() for t in toks]
            known = [t for t in low if t in ref.vocab or t in _STOP_GLOSS]
            content_known = [t for t in known if len(t) >= 3 and t not in _STOP_GLOSS]
            if key in ("rudiments1924", "rudiments1898"):
                if len(known) == len(toks):
                    pass
                elif (len(content_known) == 1 and len(toks) - len(known) <= 2
                      and all(len(t) <= 3 for t in low if t not in known)):
                    notes["ocr_gloss"] = gl
                    notes["gloss_cleaned"] = "unknown OCR tokens dropped"
                    gl = " ".join(known)
                else:
                    notes["ocr_gloss"] = gl
                    gl = ""
            elif len(known) < 0.5 * len(toks):
                notes["ocr_gloss"] = gl
                gl = ""
            elif len(known) < len(toks):
                notes["gloss_unverified"] = "yes"
        r["gloss_en"] = gl
        wf = well_formed(" ".join(r["variants"][:1]) or n)
        m = ref.match(n, gl) if wf else dict(level=None, nearest="", agree=False, via="")
        if gl == "" and notes.get("ocr_gloss"):
            m2 = ref.match(n, notes["ocr_gloss"]) if wf else m
            if m2["level"] is not None and (m["level"] is None or m2["agree"]):
                m = m2
        lvl, ag = m["level"], m["agree"]
        if needs_gloss:
            if lvl in (0, 1) and ag:
                conf = "A"
            elif lvl in (0, 1) and not ag:
                # headword attested by a clean dictionary, but head/gloss pairing is not confirmed
                notes["unverified_gloss"] = gl
                r["gloss_en"] = ""
                conf = "B"
            elif lvl == 2 and ag:
                conf = "B"
            else:
                notes["unverified_gloss"] = gl
                r["gloss_en"] = ""
                conf = "C"
        else:
            if lvl == 0 or (lvl == 1 and ag):
                conf = "A"
            elif lvl == 1 or (lvl == 2 and ag):
                conf = "B"
            else:
                conf = "C"
        if conf != "A" and not wf:
            conf = "C"
        notes["conf"] = conf
        if m["via"]:
            notes["validated"] = f"{m['via']}:{m['nearest']}" + ("+gloss" if ag else "")
        r["_conf"] = conf
        r["_wf"] = wf
        out.append(r)
    return out


def witness_upgrade(rows_by_src: Dict[str, List[dict]]) -> None:
    """Upgrade C rows that are independently read in another noisy source (skeleton + gloss agree) to B."""
    idx: Dict[str, List[Tuple[str, dict]]] = defaultdict(list)
    for key in NOISY_SOURCES:
        for r in rows_by_src.get(key, []):
            if r["_wf"]:
                idx[skeleton(r["headword"])].append((key, r))
    for key in NOISY_SOURCES:
        for r in rows_by_src.get(key, []):
            if r["_conf"] != "C" or not r["_wf"]:
                continue
            sk = skeleton(r["headword"])
            if len(sk) < 2:
                continue
            g1 = gloss_tokens(r["gloss_en"] or r["notes"].get("ocr_gloss", ""))
            for k2, r2 in idx.get(sk, []):
                if k2 == key:
                    continue
                g2 = gloss_tokens(r2["gloss_en"] or r2["notes"].get("ocr_gloss", ""))
                if g1 and g2 and g1 & g2 and (r2["_conf"] != "C" or k2 != key):
                    r["_conf"] = "B"
                    r["notes"]["conf"] = "B"
                    r["notes"]["validated"] = f"witness:{k2}:{r2['headword']}"
                    break


def grade_clean(key: str, rows: List[dict], ref: RefIndex) -> List[dict]:
    """Grade a cleanly OCR'd dictionary against the *other* clean dictionaries (leave-one-out).

    A  headword (exact or phonetic skeleton) also found in another dictionary: the OCR reading is cross-checked
    B  attested here only, but the headword is well formed
    C  not well formed (probable OCR damage)
    """
    for r in rows:
        n = r["headword"]
        wf = well_formed(r["variants"][0] if r["variants"] else n)
        m = ref.match(n, r["gloss_en"]) if wf else dict(level=None, nearest="", agree=False, via="")
        if m["level"] in (0, 1):
            conf = "A"
            r["notes"]["validated"] = f"{m['via']}:{m['nearest']}"
        elif wf:
            conf = "B"
            r["notes"]["single_source"] = "yes"
            if m["level"] == 2:
                r["notes"]["near"] = m["nearest"]
        else:
            conf = "C"
        r["notes"]["conf"] = conf
        r["_conf"] = conf
        r["_wf"] = wf
    return rows


def keep_noisy(key: str, r: dict) -> bool:
    n = r["notes"]
    if r["_conf"] in ("A", "B"):
        return True
    if not r["_wf"]:
        return False
    if key == "rudiments1924":
        # keep an unvalidated candidate only outside the free-text grammar region and only with a usable gloss
        return n.get("strict") != "yes" and bool(r["gloss_en"])
    if key == "long1909":
        return False
    return bool(r["gloss_en"])


# --------------------------------------------------------------------------
# Running the parsers
# --------------------------------------------------------------------------


def run_parser(key: str, ctx: Ctx) -> List[Rec]:
    spec = SOURCES[key]
    if key in CTX_PARSERS:
        return CTX_PARSERS[key](ctx)
    return PARSERS[key](ctx.lines(spec["ia_id"]))


def finalize_note(r: dict) -> str:
    notes = dict(r["notes"])
    order = ["conf", "validated", "near", "single_source", "type", "section", "ditto", "ocr", "unverified_gloss", "ocr_gloss", "gloss_cleaned",
             "gloss_unverified", "gloss_state"]
    ordered: Dict[str, object] = {}
    for k in order:
        if k in notes:
            ordered[k] = notes.pop(k)
    notes.pop("strict", None)
    notes.pop("line", None)
    notes.pop("ocr_page_index", None)
    ordered.update(notes)
    return note_str(ordered)


def build_all(src_dir: Path, download: bool) -> Tuple[List[dict], Dict[str, dict]]:
    ensure_sources(src_dir, download, list(SOURCES))          # also fetches sources that are documented but not parsed
    ctx = Ctx(src_dir)
    rows_by_src: Dict[str, List[dict]] = {}
    info: Dict[str, dict] = {}
    for key in PARSED_SOURCES:
        raw = run_parser(key, ctx)
        rows = to_rows(raw, key)
        info[key] = dict(n_raw=len(raw), n_rows_before=len(rows))
        rows_by_src[key] = rows
    # 1. cross-validate the clean dictionaries against each other (leave-one-out)
    for key in CLEAN_SOURCES:
        ref_k = RefIndex([r for k2 in CLEAN_SOURCES if k2 != key for r in rows_by_src[k2]])
        rows_by_src[key] = grade_clean(key, rows_by_src[key], ref_k)
    # 2. the reference for the noisy sources = A/B rows of the clean dictionaries
    ref = RefIndex([r for k in CLEAN_SOURCES for r in rows_by_src[k] if r["_conf"] in ("A", "B")])
    for key in NOISY_SOURCES:
        rows_by_src[key] = grade_noisy(key, rows_by_src[key], ref)
    witness_upgrade(rows_by_src)
    final: List[dict] = []
    for key in PARSED_SOURCES:
        kept = [r for r in rows_by_src[key] if key not in NOISY_SOURCES or keep_noisy(key, r)]
        info[key]["n_rows"] = len(kept)
        info[key]["conf"] = dict(Counter(r["_conf"] for r in kept))
        final.extend(kept)
    return final, info


# --------------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------------


def _tsv(v: str) -> str:
    """One TSV cell: single line, no tabs, no backslashes (the files are read with csv.QUOTE_NONE)."""
    return squash(str(v)).replace("\t", " ").replace("\\", "/")


def write_lexicon(rows: List[dict], path: Path) -> None:
    order = {k: i for i, k in enumerate(PARSED_SOURCES)}
    rows = sorted(rows, key=lambda r: (r["headword"], order[r["source_key"]]))
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("\t".join(LEXICON_COLS) + "\n")
        for r in rows:
            cells = [
                r["headword"], "|".join(_tsv(v).replace("|", "/") for v in r["variants"]), _tsv(r["gloss_en"]),
                _tsv(r["gloss_fr"]), r["pos"], r["source_key"], r["source_page"], r["kamloops"], finalize_note(r),
            ]
            fh.write("\t".join(_tsv(c) for c in cells) + "\n")


MERGED_COLS = ["headword", "n_sources", "sources", "kamloops", "best_conf", "variants", "gloss_en", "gloss_fr", "pos",
               "skeleton", "skeleton_siblings", "notes"]


def build_skel_index(headwords: Iterable[str]) -> Dict[str, set]:
    """skeleton -> single-token headwords, dropping uninformative skeletons (1 letter; 2 letters and > MAX_SKEL_CLUSTER members)."""
    idx: Dict[str, set] = defaultdict(set)
    for h in headwords:
        sk = skeleton(h)
        if " " not in h and len(sk) >= 2:
            idx[sk].add(h)
    return {k: v for k, v in idx.items() if len(k) >= 3 or len(v) <= LX.MAX_SKEL_CLUSTER}


def merge_rows(rows: List[dict]) -> List[dict]:
    order = {k: i for i, k in enumerate(PARSED_SOURCES)}
    groups: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        groups[r["headword"]].append(r)
    skel_groups = build_skel_index(h for h in groups if any(r["_conf"] in ("A", "B") for r in groups[h]))
    merged = []
    for h in sorted(groups):
        rs = sorted(groups[h], key=lambda r: order[r["source_key"]])
        good = [r for r in rs if r["_conf"] in ("A", "B")]
        best = min((r["_conf"] for r in rs), key=lambda c: CONF_RANK[c])
        srcs = []
        for r in good:
            if r["source_key"] not in srcs:
                srcs.append(r["source_key"])
        csrcs = [r["source_key"] for r in rs if r["_conf"] == "C" and r["source_key"] not in srcs]
        lj_good = any(r["source_key"] in LE_JEUNE for r in good)
        lj_c = any(r["source_key"] in LE_JEUNE for r in rs if r["_conf"] == "C")
        vs: List[str] = []
        for r in rs:
            for v in r["variants"]:
                if v not in vs:
                    vs.append(v)
        glosses: List[str] = []
        for r in good:
            for g in r["gloss_en"].split(" | "):
                if g and g.lower() not in [x.lower() for x in glosses]:
                    glosses.append(g)
        frs: List[str] = []
        for r in good:
            for g in r["gloss_fr"].split(" | "):
                if g and g not in frs:
                    frs.append(g)
        poss: List[str] = []
        for r in good:
            for pp in filter(None, r["pos"].split(",")):
                if pp not in poss:
                    poss.append(pp)
        sk = skeleton(h)
        sibs = sorted(skel_groups[sk] - {h}) if sk in skel_groups else []
        notes = []
        if csrcs:
            notes.append("unvalidated_candidates_in=" + "/".join(csrcs))
        if lj_c and not lj_good:
            notes.append("kamloops_candidate_only")
        merged.append(dict(
            headword=h, n_sources=len(srcs), sources="|".join(srcs), kamloops="yes" if lj_good else "no",
            best_conf=best, variants="|".join(_tsv(v).replace("|", "/") for v in vs[:14]),
            gloss_en=" | ".join(glosses[:6]), gloss_fr=" | ".join(frs[:3]), pos=",".join(poss),
            skeleton=sk, skeleton_siblings="|".join(sibs[:10]), notes="; ".join(notes),
            _rows=rs, _good=good,
        ))
    return merged


def write_merged(merged: List[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("\t".join(MERGED_COLS) + "\n")
        for m in merged:
            fh.write("\t".join(_tsv(m[c]) for c in MERGED_COLS) + "\n")


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def compute_stats(rows: List[dict], merged: List[dict]) -> dict:
    st: dict = {}
    st["rows"] = len(rows)
    st["rows_by_source"] = Counter(r["source_key"] for r in rows)
    st["rows_by_source_conf"] = {k: dict(Counter(r["_conf"] for r in rows if r["source_key"] == k)) for k in PARSED_SOURCES}
    st["merged_headwords"] = len(merged)
    ab = [m for m in merged if m["n_sources"] >= 1]
    st["merged_headwords_AB"] = len(ab)
    st["merged_headwords_A_only"] = sum(1 for m in merged if m["best_conf"] == "A")
    st["shared_exact"] = Counter(m["n_sources"] for m in ab)
    # exact overlap between clean dictionaries
    keys = PARSED_SOURCES
    sets = {k: {m["headword"] for m in ab if k in m["sources"].split("|")} for k in keys}
    st["distinct_AB_by_source"] = {k: len(sets[k]) for k in keys}
    st["pair_overlap_exact"] = {(a, b): len(sets[a] & sets[b]) for i, a in enumerate(keys) for b in keys[i + 1:]}
    # skeleton-level
    sidx = build_skel_index(m["headword"] for m in ab)

    def _sk(h):
        sk = skeleton(h)
        return sk if (" " not in h and sk in sidx) else None
    def _key(h):
        return _sk(h) or ("h:" + h)           # skeleton when informative, else the headword itself
    sk_sets = {k: {_key(h) for h in sets[k]} for k in keys}
    st["pair_overlap_skeleton"] = {(a, b): len(sk_sets[a] & sk_sets[b]) for i, a in enumerate(keys) for b in keys[i + 1:]}
    lj = {m["headword"] for m in ab if m["kamloops"] == "yes"}
    non_lj_heads = {m["headword"] for m in ab if any(s not in LE_JEUNE for s in m["sources"].split("|"))}
    non_lj_sk = {_sk(h) for h in non_lj_heads if _sk(h)}
    st["kamloops_headwords"] = len(lj)
    st["kamloops_only_exact"] = len([h for h in lj if h not in non_lj_heads])
    st["kamloops_only_skeleton"] = len([h for h in lj if h not in non_lj_heads and (_sk(h) is None or _sk(h) not in non_lj_sk)])
    st["kamloops_also_elsewhere_exact"] = len([h for h in lj if h in non_lj_heads])
    lj1 = [h for h in lj if " " not in h]
    st["kamloops_single_word"] = len(lj1)
    st["kamloops_single_only_exact"] = len([h for h in lj1 if h not in non_lj_heads])
    st["kamloops_single_only_skeleton"] = len([h for h in lj1 if h not in non_lj_heads and (_sk(h) is None or _sk(h) not in non_lj_sk)])
    ljc = {m["headword"] for m in merged if any(r["source_key"] in LE_JEUNE for r in m["_rows"])}
    st["kamloops_incl_C"] = len(ljc)
    # skeleton-level clusters over all A/B headwords
    cl: Dict[str, set] = defaultdict(set)
    for m in ab:
        if _sk(m["headword"]):
            cl[m["skeleton"]].add(m["headword"])
    st["skeleton_clusters"] = len(cl)
    st["skeleton_clusters_multi"] = sum(1 for v in cl.values() if len(v) > 1)
    srcs_by_head = {m["headword"]: set(m["sources"].split("|")) for m in ab}
    st["skeleton_clusters_by_nsources"] = Counter(
        len(set().union(*[srcs_by_head[h] for h in v])) for v in cl.values())
    return st


def print_stats(st: dict, info: Dict[str, dict]) -> None:
    print("rows (headword x source):", st["rows"])
    for k in PARSED_SOURCES:
        print(f"  {k:14s} raw={info[k]['n_raw']:5d} rows={st['rows_by_source'].get(k, 0):5d} conf={st['rows_by_source_conf'][k]}")
    print("merged headwords (all conf):", st["merged_headwords"], " with >=1 A/B source:", st["merged_headwords_AB"])
    print("headwords by number of A/B sources:", dict(sorted(st["shared_exact"].items())))
    print("distinct A/B headwords by source:", st["distinct_AB_by_source"])
    print("Kamloops (Le Jeune A/B) headwords:", st["kamloops_headwords"],
          "| only in Le Jeune (exact):", st["kamloops_only_exact"],
          "| only in Le Jeune (no skeleton sibling elsewhere):", st["kamloops_only_skeleton"])
    print("  single-word Le Jeune headwords:", st["kamloops_single_word"], "| only in Le Jeune (exact):",
          st["kamloops_single_only_exact"], "| no skeleton sibling elsewhere:", st["kamloops_single_only_skeleton"],
          "| incl. unvalidated C rows (all Le Jeune headwords):", st["kamloops_incl_C"])
    print("skeleton clusters:", st["skeleton_clusters"], "multi-spelling clusters:", st["skeleton_clusters_multi"])
    print("skeleton clusters by number of sources:", dict(sorted(st["skeleton_clusters_by_nsources"].items())))


# --------------------------------------------------------------------------
# sources.yaml (written by hand: no PyYAML dependency)
# --------------------------------------------------------------------------


def _yq(v) -> str:
    s = str(v)
    if re.fullmatch(r"-?\d+", s):
        return s
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


PARSE_QUALITY = {
    "rudiments1924": (
        "Text is lithographed handwriting; OCR garbles headwords and glosses (e.g. 'Chimook', 'masathi'). Only the "
        "glossary sections are read (163 original words, 22 more words, French words, religious words, words of other "
        "districts, the 1879 Durieu vocabulary, and word lists in the grammar part). Every candidate is graded A/B/C "
        "against the cleanly OCR'd dictionaries; C rows keep the as-read spelling and are excluded by default. Glosses "
        "are kept only if every token is a known English word, otherwise the raw OCR gloss is stored in notes "
        "(ocr_gloss=). Shorthand-only material (mamook compounds, lessons) is not extracted. Page numbers are not "
        "recoverable from the OCR; notes carry section=."),
    "practical1886": (
        "Printed two-column vocabulary read with a layout-aware column splitter on the djvu word coordinates. Entries "
        "fused on one OCR line are split at 'Capitalized head,' boundaries; ditto marks are expanded to 'Mamook'. "
        "Gloss OCR errors are NOT corrected (e.g. 'beet' for 'beef'); the numbers table is read English-first and glosses "
        "are assigned in fixed order. source_page is the printed page (xml index minus 2, INFERRED)."),
    "rudiments1898": (
        "Printed vocabulary (pages 7-10 of the 43-page booklet), read as two columns from the djvu word coordinates; "
        "margin noise left of the headword column is discarded. Each line is 'headword, shorthand-glyph noise, gloss': "
        "the gloss is the run of alphabetic words at the end of the line, then reduced to verified English words "
        "(raw reading kept in notes as ocr_gloss=). OCR errors in headwords remain (e.g. 'Athe', 'eskKom'); such rows "
        "grade B/C. The English-first word list on page 9 is skipped. Page numbers are the printed page (xml index, INFERRED)."),
    "vocab1892": "Not parsed: the OCR holds only English words and shorthand-glyph noise (Chinook given in phonography).",
    "gibbs1863": (
        "Clean OCR; Part I Chinook-English only. Etymology brackets stripped from glosses (kept as origin= in notes). "
        "Page numbers come from running heads and may be off by one near plates."),
    "hibben1889": "Clean OCR; abridged Gibbs, so it is not an independent witness for Gibbs's entries.",
    "shaw1909": (
        "Clean OCR; main lexicon plus supplemental vocabulary. Origin tags (C/E/F/N/S/J) kept as origin= in notes; "
        "a few lost opening brackets are repaired by a fallback."),
    "gill1909": "Clean OCR; Chinook-English part only. Trailing dots on headwords stripped.",
    "gill1887": "Clean OCR; Chinook-English part only; a little running-head leakage may remain in glosses.",
    "demers1871": (
        "Dictionary pages read as two columns. Ditto marks expand to the previous headword. Entry splitting relies on "
        "'Capitalized head,' boundaries so capitalised words inside glosses (Bible, Indian) can create spurious rows; "
        "these usually grade C and are dropped unless they carry a gloss. Graded A/B/C against the clean dictionaries."),
    "long1909": (
        "OCR emits a block of headwords followed by a block of glosses; pairs are formed only when both blocks have "
        "the same number of lines, so coverage is partial (see parse_info). A pair is trusted (A) only if the headword "
        "is attested in a clean dictionary AND the glosses overlap; otherwise the gloss is moved to notes "
        "(unverified_gloss=). The English-Chinook half is not used."),
}


def write_sources_yaml(path: Path, src_dir: Path, info: Dict[str, dict], rows: List[dict]) -> None:
    lines = [
        "# Generated by scripts/build_lexicon.py - do not edit by hand.",
        "# One entry per source dictionary.  entry_count = rows in lexicon.tsv (headword x source);",
        "# conf_counts = confidence grade (A validated/clean, B partial, C unvalidated OCR candidate).",
        "sources:",
    ]
    by = {k: [r for r in rows if r["source_key"] == k] for k in SOURCES}
    for key, spec in SOURCES.items():
        lines.append(f"  - key: {_yq(key)}")
        lines.append(f"    title: {_yq(spec['title'])}")
        lines.append(f"    author: {_yq(spec['author'])}")
        lines.append(f"    year: {spec['year']}")
        lines.append(f"    citation: {_yq(spec['citation'])}")
        lines.append(f"    archive_org_id: {_yq(spec['ia_id'])}")
        lines.append(f"    archive_org_url: {_yq('https://archive.org/details/' + spec['ia_id'])}")
        if spec.get("ia_alt"):
            lines.append("    archive_org_alternate_copies:")
            for a in spec["ia_alt"]:
                lines.append(f"      - {_yq(a)}")
        lines.append(f"    ocr_text_url: {_yq(IA_URL.format(id=spec['ia_id']))}")
        p = source_path(src_dir, spec["ia_id"])
        if p.exists():
            lines.append(f"    ocr_text_sha256: {_yq(sha256_of(p))}")
        lines.append(f"    kamloops_variety: {'true' if spec.get('kamloops') else 'false'}")
        lines.append(f"    kind: {_yq(spec['kind'])}")
        lines.append(f"    license: {_yq(spec['license'])}")
        lines.append(f"    status: {_yq(spec.get('status', ''))}")
        rr = by.get(key, [])
        lines.append(f"    entry_count: {len(rr)}")
        if rr:
            cc = Counter(r["_conf"] for r in rr)
            lines.append("    conf_counts: {" + ", ".join(f"{c}: {cc.get(c, 0)}" for c in 'ABC') + "}")
            lines.append(f"    raw_candidates: {info.get(key, {}).get('n_raw', 0)}")
        if key in PARSE_INFO:
            lines.append("    parse_info: {" + ", ".join(f"{k}: {v}" for k, v in PARSE_INFO[key].items()) + "}")
        lines.append(f"    parse_quality: {_yq(PARSE_QUALITY.get(key, ''))}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


EXPECTED_MIN_ROWS = {      # ~75% of the counts obtained on the archive.org OCR texts of 2026-10; catches a changed OCR text
    "rudiments1924": 240, "rudiments1898": 90, "practical1886": 240, "gibbs1863": 340, "hibben1889": 320,
    "shaw1909": 360, "gill1909": 430, "gill1887": 400, "demers1871": 270, "long1909": 160,
}
MUST_HAVE = ["mamook", "kloshe", "klootchman", "siwash", "wawa", "tilikum", "potlatch", "skookum"]


def sanity_check(rows: List[dict], merged: List[dict]) -> List[str]:
    """Return a list of human-readable problems (empty = fine)."""
    problems = []
    cnt = Counter(r["source_key"] for r in rows)
    for k, n in EXPECTED_MIN_ROWS.items():
        if cnt.get(k, 0) < n:
            problems.append(f"{k}: only {cnt.get(k, 0)} rows (expected >= {n}); did the OCR text change?")
    have = {m["headword"] for m in merged if m["n_sources"] >= 1}
    for w in MUST_HAVE:
        if w not in have:
            problems.append(f"headword {w!r} missing from the lexicon")
    return problems


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src-dir", type=Path, default=DEFAULT_SRC, help="cache directory for archive.org texts")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--no-download", action="store_true", help="never touch the network; fail if a text is missing")
    ap.add_argument("--stats", action="store_true", help="print overlap statistics")
    a = ap.parse_args(argv)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    rows, info = build_all(a.src_dir, download=not a.no_download)
    merged = merge_rows(rows)
    write_lexicon(rows, a.out_dir / "lexicon.tsv")
    write_merged(merged, a.out_dir / "lexicon_merged.tsv")
    write_sources_yaml(a.out_dir / "sources.yaml", a.src_dir, info, rows)
    st = compute_stats(rows, merged)
    print_stats(st, info)
    problems = sanity_check(rows, merged)
    for pr in problems:
        print("WARNING:", pr, file=sys.stderr)
    if a.stats:
        print("pair overlap (exact, A/B headwords):")
        for (x, y), n in sorted(st["pair_overlap_exact"].items(), key=lambda kv: -kv[1]):
            print(f"  {x:14s} {y:14s} {n:5d}  (skeleton {st['pair_overlap_skeleton'][(x, y)]})")
        print("PARSE_INFO:", PARSE_INFO)
    return 2 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
