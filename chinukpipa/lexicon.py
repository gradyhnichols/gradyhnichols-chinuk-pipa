"""Chinook Jargon lexicon: loader, lookup and fuzzy matching.

The lexicon itself lives in ``data/lexicon/`` and is produced by
``scripts/build_lexicon.py`` from public-domain dictionaries (see
``data/lexicon/sources.yaml``).

Two TSV files are read:

* ``lexicon.tsv``        one row per (headword, source)
* ``lexicon_merged.tsv`` one row per normalized headword, with the list of
                         sources that attest it

This module has **no third-party dependencies** and no relative imports, so it
can be imported as ``chinukpipa.lexicon`` or loaded directly by file path (the
build script does the latter, to share ``normalize_headword`` / ``skeleton``).

Typical use::

    from chinukpipa.lexicon import Lexicon
    lex = Lexicon.load()                       # merged lexicon, all sources
    lex.lookup("klootchman")                   # exact (after normalization)
    lex.fuzzy("tloos", max_dist=1)             # edit-distance neighbours (+ skeleton siblings)
    lex.fuzzy("kloochmin", kamloops=True)      # only words in Le Jeune
    lex.variants_of("tloos")                   # same phonetic skeleton

Normalization (``normalize_headword``) lower-cases, strips accents, drops
hyphens/apostrophes/periods (Gibbs and Shaw use them to mark syllables and
stress) and keeps single spaces between the words of a multi-word entry.
``skeleton`` is a coarser, *inferred* key that folds the main spelling
differences between the Gibbs/Shaw/Gill tradition and Le Jeune's own Latin
spelling (``kl``~``tl``, ``c``~``k``, ``ch``~``tch`` ...) so that
``klosh``/``tloos``/``kloshe`` share one key.  It is a heuristic and is
documented as such in ``data/lexicon/sources.yaml``.
"""

from __future__ import annotations

import csv
import heapq
import os
import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

__all__ = [
    "MAX_SKEL_CLUSTER",
    "normalize_headword",
    "skeleton",
    "levenshtein",
    "Entry",
    "MergedEntry",
    "Lexicon",
    "DEFAULT_DATA_DIR",
]

MAX_SKEL_CLUSTER = 8   # two-letter skeletons shared by more headwords than this are treated as uninformative

DEFAULT_DATA_DIR = Path(
    os.environ.get(
        "CHINUKPIPA_LEXICON_DIR",
        Path(__file__).resolve().parents[1] / "data" / "lexicon",
    )
)

# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------

_APOS = "'’‘ʼʻ`´′"
_STRIP_RE = re.compile(r"[^a-z ]+")
_WS_RE = re.compile(r"\s+")


def normalize_headword(s: str) -> str:
    """Normalize a spelling as found in a source to a lowercase Latin key.

    * Unicode NFKD, combining marks (accents, macrons, breves) removed;
    * lower-cased;
    * apostrophes, hyphens, periods, digits and any other non a-z characters
      removed (a hyphen *between* two words, e.g. ``Mam-ook``, is a syllable
      mark, so the pieces are glued: ``mamook``);
    * runs of whitespace collapsed to a single space, so a multi-word entry
      such as ``Chah-ko kloshe`` becomes ``chahko kloshe``.
    """
    if not s:
        return ""
    t = unicodedata.normalize("NFKD", s)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.lower()
    # ligatures / letters that NFKD leaves alone
    t = t.replace("ß", "ss").replace("æ", "ae").replace("œ", "oe")
    for a in _APOS:
        t = t.replace(a, "")
    t = t.replace("-", "").replace(".", "")
    t = _STRIP_RE.sub(" ", t)
    return _WS_RE.sub(" ", t).strip()


_SKEL_RULES: Tuple[Tuple[str, str], ...] = (
    ("gh", "h"),
    ("tch", "C"),
    ("thl", "L"),
    ("tlh", "L"),
    ("klh", "L"),
    ("tl", "L"),
    ("kl", "L"),
    ("dj", "C"),
    ("ch", "C"),
    ("sh", "s"),
    ("qu", "kw"),
    ("cw", "kw"),
    ("ck", "k"),
    ("ph", "f"),
    ("x", "ks"),
    ("j", "C"),
    ("c", "k"),
    ("ts", "z"),
    ("dz", "z"),
)
_VOWELS = set("aeiouy")


def skeleton(s: str) -> str:
    """Coarse phonetic key used to link spelling variants (INFERRED).

    Digraphs are folded (``tl``/``kl``/``thl`` -> ``L``; ``ch``/``tch``/``j``
    -> ``C``; ``sh`` -> ``s``; ``gh`` -> ``h``; ``c``/``ck`` -> ``k`` ...), then every vowel,
    ``h`` and ``w``-after-consonant is dropped and repeated letters are
    collapsed.  Two spellings with the same skeleton are *candidate* variants
    of one word - never proof.
    """
    t = normalize_headword(s).replace(" ", "")
    for a, b in _SKEL_RULES:
        t = t.replace(a, b)
    out: List[str] = []
    for ch in t:
        if ch in _VOWELS or ch == "h":
            continue
        if out and out[-1] == ch:
            continue
        out.append(ch)
    return "".join(out)


# --------------------------------------------------------------------------
# Edit distance
# --------------------------------------------------------------------------


def levenshtein(a: str, b: str, max_dist: Optional[int] = None) -> int:
    """Levenshtein distance; returns ``max_dist + 1`` early when exceeded."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la < lb:
        a, b, la, lb = b, a, lb, la
    if lb == 0:
        return la if max_dist is None else min(la, max_dist + 1)
    if max_dist is not None and la - lb > max_dist:
        return max_dist + 1
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ca = a[i - 1]
        row_min = cur[0]
        for j in range(1, lb + 1):
            cost = 0 if ca == b[j - 1] else 1
            v = prev[j - 1] + cost
            x = prev[j] + 1
            if x < v:
                v = x
            y = cur[j - 1] + 1
            if y < v:
                v = y
            cur[j] = v
            if v < row_min:
                row_min = v
        if max_dist is not None and row_min > max_dist:
            return max_dist + 1
        prev = cur
    return prev[lb]


class _BKTree:
    """Burkhard-Keller tree over normalized headwords."""

    def __init__(self, words: Iterable[str]):
        self.root: Optional[list] = None  # [word, {dist: child}]
        for w in words:
            self.add(w)

    def add(self, w: str) -> None:
        if self.root is None:
            self.root = [w, {}]
            return
        node = self.root
        while True:
            d = levenshtein(w, node[0])
            if d == 0:
                return
            child = node[1].get(d)
            if child is None:
                node[1][d] = [w, {}]
                return
            node = child

    def query(self, w: str, max_dist: int) -> List[Tuple[int, str]]:
        if self.root is None:
            return []
        out: List[Tuple[int, str]] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            d = levenshtein(w, node[0])      # exact distance: the pruning below needs it
            if d <= max_dist:
                out.append((d, node[0]))
            for dd, child in node[1].items():
                if d - max_dist <= dd <= d + max_dist:
                    stack.append(child)
        return out


# --------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------

_LEXICON_COLS = (
    "headword",
    "variants",
    "gloss_en",
    "gloss_fr",
    "pos",
    "source_key",
    "source_page",
    "kamloops",
    "notes",
)


@dataclass(frozen=True)
class Entry:
    """One row of ``lexicon.tsv``: a headword as given by one source."""

    headword: str
    variants: Tuple[str, ...]
    gloss_en: str
    gloss_fr: str
    pos: str
    source_key: str
    source_page: str
    kamloops: bool
    notes: str

    @property
    def confidence(self) -> str:
        """``A``/``B``/``C`` from the ``conf=`` note, ``A`` if absent."""
        m = re.search(r"(?:^|;\s*)conf=([ABC])", self.notes or "")
        return m.group(1) if m else "A"


@dataclass
class MergedEntry:
    """One row of ``lexicon_merged.tsv``: a normalized headword."""

    headword: str
    variants: Tuple[str, ...]
    glosses_en: Tuple[str, ...]
    glosses_fr: Tuple[str, ...]
    pos: Tuple[str, ...]
    sources: Tuple[str, ...]
    kamloops: bool
    best_conf: str = "A"
    notes: str = ""
    entries: List[Entry] = field(default_factory=list)

    @property
    def n_sources(self) -> int:
        return len(self.sources)


def _split(s: str, sep: str = "|") -> Tuple[str, ...]:
    return tuple(x for x in (s or "").split(sep) if x)


def _read_tsv(path: Path) -> Iterator[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in rd:
            yield {k: (v or "") for k, v in row.items() if k is not None}


# --------------------------------------------------------------------------
# Lexicon
# --------------------------------------------------------------------------


class Lexicon:
    """In-memory lexicon with exact, skeleton and edit-distance lookup."""

    def __init__(self, entries: Sequence[Entry]):
        self.entries: List[Entry] = list(entries)
        self._by_head: Dict[str, List[Entry]] = defaultdict(list)
        for e in self.entries:
            self._by_head[e.headword].append(e)
        self._merged: Dict[str, MergedEntry] = {}
        for h, es in self._by_head.items():
            self._merged[h] = self._merge(h, es)
        by_skel: Dict[str, List[str]] = defaultdict(list)
        for h in self._merged:
            sk = skeleton(h)
            if " " not in h and len(sk) >= 2:
                by_skel[sk].append(h)
        # two-letter skeletons shared by more than MAX_SKEL_CLUSTER headwords are too ambiguous to be useful
        self._by_skel: Dict[str, List[str]] = {
            k: v for k, v in by_skel.items() if len(k) >= 3 or len(v) <= MAX_SKEL_CLUSTER}
        self._bk: Optional[_BKTree] = None

    # -- construction ------------------------------------------------------
    @staticmethod
    def _merge(h: str, es: List[Entry]) -> MergedEntry:
        seen_v: List[str] = []
        for e in es:
            for v in e.variants:
                nv = normalize_headword(v)
                if nv and nv != h and nv not in seen_v:
                    seen_v.append(nv)

        def uniq(xs: Iterable[str]) -> Tuple[str, ...]:
            out: List[str] = []
            for x in xs:
                if x and x not in out:
                    out.append(x)
            return tuple(out)

        confs = sorted(e.confidence for e in es)
        return MergedEntry(
            headword=h,
            variants=tuple(seen_v),
            glosses_en=uniq(e.gloss_en for e in es),
            glosses_fr=uniq(e.gloss_fr for e in es),
            pos=uniq(e.pos for e in es),
            sources=uniq(e.source_key for e in es),
            kamloops=any(e.kamloops for e in es),
            best_conf=confs[0] if confs else "A",
            entries=list(es),
        )

    @classmethod
    def from_entries(cls, entries: Iterable[Entry]) -> "Lexicon":
        return cls(list(entries))

    @classmethod
    def load(
        cls,
        data_dir: Optional[os.PathLike] = None,
        min_conf: str = "B",
        sources: Optional[Iterable[str]] = None,
    ) -> "Lexicon":
        """Load ``lexicon.tsv`` from *data_dir* (default: repo ``data/lexicon``).

        ``min_conf``: drop rows whose parse confidence is worse than this
        (``A`` validated/clean, ``B`` partially validated, ``C`` unvalidated
        OCR candidate; default ``B`` so that unvalidated OCR candidates are
        not served unless asked for).  ``sources``: restrict to these
        ``source_key`` values.
        """
        d = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        path = d / "lexicon.tsv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found - run scripts/build_lexicon.py first"
            )
        want = set(sources) if sources else None
        order = {"A": 0, "B": 1, "C": 2}
        entries: List[Entry] = []
        for r in _read_tsv(path):
            e = Entry(
                headword=r["headword"],
                variants=_split(r["variants"]),
                gloss_en=r["gloss_en"],
                gloss_fr=r["gloss_fr"],
                pos=r["pos"],
                source_key=r["source_key"],
                source_page=r["source_page"],
                kamloops=r["kamloops"].strip().lower() == "yes",
                notes=r["notes"],
            )
            if want is not None and e.source_key not in want:
                continue
            if order[e.confidence] > order[min_conf]:
                continue
            entries.append(e)
        return cls(entries)

    # -- basic access ------------------------------------------------------
    def __len__(self) -> int:
        return len(self._merged)

    def __contains__(self, word: str) -> bool:
        return normalize_headword(word) in self._merged

    def __iter__(self) -> Iterator[MergedEntry]:
        return iter(self._merged.values())

    def headwords(
        self,
        kamloops: Optional[bool] = None,
        min_sources: int = 1,
        source: Optional[str] = None,
    ) -> List[str]:
        out = []
        for h, m in self._merged.items():
            if kamloops is not None and m.kamloops != kamloops:
                continue
            if m.n_sources < min_sources:
                continue
            if source is not None and source not in m.sources:
                continue
            out.append(h)
        return sorted(out)

    def sources(self) -> List[str]:
        return sorted({e.source_key for e in self.entries})

    # -- lookup ------------------------------------------------------------
    def lookup(self, word: str) -> Optional[MergedEntry]:
        """Exact lookup after normalization.  Also tries listed variants."""
        n = normalize_headword(word)
        m = self._merged.get(n)
        if m is not None:
            return m
        return self._variant_index().get(n)

    def _variant_index(self) -> Dict[str, MergedEntry]:
        idx = getattr(self, "_vidx", None)
        if idx is None:
            idx = {}
            for h, m in self._merged.items():
                for v in m.variants:
                    idx.setdefault(v, m)
            self._vidx = idx
        return idx

    def entries_for(self, word: str) -> List[Entry]:
        m = self.lookup(word)
        return list(m.entries) if m else []

    def variants_of(self, word: str) -> List[str]:
        """Headwords sharing the phonetic ``skeleton`` of *word* (INFERRED).

        Empty when the skeleton is a single letter, or has two letters and is
        shared by more than ``MAX_SKEL_CLUSTER`` headwords (too ambiguous)."""
        return sorted(self._by_skel.get(skeleton(word), []))

    # -- fuzzy -------------------------------------------------------------
    def _tree(self) -> _BKTree:
        if self._bk is None:
            self._bk = _BKTree(sorted(self._merged))
        return self._bk

    def fuzzy(
        self,
        word: str,
        max_dist: int = 2,
        limit: int = 5,
        kamloops: Optional[bool] = None,
        use_skeleton: bool = True,
    ) -> List[Tuple[str, int, MergedEntry]]:
        """Nearest headwords by edit distance on the normalized form.

        Returns ``[(headword, distance, merged_entry), ...]`` sorted by
        distance, then by number of attesting sources (more first), then
        alphabetically.  Headwords sharing the phonetic skeleton are included
        at distance 1 even when their raw edit distance is larger (set
        ``use_skeleton=False`` for pure Levenshtein).
        """
        n = normalize_headword(word)
        if not n:
            return []
        hits: Dict[str, int] = {h: d for d, h in self._tree().query(n, max_dist)}
        sk = skeleton(n)
        if use_skeleton and max_dist >= 1 and " " not in n and len(sk) >= 2:
            for h in self._by_skel.get(sk, []):
                hits[h] = min(hits.get(h, 99), 1)
        rows = []
        for h, d in hits.items():
            m = self._merged[h]
            if kamloops is not None and m.kamloops != kamloops:
                continue
            rows.append((h, d, m))
        rows.sort(key=lambda r: (r[1], -r[2].n_sources, r[0]))
        return rows[:limit]

    def best_match(
        self, word: str, max_dist: int = 2, kamloops: Optional[bool] = None
    ) -> Optional[Tuple[str, int, MergedEntry]]:
        r = self.fuzzy(word, max_dist=max_dist, limit=1, kamloops=kamloops)
        return r[0] if r else None

    # -- convenience for decoders / generators -----------------------------
    def word_list(self, **kw) -> List[str]:
        """Sorted single-token headwords (no spaces) - handy for LM vocab."""
        return [h for h in self.headwords(**kw) if " " not in h]

    def gloss(self, word: str, lang: str = "en") -> str:
        m = self.lookup(word)
        if not m:
            return ""
        g = m.glosses_en if lang == "en" else m.glosses_fr
        return g[0] if g else ""


def _cli(argv: Sequence[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Query the Chinook Jargon lexicon")
    ap.add_argument("word", nargs="+")
    ap.add_argument("--dir", default=None)
    ap.add_argument("--fuzzy", type=int, default=0, help="max edit distance")
    ap.add_argument("--kamloops", action="store_true")
    a = ap.parse_args(argv)
    lex = Lexicon.load(a.dir)
    for w in a.word:
        m = lex.lookup(w)
        if m:
            print(
                f"{w} -> {m.headword}  [{', '.join(m.sources)}]"
                f"{'  KAMLOOPS' if m.kamloops else ''}"
            )
            for g in m.glosses_en[:4]:
                print("    en:", g)
        else:
            print(f"{w}: not found")
        if a.fuzzy or not m:
            for h, d, mm in lex.fuzzy(
                w, max_dist=a.fuzzy or 2, kamloops=True if a.kamloops else None
            ):
                print(f"    ~{d} {h}  [{', '.join(mm.sources)}]  {(mm.glosses_en or [''])[0][:50]}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_cli(sys.argv[1:]))
