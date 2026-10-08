"""Word list with Le Jeune's own spellings added, for the recognizer's word-list reading.

    python scripts/build_lj_wordlist.py --sources LJ1892 LJ1898 --out lex_lj_all.tsv
        [--lexicon data/lexicon/lexicon_merged.tsv] [--gt FILE.jsonl [FILE.jsonl ...]]

Writes a TSV with the columns headword, best_conf, tokens, source:

* the A/B headwords of the merged word list (`--lexicon`), with `tokens` left empty (the reader computes them with
  the spelling rules) and source `lexicon`;
* then, for every Chinook row (language `chn`, with a `tokens_rule`) of the chosen ground-truth sources
  (`--sources`, as named in the `source` field of the `--gt` files), one row per distinct token string: headword =
  the Roman spelling as printed, best_conf `A`, tokens = `tokens_rule`, source = the source id.

Use the result as `--lexicon` of `python -m chinukpipa.text.readcrops` and `remerge`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def gt_rows(paths):
    """(source, latin, tokens_rule) of every Chinook row with a rule reading in the JSONL files."""
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("language") == "chn" and r.get("tokens_rule"):
                    yield r["source"], r["latin"], r["tokens_rule"]


def build(lexicon: str, gt: list[str], sources: set[str], out: str) -> tuple[int, int]:
    """Write the word list to `out`. Returns (headwords from the lexicon, token strings added from the ground
    truth)."""
    base = []
    with open(lexicon, encoding="utf-8") as f:
        h = f.readline().rstrip("\n").split("\t")
        hi, ci = h.index("headword"), h.index("best_conf")
        for line in f:
            c = line.rstrip("\n").split("\t")
            if c[ci] in ("A", "B"):
                base.append((c[hi], c[ci], "", "lexicon"))
    seen, extra, found = set(), [], set()
    for src, latin, toks in gt_rows(gt):
        found.add(src)
        if src in sources and toks not in seen:
            seen.add(toks)
            extra.append((latin.replace("\t", " "), "A", toks, src))
    missing = sources - found
    if missing:
        sys.exit(f"no Chinook rows with a rule reading for source(s) {', '.join(sorted(missing))} in the --gt files")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("headword\tbest_conf\ttokens\tsource\n")
        for r in base + extra:
            fh.write("\t".join(r) + "\n")
    return len(base), len(extra)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lexicon", default=str(REPO / "data" / "lexicon" / "lexicon_merged.tsv"),
                    help="merged word list TSV (default: %(default)s)")
    ap.add_argument("--gt", nargs="+", metavar="JSONL",
                    default=[str(REPO / "data" / "gt" / "vocab_annotations.jsonl"),
                             str(REPO / "data" / "gt" / "copy_annotations.jsonl")],
                    help="ground-truth annotation files (default: the two files in data/gt/)")
    ap.add_argument("--sources", nargs="+", required=True, metavar="ID",
                    help="ground-truth sources whose words are added, e.g. LJ1892 LJ1898")
    ap.add_argument("--out", required=True, help="output TSV")
    a = ap.parse_args()
    n_base, n_extra = build(a.lexicon, a.gt, set(a.sources), a.out)
    print(a.out, "base", n_base, "LJ token strings added", n_extra)


if __name__ == "__main__":
    main()
