#!/usr/bin/env python3
"""Validate data/signs/signs.yaml and brief_forms.yaml, and optionally check Unicode coverage.

Usage:
  python3 -I validate_signs.py [--signs signs.yaml] [--brief brief_forms.yaml]
                               [--nameslist NamesList-excerpt.txt] [--keyboard dscorbett_test/index.html]

Structural checks (always run): unique ids; required fields; category and confidence vocabularies;
code-point format; confusables / same_shape_as resolve to existing ids; every sources entry starts with a
key declared in source_keys (parenthesised notes are skipped); labels use the legend vocabulary; brief-form items have
required fields and a valid label.

Coverage checks (only with --nameslist / --keyboard; the inputs are not shipped in this repo):
  --nameslist  a Unicode NamesList excerpt for the Duployan block (lines '1BC02<TAB>NAME', then
               '<TAB>x ...' / '<TAB>* ...' annotation lines). Every code point annotated 'Chinook' or
               'Salishan' must appear in the inventory (as unicode or unicode_variants).
  --keyboard   D. S. Corbett's Chinuk Pipa keyboard page (index.html). Every Duployan code point on it must appear
               in the inventory except those listed in KEYBOARD_EXEMPT.
Exit status 0 if all checks pass, 1 otherwise.
"""
import argparse
import os
import re
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
CATEGORIES = {"consonant", "vowel", "diphthong", "diacritic", "punctuation", "numeral", "logogram", "abbreviation"}
CONFIDENCE = {"high", "med", "low"}
STATUS = ("VERIFIED", "COMPUTED", "INFERRED", "UNVERIFIED")
SIGN_REQUIRED = ["id", "category", "unicode", "unicode_name", "chinook_value", "lejeune_latin", "stroke",
                 "joining", "confusables", "sources", "confidence", "notes"]
BRIEF_REQUIRED = ["id", "kind", "chinook", "expansion", "construction", "unicode", "label", "sources", "confidence", "notes"]
CP_RE = re.compile(r"U\+[0-9A-F]{4,6}$")
# Keyboard code points deliberately not in the inventory (both stated unused for Chinook by the keyboard help).
KEYBOARD_EXEMPT = {0x1BC9D: "DUPLOYAN THICK LETTER SELECTOR (shade; 'not used in Chinook')",
                   0x1BCA0: "SHORTHAND FORMAT LETTER OVERLAP ('not needed for Chinook')"}


def flat(x):
    if x is None:
        return []
    if isinstance(x, list):
        return [y for e in x for y in flat(e)]
    return [x]


def cps_of(x):
    return [int(t, 16) for s in flat(x) if isinstance(s, str) for t in re.findall(r"U\+([0-9A-F]{4,6})", s)]


def check_structure(signs_doc, brief_doc, err):
    keys = set(signs_doc.get("source_keys", {}))
    # short forms used in sources lists (prefix before first space)
    signs = signs_doc["signs"]
    ids = [s["id"] for s in signs]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        err("duplicate sign ids: %s" % sorted(dup))
    idset = set(ids)
    for s in signs:
        i = s.get("id", "?")
        for f in SIGN_REQUIRED:
            if f not in s:
                err("%s: missing field %s" % (i, f))
        if not re.fullmatch(r"[A-Za-z0-9_]+", str(i)):
            err("%s: id is not ascii [A-Za-z0-9_]" % i)
        if s.get("category") not in CATEGORIES:
            err("%s: bad category %r" % (i, s.get("category")))
        if s.get("confidence") not in CONFIDENCE:
            err("%s: bad confidence %r" % (i, s.get("confidence")))
        for cp in flat(s.get("unicode")) + flat(s.get("unicode_variants")):
            toks = re.findall(r"U\+[0-9A-Fa-f]+", str(cp))
            if not toks:
                err("%s: unicode entry %r has no U+ token" % (i, cp))
            for t in toks:
                if not CP_RE.fullmatch(t):
                    err("%s: malformed code point %r" % (i, t))
        for c in s.get("confusables") or []:
            if c not in idset:
                err("%s: confusable %r is not a sign id" % (i, c))
            if c == i:
                err("%s: lists itself as confusable" % i)
        if s.get("same_shape_as") is not None and s["same_shape_as"] not in idset:
            err("%s: same_shape_as %r is not a sign id" % (i, s["same_shape_as"]))
        if not s.get("sources"):
            err("%s: empty sources" % i)
        else:
            for src in s["sources"]:
                if str(src).startswith("("):
                    continue  # a parenthesised note such as '(page not located)' is not a citation
                head = re.split(r"[ ,;:(]", str(src), maxsplit=1)[0]
                if keys and head not in keys:
                    err("%s: source %r does not start with a declared source key (%s)" % (i, src, head))
        for fld, st in (s.get("labels") or {}).items():
            if not any(w in str(st) for w in STATUS):
                err("%s: label %s=%r has no status word" % (i, fld, st))
    items = brief_doc["items"]
    bids = [b["id"] for b in items]
    bdup = {i for i in bids if bids.count(i) > 1}
    if bdup:
        err("duplicate brief-form ids: %s" % sorted(bdup))
    for b in items:
        i = b.get("id", "?")
        for f in BRIEF_REQUIRED:
            if f not in b:
                err("brief %s: missing field %s" % (i, f))
        if b.get("confidence") not in CONFIDENCE:
            err("brief %s: bad confidence %r" % (i, b.get("confidence")))
        if not any(w in str(b.get("label")) for w in STATUS):
            err("brief %s: label %r has no status word" % (i, b.get("label")))
        for cp in cps_of(b.get("unicode")):
            if not (0 < cp <= 0x10FFFF):
                err("brief %s: code point out of range" % i)
    return signs, items


def parse_nameslist(path):
    cur, d = None, {}
    for ln in open(path, encoding="utf-8"):
        ln = ln.rstrip("\n")
        m = re.match(r"^([0-9A-F]{4,6})\t(.*)$", ln)
        if m:
            cur = int(m.group(1), 16)
            d[cur] = {"name": m.group(2), "notes": []}
        elif ln.startswith("\t") and cur is not None:
            d[cur]["notes"].append(ln.strip())
    return d


def keyboard_cps(path):
    html = open(path, encoding="utf-8").read()
    kb = set()
    for m in re.finditer(r'<span class="keycap[^"]*"[^>]*>([^<]*)</span>', html):
        kb.update(ord(ch) for ch in m.group(1) if 0x1BC00 <= ord(ch) <= 0x1BCA3)
    for m in re.finditer(r'data-string="((?:&#x[0-9A-Fa-f]+;)+)"', html):
        kb.update(int(t, 16) for t in re.findall(r"&#x([0-9A-Fa-f]+);", m.group(1)) if 0x1BC00 <= int(t, 16) <= 0x1BCA3)
    return kb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--signs", default=os.path.join(HERE, "signs.yaml"))
    ap.add_argument("--brief", default=os.path.join(HERE, "brief_forms.yaml"))
    ap.add_argument("--nameslist")
    ap.add_argument("--keyboard")
    a = ap.parse_args()
    errors = []
    err = errors.append
    sd = yaml.safe_load(open(a.signs, encoding="utf-8"))
    bd = yaml.safe_load(open(a.brief, encoding="utf-8"))
    signs, items = check_structure(sd, bd, err)
    from collections import Counter
    cat = Counter(s["category"] for s in signs)
    have = set()
    for s in signs:
        have.update(cps_of(s.get("unicode")))
        have.update(cps_of(s.get("unicode_variants")))
    print("signs: %d (%s); with a code point: %d; brief-form items: %d"
          % (len(signs), ", ".join("%s %d" % kv for kv in sorted(cat.items(), key=lambda kv: -kv[1])),
             sum(1 for s in signs if s.get("unicode")), len(items)))
    if a.nameslist:
        nl = parse_nameslist(a.nameslist)
        chinook = sorted(cp for cp, v in nl.items() if any("Chinook" in n for n in v["notes"]))
        salish = sorted(cp for cp, v in nl.items() if any("Salishan" in n for n in v["notes"]))
        mc = [cp for cp in chinook if cp not in have]
        ms = [cp for cp in salish if cp not in have]
        print("NamesList 'Chinook' code points: %d, missing: %s" % (len(chinook), [hex(c) for c in mc]))
        print("NamesList 'Salishan' code points: %d, missing: %s" % (len(salish), [hex(c) for c in ms]))
        if mc or ms:
            err("NamesList code points missing from inventory")
    if a.keyboard:
        kb = keyboard_cps(a.keyboard)
        miss = sorted(cp for cp in kb if cp not in have and cp not in KEYBOARD_EXEMPT)
        exempt = sorted(cp for cp in kb if cp in KEYBOARD_EXEMPT and cp not in have)
        print("keyboard Duployan code points: %d, missing: %s, exempt: %s"
              % (len(kb), [hex(c) for c in miss], [hex(c) for c in exempt]))
        if miss:
            err("keyboard code points missing from inventory")
    for e in errors:
        print("ERROR:", e)
    print("FAIL (%d errors)" % len(errors) if errors else "OK")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
