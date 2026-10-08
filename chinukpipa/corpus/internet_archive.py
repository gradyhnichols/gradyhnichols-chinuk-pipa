"""Internet Archive access for Le Jeune / Chinuk Pipa items.

The Internet Archive (IA) holds digitized CIHM microfilm of most of Le Jeune's
shorthand prints as ``cihm_<number>`` (contributed by Canadiana.org, scanned by
University of Alberta Libraries), several Newberry Library (Ayer collection)
items, and a few other scans.  IA
publishes a documented JSON API (``/advancedsearch.php``, ``/metadata/<id>``)
and a per-file download URL (``/download/<id>/<file>``), and lists an MD5 for
every file, so downloads here are verified against IA's own checksums.

Command line (run from the repository root)::

    python -m chinukpipa.corpus.internet_archive search 'creator:("Le Jeune, J. M. R.")'
    python -m chinukpipa.corpus.internet_archive download --ids cihm_15465 Ayer_PM846_L47_1891 \\
        --dest corpus/lejeune --kinds orig_jp2 pdf text meta

Files are fetched with :class:`chinukpipa.corpus.polite.PoliteSession` (>= 1 s
between requests, robots.txt honoured, retry with backoff).  Rights vary by
item; check the ``licenseurl`` / ``rights`` fields in each item's metadata
(saved next to the files as ``<id>/<id>_ia_metadata.json``).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import sys
import urllib.parse
from pathlib import Path
from typing import Iterable, Sequence

from .polite import HttpError, PoliteSession

log = logging.getLogger("chinukpipa.corpus.internet_archive")

API = "https://archive.org"

# --kinds -> predicate on an IA file record.  "orig_jp2" falls back to the
# processed jp2 zip when an item has no original JP2 tar.
KINDS = {
    "orig_jp2": lambda f: f["name"].endswith("_orig_jp2.tar"),
    "jp2": lambda f: f["name"].endswith("_jp2.zip"),
    "pdf": lambda f: f.get("format") in ("Text PDF", "Image Container PDF") and f["name"].endswith(".pdf"),
    "text": lambda f: f["name"].endswith("_djvu.txt"),
    "meta": lambda f: f["name"].endswith("_meta.xml") or f["name"].endswith("_marc.xml"),
    "scandata": lambda f: f["name"].endswith("_scandata.xml"),
}


def search(session: PoliteSession, query: str, *, rows: int = 200) -> list[dict]:
    """Run an ``advancedsearch`` query; returns the matching docs."""
    fields = ["identifier", "title", "creator", "year", "language", "collection", "imagecount", "publisher", "mediatype"]
    params = [("q", query)] + [("fl[]", f) for f in fields] + [("rows", str(rows)), ("output", "json")]
    url = f"{API}/advancedsearch.php?" + urllib.parse.urlencode(params)
    return session.get(url).json()["response"]["docs"]


def item_metadata(session: PoliteSession, identifier: str) -> dict:
    """Full ``/metadata/<identifier>`` record (item metadata plus the file list)."""
    return session.get(f"{API}/metadata/{urllib.parse.quote(identifier)}").json()


def select_files(meta: dict, kinds: Sequence[str]) -> list[dict]:
    """Choose the files of an item matching ``kinds`` (see :data:`KINDS`)."""
    files = [f for f in meta.get("files", []) if not f["name"].startswith("history/")]
    chosen: list[dict] = []
    for kind in kinds:
        hits = [f for f in files if KINDS[kind](f)]
        if not hits and kind == "orig_jp2":  # fall back to the processed zip
            hits = [f for f in files if KINDS["jp2"](f)]
        for f in hits:
            if f not in chosen:
                chosen.append(f)
    return chosen


def _md5(path: Path) -> str:
    h = hashlib.md5()  # noqa: S324 - IA publishes MD5s; used for integrity only
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download_file(session: PoliteSession, identifier: str, rec: dict, dest_dir: Path, ledger: Path) -> str:
    """Download one IA file into ``dest_dir`` and verify its MD5.

    Returns ``"skipped"`` (already present with the right MD5), ``"ok"``, or
    raises ``ValueError`` on a checksum/size mismatch.  A partial ``.part``
    file is resumed with an HTTP Range request.
    """
    name = rec["name"]
    target = dest_dir / name
    expected_md5 = rec.get("md5")
    if target.exists() and expected_md5 and _md5(target) == expected_md5:
        return "skipped"
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    url = f"{API}/download/{urllib.parse.quote(identifier)}/{urllib.parse.quote(name)}"
    offset = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else None
    resp = session.get(url, stream=True, accept_statuses=(200, 206), headers=headers)
    mode = "ab" if resp.status_code == 206 else "wb"
    try:
        with part.open(mode) as fh:
            for block in resp.iter_content(1 << 16):
                fh.write(block)
    finally:
        resp.close()
    if expected_md5 and _md5(part) != expected_md5:
        part.unlink(missing_ok=True)
        raise ValueError(f"{identifier}/{name}: MD5 mismatch (expected {expected_md5})")
    part.replace(target)
    entry = {
        "identifier": identifier,
        "file": name,
        "bytes": target.stat().st_size,
        "md5": expected_md5 or _md5(target),
        "sha256": _sha256(target),
        "url": url,
        "fetched_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    with ledger.open("a", encoding="utf8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")
    return "ok"


def download_item(session: PoliteSession, identifier: str, dest: Path, kinds: Sequence[str]) -> dict[str, int]:
    """Fetch metadata and the selected files of one IA item into ``dest/<identifier>/``."""
    item_dir = dest / identifier
    item_dir.mkdir(parents=True, exist_ok=True)
    meta_path = item_dir / f"{identifier}_ia_metadata.json"
    meta = item_metadata(session, identifier)
    if not meta.get("metadata"):
        raise ValueError(f"{identifier}: no such item")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), "utf8")
    counts = {"ok": 0, "skipped": 0, "error": 0}
    for rec in select_files(meta, kinds):
        try:
            counts[download_file(session, identifier, rec, item_dir, dest / "download_ledger.jsonl")] += 1
        except (HttpError, ValueError, OSError) as exc:
            counts["error"] += 1
            log.error("%s/%s: %s", identifier, rec["name"], exc)
    return counts


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m chinukpipa.corpus.internet_archive", description=__doc__.split("\n\n")[0])
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--min-interval", type=float, default=1.0)
    p.add_argument("--contact", help="contact address for the User-Agent (or $CHINUKPIPA_CONTACT)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("search", help="print identifier | year | title for an advancedsearch query")
    sp.add_argument("query")
    sp = sub.add_parser("download", help="download selected files of the given items")
    sp.add_argument("--ids", nargs="+", required=True)
    sp.add_argument("--dest", default="corpus/lejeune")
    sp.add_argument("--kinds", nargs="+", default=["orig_jp2", "pdf", "text", "meta"], choices=sorted(KINDS))
    return p


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)  # fmt: skip
    with PoliteSession(min_interval=args.min_interval, contact=args.contact) as s:
        if args.cmd == "search":
            for d in search(s, args.query):
                print(f"{d.get('identifier')} | {d.get('year')} | {d.get('title')}")
            return 0
        dest = Path(args.dest)
        dest.mkdir(parents=True, exist_ok=True)
        failures = 0
        for i, ident in enumerate(args.ids, 1):
            try:
                counts = download_item(s, ident, dest, args.kinds)
                log.info("[%d/%d] %s %s", i, len(args.ids), ident, counts)
                failures += counts["error"]
            except (HttpError, ValueError) as exc:
                failures += 1
                log.error("[%d/%d] %s: %s", i, len(args.ids), ident, exc)
        return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
