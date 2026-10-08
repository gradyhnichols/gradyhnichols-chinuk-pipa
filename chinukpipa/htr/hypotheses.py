"""Test alternative spelling-to-sign rules against the word images, using recognizers that never saw the words.

    python -m chinukpipa.htr.hypotheses --fold v0 runs/a_v0/model.pt runs/b_v0/model.pt --fold v1 ... \\
        --crops gt_crops [--real-scales LJ1892=0.5] [--out hyp.json]

For every held-out word of each fold, each hypothesis below proposes an alternative token sequence (where it
applies). Both the rule sequence and the alternative are scored by CTC loss under that fold's models (averaged
over models and test-time rescalings); the lower loss "wins". The models were trained on labels made with the
current rules, which biases them toward those rules, so an alternative that still wins across many different
words is a lead worth checking by eye. It is not proof. A deletion can win simply because a model is unsure of a
small sign, so the "control" rows (deleting one sign that the rules have no reason to doubt) give a base rate
for single deletions; hypotheses that delete several signs are not fully matched by them.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

from chinukpipa.htr.contexts import VOWELS, token_sources
from chinukpipa.htr.data import RealDataset, encode, parse_scales, to_tensor, vocab
from chinukpipa.htr.model import load_model
from chinukpipa.htr.normalize import rescale


def _drop_vowel_adjacent_i(src):
    return [s for s in src if not (s[0] == "E" and s[1] in ("i", "y") and (s[2] in VOWELS or s[3] in VOWELS))]


def _i_before_vowel_only(src):
    return [s for s in src if not (s[0] == "E" and s[1] in ("i", "y") and s[3] in VOWELS)]


def _i_after_vowel_only(src):
    return [s for s in src if not (s[0] == "E" and s[1] in ("i", "y") and s[2] in VOWELS and s[3] not in VOWELS)]


def _tsh(repl):
    def f(src):
        out, i = [], 0
        while i < len(src):
            if src[i][0] == "TS" and i + 1 < len(src) and src[i + 1][0] == "H" and src[i + 1][1] == "h":
                out += [(t, "", "", "") for t in repl]
                i += 2
            else:
                out.append(src[i])
                i += 1
        return out
    return f


def _final_s_as_ts(src):
    """The old z rule: an S that came from a written z -> TS."""
    return [("TS", "z", s[2], s[3]) if (s[0] == "S" and s[1] == "z") else s for s in src]


def _u_for_iu(src):
    out, i = [], 0
    while i < len(src):
        if src[i][0] == "E" and src[i][1] == "i" and i + 1 < len(src) and src[i + 1][0] == "U":
            out.append(src[i + 1])
            i += 2
        else:
            out.append(src[i])
            i += 1
    return out


def _old_rule(new_token, latins, old_tokens):
    """Undo a revised rule (2026-10-08): a `new_token` written with one of `latins` -> the old token sequence."""
    def f(src):
        out = []
        for s in src:
            if s[0] == new_token and s[1].lower() in latins:
                out += [(t, "", "", "") for t in old_tokens]
            else:
                out.append(s)
        return out
    return f


def _drop_first(token, latin=None, consonant_context=False):
    """Control: drop the first `token` (optionally only one written with `latin` with no vowel letter on either
    side, which includes word-initial and word-final positions)."""
    def f(src):
        for k, t in enumerate(src):
            if t[0] == token and (latin is None or t[1] == latin) and \
                    (not consonant_context or (t[2] not in VOWELS and t[3] not in VOWELS)):
                return src[:k] + src[k + 1:]
        return src
    return f


HYPOTHESES = {
    "i_next_to_vowel_not_written": _drop_vowel_adjacent_i,
    "i_before_vowel_not_written": _i_before_vowel_only,
    "i_after_vowel_not_written": _i_after_vowel_only,
    "iu_is_U_alone": _u_for_iu,
    "tsh_is_CH": _tsh(["CH"]),
    "tsh_is_TS_without_H": _tsh(["TS"]),
    "tsh_is_SH": _tsh(["SH"]),
    "z_is_TS (old rule)": _final_s_as_ts,
    # under the revised rules (tsh/tch -> CH, iu/yu/yoo -> U) the old readings become the alternatives
    "tsh_is_TS_H (old rule)": _old_rule("CH", {"tsh"}, ["TS", "H"]),
    "tch_is_T_CH (old rule)": _old_rule("CH", {"tch"}, ["T", "CH"]),
    "iu_yu_is_E_U (old rule)": _old_rule("U", {"iu", "yu"}, ["E", "U"]),
    "yoo_is_E_OO (old rule)": _old_rule("U", {"yoo"}, ["E", "OO"]),
    # controls: deleting a sign that is surely written should rarely win; they measure the bias toward shorter
    # sequences that a deletion hypothesis enjoys
    "control: drop an i with no vowel beside it": _drop_first("E", "i", consonant_context=True),
    "control: drop the first A": _drop_first("A"),
    "control: drop the first O": _drop_first("O"),
}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", nargs="+", action="append", required=True, metavar=("SEED", "MODEL"),
                    help="a split seed followed by the model files trained without its test words")
    ap.add_argument("--crops", required=True)
    ap.add_argument("--real", nargs="+", default=["data/gt/vocab_annotations.jsonl"])
    ap.add_argument("--real-scales")
    ap.add_argument("--tta", default="0.85,1,1.2")
    ap.add_argument("--out")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    v = vocab()
    tta = [float(x) for x in a.tta.split(",")]
    ctc = torch.nn.CTCLoss(blank=0, reduction="none", zero_infinity=False)
    results = defaultdict(list)
    for fold in a.fold:
        seed, paths = fold[0], fold[1:]
        if not paths:
            ap.error(f"--fold {seed}: give at least one model file")
        models = []
        for p in paths:
            m = load_model(p, len(v), device)
            models.append(m)
        ds = RealDataset(a.real, a.crops, v, split="test", scales=parse_scales(a.real_scales), split_seed=seed)
        for im, toks, rid, latin, source in ds.items:
            src = token_sources(latin)
            if [s[0] for s in src] != toks:
                continue
            alts = {}
            for name, f in HYPOTHESES.items():
                alt = [s[0] for s in f(src)]
                if alt != toks and alt and all(t in v for t in alt):
                    alts[name] = alt
            if not alts:
                continue
            arr = np.asarray(im.convert("L"))
            xs = [to_tensor(im) if s == 1 else to_tensor(Image.fromarray(rescale(arr, s))) for s in tta]
            seqs = [toks] + list(alts.values())
            targets = [torch.tensor(encode(s, v)) for s in seqs]
            tot = torch.zeros(len(seqs))
            for m in models:
                for x in xs:
                    lp = m(x[None].to(device))
                    T = lp.shape[0]
                    n = len(seqs)
                    loss = ctc(lp.expand(T, n, lp.shape[-1]), torch.cat(targets).to(device),
                               torch.full((n,), T, dtype=torch.long), torch.tensor([len(t) for t in targets]))
                    tot += loss.cpu()
            tot /= len(models) * len(xs)
            for k, name in enumerate(alts, start=1):
                results[name].append({"fold": seed, "id": rid, "latin": latin, "source": source,
                                      "rule": " ".join(toks), "alt": " ".join(alts[name]),
                                      "nll_rule": float(tot[0]), "nll_alt": float(tot[k])})
    summary = {}
    for name, rows in results.items():
        fin = [r for r in rows if np.isfinite(r["nll_rule"]) or np.isfinite(r["nll_alt"])]
        wins = sum(r["nll_alt"] < r["nll_rule"] for r in fin)
        diffs = [r["nll_rule"] - r["nll_alt"] for r in fin if np.isfinite(r["nll_rule"]) and np.isfinite(r["nll_alt"])]
        summary[name] = {"occurrences": len(fin), "distinct_words": len({r["latin"].lower() for r in fin}),
                         "alt_wins": wins, "median_nll_gain_for_alt": float(np.median(diffs)) if diffs else None}
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    if a.out:
        json.dump({"summary": summary, "rows": results}, open(a.out, "w", encoding="utf-8"), indent=1,
                  ensure_ascii=False)


if __name__ == "__main__":
    sys.exit(main())
