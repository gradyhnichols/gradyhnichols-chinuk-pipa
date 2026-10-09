"""Checks of two-word decoding and brief forms (v0.4, experimental): chinukpipa.text.twoword, and the options
--pair-penalty and --extra-lexicon of readcrops and remerge. CPU only; no data files beyond the repository's,
no models, no network.

The tests of the proposal step need no PyTorch. The tests of the ranking and of the command line need it, and are
skipped where it is not installed; they use small random models, or log-probabilities made by hand.
"""
import csv
import itertools
import json
import math
import os
import random
import sys

import numpy as np
import pytest

from chinukpipa.text import twoword
from chinukpipa.text.twoword import (DeletionIndex, deletion_variants, edit_distance, max_edit, merge_ranked,
                                     propose_pairs)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------------------- edit distance


def test_edit_distance_counts_insertions_deletions_and_substitutions():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance(["K", "A", "T"], ["K", "A", "T"]) == 0
    assert edit_distance(["K", "A", "T"], ["K", "T"]) == 1                    # one deletion
    assert edit_distance(["K", "T"], ["K", "A", "T"]) == 1                    # one insertion
    assert edit_distance(["K", "A", "T"], ["K", "O", "T"]) == 1               # one substitution
    assert edit_distance([], ["K", "A"]) == 2 and edit_distance(["K", "A"], []) == 2 and edit_distance([], []) == 0
    assert edit_distance("ab", "ba") == 2                                      # a swap is two edits
    assert edit_distance(("K", "A"), ["K", "A"]) == 0                          # lists and tuples alike


def test_edit_distance_is_symmetric_and_obeys_the_triangle_inequality():
    rng = random.Random(1)
    seqs = ["".join(rng.choice("abc") for _ in range(rng.randint(0, 6))) for _ in range(40)]
    for a, b in itertools.combinations(seqs, 2):
        assert edit_distance(a, b) == edit_distance(b, a)
    for a, b, c in itertools.islice(itertools.permutations(seqs[:12], 3), 400):
        assert edit_distance(a, c) <= edit_distance(a, b) + edit_distance(b, c)


def test_edit_distance_is_the_same_function_gtrows_uses():
    from chinukpipa.text import gtrows
    assert gtrows.edit_distance is edit_distance


def test_max_edit_by_length_of_the_part():
    assert [max_edit(n) for n in range(1, 9)] == [0, 0, 1, 1, 2, 2, 2, 2]
    assert twoword.MAX_PROPOSALS == 400 and twoword.MAX_DELETIONS == 2 and twoword.WORD_SPACE == "_"


# ---------------------------------------------------------------------------------------- deletion index


def test_deletion_variants():
    assert deletion_variants(["A", "B", "C"], 0) == {("A", "B", "C")}
    assert deletion_variants(["A", "B", "C"], 1) == {("A", "B", "C"), ("B", "C"), ("A", "C"), ("A", "B")}
    two = deletion_variants(["A", "B", "C"], 2)
    assert len(two) == 1 + 3 + 3 and ("A",) in two and ("B",) in two and ("C",) in two and () not in two
    # equal tokens give the same variant: the set has no duplicates
    assert deletion_variants(["A", "A"], 1) == {("A", "A"), ("A",)}
    assert deletion_variants([], 2) == {()}
    assert deletion_variants(["A"], 2) == {("A",), ()}                          # cannot delete more than there is


def test_index_query_finds_the_keys_within_the_distance():
    keys = [["K", "A"], ["K", "A", "T", "A"], ["T", "A"], ["K", "O", "T", "A"], ["S", "A", "L", "E", "M"]]
    idx = DeletionIndex(keys)
    assert idx.query(["K", "A", "T", "A"], 0) == [(1, 0)]
    assert idx.query(["K", "A", "T", "A"], 1) == [(1, 0), (3, 1)]              # nearest first
    assert idx.query(["K", "A", "T", "A"], 2) == [(1, 0), (3, 1), (0, 2), (2, 2)]    # ties by position
    assert idx.query(["K", "A", "T"], 1) == [(0, 1), (1, 1)]                    # one deletion from key 1 and key 0
    assert idx.query(["S", "A", "L", "M"], 1) == [(4, 1)]
    assert idx.query(["X", "Y", "Z"], 2) == []
    assert idx.query([], 0) == [] and idx.query([], 1) == []
    assert idx.query([], 2) == [(0, 2), (2, 2)]                                 # the two-sign keys, two insertions away


def test_index_query_checks_hits_with_a_real_edit_distance():
    # "A B" and "B A" share the variants "A" and "B" (one deletion each), but they are two edits apart
    idx = DeletionIndex([["A", "B"]])
    assert idx.query(["B", "A"], 1) == []
    assert idx.query(["B", "A"], 2) == [(0, 2)]
    # likewise a key and a query that share only the empty variant
    assert DeletionIndex([["X"]]).query(["Y"], 1) == [(0, 1)]
    assert DeletionIndex([["X", "Y"]]).query(["P", "Q"], 1) == []


def test_index_query_refuses_a_distance_it_does_not_hold():
    with pytest.raises(ValueError):
        DeletionIndex([["A"]]).query(["A"], 3)
    assert DeletionIndex([["A", "B", "C"]], max_deletions=3).query(["A"], 3) == [(0, 2)]
    assert DeletionIndex([]).query(["A"], 2) == []


def _random_keys(rng, n, alphabet, longest):
    keys = set()
    while len(keys) < n:
        keys.add(tuple(rng.choice(alphabet) for _ in range(rng.randint(1, longest))))
    return sorted(keys)


def test_index_query_equals_brute_force():
    """For many random word lists and queries, the index returns exactly the keys within the distance."""
    rng = random.Random(2)
    for alphabet, n_keys, longest in (("ab", 25, 6), ("abcd", 60, 7), ("abcdefgh", 80, 8)):
        keys = _random_keys(rng, n_keys, alphabet, longest)
        idx = DeletionIndex(keys)
        for _ in range(150):
            q = tuple(rng.choice(alphabet) for _ in range(rng.randint(0, longest + 2)))
            for d in (0, 1, 2):
                want = sorted(((edit_distance(q, k), i) for i, k in enumerate(keys) if edit_distance(q, k) <= d))
                assert idx.query(q, d) == [(i, dd) for dd, i in want], (q, d)


# ---------------------------------------------------------------------------------------- proposals

TOY = [["K", "A"], ["T", "A"], ["K", "A", "T", "A"], ["S", "A", "L"], ["M", "A", "K", "E", "M", "A", "K"],
       ["P", "A", "P", "A"]]


def test_propose_pairs_toy_example():
    idx = DeletionIndex(TOY)
    # two exact one- or two-sign parts; the word-space token is dropped before the reading is cut
    assert propose_pairs([["K", "A", "_", "T", "A"]], idx) == [(0, 1, 0)]
    assert propose_pairs([["K", "A", "T", "A"]], idx) == [(0, 1, 0)]
    # a part of three or four signs may be one edit away: S A L E (4 signs) ~ S A L, T A exact
    assert propose_pairs([["S", "A", "L", "E", "T", "A"]], idx) == [(3, 1, 1)]
    # a part of five or more signs may be two edits away: M A K E M O K ~ M A K E M A K (1 edit), P A P A exact. A
    # cut one sign earlier also gives a pair, from parts two edits away each: ranked after the better one
    assert propose_pairs([["M", "A", "K", "E", "M", "O", "K", "P", "A", "P", "A"]], idx) == [(4, 5, 1), (4, 2, 4)]
    # a part of one or two signs must match exactly
    assert propose_pairs([["K", "A", "T", "O"]], idx) == []
    assert propose_pairs([["K", "O", "T", "A"]], idx) == []
    # a reading with one sign, or none, cannot be cut
    assert propose_pairs([["K"], [], ["_"], ["_", "K", "_"]], idx) == []


def test_propose_pairs_uses_every_free_reading_once_and_keeps_the_smaller_distance():
    idx = DeletionIndex(TOY)
    readings = [["K", "A", "_", "T", "A"], ["P", "A", "P", "A", "_", "K", "A"], ["K", "A", "T", "A"]]
    assert propose_pairs(readings, idx) == [(0, 1, 0), (5, 0, 0)]               # ranked by (sum, A, B)
    assert propose_pairs(readings[::-1], idx) == [(0, 1, 0), (5, 0, 0)]         # the order of the readings is free
    assert propose_pairs(readings + readings, idx) == [(0, 1, 0), (5, 0, 0)]    # repeats change nothing
    # (S A L, T A) is found at two cut points with sums 1 and 1; (K A T A, T A) cannot be exact: pair kept once
    two_cuts = [["S", "A", "L", "E", "T", "A"]]
    assert len(propose_pairs(two_cuts, idx)) == 1
    # the same pair from a nearer reading wins: S A L T A ~ (S A L, T A) at distance 0
    assert propose_pairs([["S", "A", "L", "E", "T", "A"], ["S", "A", "L", "T", "A"]], idx) == [(3, 1, 0)]


def _brute_force(readings, keys, cap):
    best = {}
    for reading in readings:
        toks = [t for t in reading if t != "_"]
        for cut in range(1, len(toks)):
            left, right = toks[:cut], toks[cut:]
            ml = [(i, edit_distance(left, k)) for i, k in enumerate(keys) if edit_distance(left, k) <= max_edit(len(left))]
            mr = [(i, edit_distance(right, k)) for i, k in enumerate(keys) if edit_distance(right, k) <= max_edit(len(right))]
            for ia, da in ml:
                for ib, db in mr:
                    best[(ia, ib)] = min(best.get((ia, ib), 99), da + db)
    return [(ia, ib, d) for d, ia, ib in sorted((d, ia, ib) for (ia, ib), d in best.items())][:cap]


def test_propose_pairs_equals_brute_force():
    rng = random.Random(3)
    for alphabet, n_keys, longest in (("ab", 20, 5), ("abc", 40, 6), ("abcde", 70, 7)):
        keys = _random_keys(rng, n_keys, alphabet, longest)
        idx = DeletionIndex(keys)
        for _ in range(60):
            readings = [[rng.choice(alphabet + "__") for _ in range(rng.randint(0, 11))] for _ in range(rng.randint(1, 4))]
            for cap in (400, 5):
                assert propose_pairs(readings, idx, cap) == _brute_force(readings, keys, cap), (readings, cap)


def test_propose_pairs_keeps_at_most_400_ranked_by_distance_then_position():
    # 26 words within two edits of "a b c d e" and 26 of "f g h i j": 676 pairs from the cut in the middle alone
    def near(base, subs):
        out = [tuple(base)]
        out += [tuple(base[:p]) + (s,) + tuple(base[p + 1:]) for p in range(len(base)) for s in subs]
        return out
    left, right = near("abcde", "xyzwv"), near("fghij", "xyzwv")
    keys = left + right
    idx = DeletionIndex(keys)
    reading = list("abcdefghij")
    got = propose_pairs([reading], idx)
    assert len(got) == twoword.MAX_PROPOSALS == 400
    assert got == _brute_force([reading], keys, 400)
    assert got[0] == (0, 26, 0)                                                 # both parts exact
    sums = [d for _, _, d in got]
    assert sums == sorted(sums)
    assert sums.count(0) == 1 and sums.count(1) == 50 and set(sums) == {0, 1, 2}
    within_sum2 = [(ia, ib) for ia, ib, d in got if d == 2]
    assert within_sum2 == sorted(within_sum2)                                   # ties by (A, B)
    assert len(within_sum2) == 349 and len(set((ia, ib) for ia, ib, _ in got)) == 400
    assert len(propose_pairs([reading], idx, 7)) == 7


# ---------------------------------------------------------------------------------------- merged ranking, pure


def test_merge_ranked_orders_words_and_pairs_by_effective_score():
    singles = [(3.0, 5), (4.0, 6), (9.5, 7)]
    pairs = [(1, 2, -1.0), (3, 4, 1.0), (8, 9, 0.5)]            # losses; the penalty is added
    got = merge_ranked(singles, pairs, 8.0, k=5)
    assert got == [("word", 5, None, 3.0, 3.0), ("word", 6, None, 4.0, 4.0), ("pair", 1, 2, -1.0, 7.0),
                   ("pair", 8, 9, 0.5, 8.5), ("pair", 3, 4, 1.0, 9.0)]
    assert merge_ranked(singles, pairs, 0.0, k=2) == [("pair", 1, 2, -1.0, -1.0), ("pair", 8, 9, 0.5, 0.5)]
    assert merge_ranked(singles, pairs, 100.0, k=4) == [("word", 5, None, 3.0, 3.0), ("word", 6, None, 4.0, 4.0),
                                                        ("word", 7, None, 9.5, 9.5), ("pair", 1, 2, -1.0, 99.0)]
    assert merge_ranked([], [], 8.0) == []
    assert merge_ranked(singles, [], 8.0, k=2) == [("word", 5, None, 3.0, 3.0), ("word", 6, None, 4.0, 4.0)]


def test_merge_ranked_ties_put_words_first_and_infinite_scores_last():
    inf = float("inf")
    got = merge_ranked([(8.0, 1), (inf, 2)], [(3, 4, 0.0), (5, 6, inf), (7, 8, 0.0)], 8.0, k=10)
    # the word and the first pair both have 8.0: the word first; then the other pair; infinities last, words first
    assert [(e[0], e[1]) for e in got] == [("word", 1), ("pair", 3), ("pair", 7), ("word", 2), ("pair", 5)]
    assert got[-1][4] == inf


# ---------------------------------------------------------------------------------------- the ensemble


def _toy_ensemble(tmp_path, words, n_models=2, extra=()):
    """An Ensemble with the project's alphabet and a word list of `words` = [(headword, tokens)]. Its models are
    small and random and are never run: a test replaces `logps` with log-probabilities made by hand."""
    torch = pytest.importorskip("torch")
    from chinukpipa.htr.data import vocab
    from chinukpipa.htr.model import CRNN
    from chinukpipa.text.readcrops import Ensemble
    lex = tmp_path / "toy.tsv"
    lex.write_text("headword\tbest_conf\ttokens\n" + "".join(f"{h}\tA\t{t}\n" for h, t in words), encoding="utf-8")
    torch.manual_seed(0)
    models = [CRNN(len(vocab()), hidden=8).eval() for _ in range(n_models)]
    return Ensemble.from_models(models, lexicon=str(lex), extra_lexicons=extra)


def _handmade(ens, readings, n_frames, margin=8.0, blanks=True):
    """Log-probabilities as if each model read each word as given. readings[m][i] is a string of tokens ("K A _ T A"):
    each token is peaked for one frame, followed by one blank frame (unless blanks=False), and the rest of the frames
    are blank. The ensemble's `logps` is replaced so that read_batch uses them. Returns (outs, T)."""
    torch = ens.torch
    outs = []
    for per_word in readings:
        x = torch.zeros(n_frames, len(per_word), ens.C)
        for i, tokens in enumerate(per_word):
            labels = [lab for tok in tokens.split() for lab in ((tok, "-") if blanks else (tok,))]
            labels += ["-"] * (n_frames - len(labels))
            for t, lab in enumerate(labels[:n_frames]):
                x[t, i, 0 if lab == "-" else ens.token_class[lab]] += margin
        outs.append(x.log_softmax(-1))
    T = torch.full((len(readings[0]),), n_frames, dtype=torch.long)
    ens.logps = lambda arrs: (outs, T)
    return outs, T


def _loss(ens, outs, i, n_frames, tokens):
    """Independent computation: the mean over the models of the CTC loss of `tokens`, one model and one target at a time."""
    torch = ens.torch
    tg = torch.tensor([[ens.token_class[t] for t in tokens.split()]])
    losses = [torch.nn.functional.ctc_loss(lp[:n_frames, i:i + 1], tg, torch.tensor([n_frames]),
                                           torch.tensor([tg.shape[1]]), blank=0, reduction="none")[0] for lp in outs]
    return float(sum(losses) / len(losses))


WORDS = [("ka", "K A"), ("ta", "T A"), ("sa", "S A"), ("ma", "M A"), ("kam", "K A M")]


def test_pair_decoding_ranks_words_and_pairs_by_effective_score(tmp_path):
    ens = _toy_ensemble(tmp_path, WORDS)
    n = 14
    # word 0: both models read "K A _ T A"; word 1: "S A"; word 2: one model reads "K A _ T A", the other "M A _ T A"
    outs, T = _handmade(ens, [["K A _ T A", "S A", "K A _ T A"], ["K A _ T A", "S A", "M A _ T A"]], n)
    arrs = [np.zeros((64, 8), np.uint8)] * 3
    rows = ens.read_batch(arrs, pair_penalty=8.0)
    assert [r["n_pair_proposals"] for r in rows] == [1, 0, 2]
    assert [r["free_tokens"] for r in rows] == ["K A _ T A", "S A", "K A _ T A"]

    first = rows[0]["top5"][0]
    assert first["kind"] == "pair" and first["headword"] == "ka + ta" and first["tokens"] == "K A _ T A"
    assert first["also"] == [] and set(first) == {"headword", "also", "tokens", "score", "p_rel", "kind", "loss"}
    assert first["loss"] < 1.0 and first["score"] == pytest.approx(first["loss"] + 8.0, abs=2e-3)
    assert rows[1]["top5"][0]["kind"] == "word" and rows[1]["top5"][0]["tokens"] == "S A"
    assert all(e["kind"] == "word" for e in rows[1]["top5"])
    # sorted by effective score, nothing out of order, and a word's score is its loss
    for r in rows:
        scores = [e["score"] for e in r["top5"]]
        assert scores == sorted(scores) and len(scores) == 5
        for e in r["top5"]:
            assert e["kind"] in ("word", "pair") and e["loss"] is not None
            if e["kind"] == "word":
                assert e["score"] == e["loss"] and set(e) == {"headword", "also", "tokens", "score", "p_rel", "kind", "loss"}
    # every number agrees with an independent computation over all list words and all proposals
    singles = {t: None for _, t in WORDS}
    pair_tokens = {0: ["K A _ T A"], 1: [], 2: ["K A _ T A", "M A _ T A"]}
    for i in range(3):
        loss = {t: _loss(ens, outs, i, n, t) for t in singles}
        eff = dict(loss)
        for p in pair_tokens[i]:
            a, b = p.split(" _ ")
            loss[p] = min(_loss(ens, outs, i, n, p), _loss(ens, outs, i, n, f"{a} {b}"))
            eff[p] = loss[p] + 8.0
        log_z = math.log(sum(math.exp(-v) for v in eff.values()))
        for e in rows[i]["top5"]:
            assert e["loss"] == pytest.approx(loss[e["tokens"]], abs=2e-3)
            assert e["score"] == pytest.approx(eff[e["tokens"]], abs=2e-3)
            assert e["p_rel"] == pytest.approx(math.exp(-eff[e["tokens"]] - log_z), abs=6e-4)
        want_first = min(eff, key=eff.get)
        assert rows[i]["top5"][0]["tokens"] == want_first


def test_pair_penalty_decides_between_a_pair_and_a_word(tmp_path):
    ens = _toy_ensemble(tmp_path, WORDS)
    _handmade(ens, [["K A _ T A"], ["K A _ T A"]], 14)
    arrs = [np.zeros((64, 8), np.uint8)]
    by_penalty = {p: ens.read_batch(arrs, k=10, pair_penalty=p)[0]["top5"] for p in (0.0, 8.0, 40.0)}
    assert [by_penalty[p][0]["kind"] for p in (0.0, 8.0, 40.0)] == ["pair", "pair", "word"]
    assert by_penalty[40.0][0]["loss"] > 10 and any(e["kind"] == "pair" for e in by_penalty[40.0])
    pair0 = next(e for e in by_penalty[0.0] if e["kind"] == "pair")
    pair40 = next(e for e in by_penalty[40.0] if e["kind"] == "pair")
    assert pair0["loss"] == pair40["loss"] and pair40["score"] == pytest.approx(pair0["score"] + 40.0, abs=2e-3)
    # a larger penalty leaves less of the probability to the pair
    assert pair40["p_rel"] < pair0["p_rel"]
    with pytest.raises(ValueError):
        ens.read_batch(arrs, pair_penalty=float("nan"))
    with pytest.raises(ValueError):
        ens.read_batch(arrs, pair_penalty=float("inf"))


def test_a_pair_is_scored_by_the_better_of_the_targets_with_and_without_the_separator(tmp_path):
    ens = _toy_ensemble(tmp_path, [("ka", "K A"), ("ta", "T A")])
    n = 14
    outs, T = _handmade(ens, [["K A T A", "K A _ T A"], ["K A T A", "K A _ T A"]], n)
    rows = ens.read_batch([np.zeros((64, 8), np.uint8)] * 2, pair_penalty=8.0)
    for i in range(2):
        pair = next(e for e in rows[i]["top5"] if e["kind"] == "pair")
        assert pair["tokens"] == "K A _ T A" and pair["headword"] == "ka + ta"            # always "A _ B"
        with_sep, without = _loss(ens, outs, i, n, "K A _ T A"), _loss(ens, outs, i, n, "K A T A")
        assert pair["loss"] == pytest.approx(min(with_sep, without), abs=2e-3)
        assert pair["loss"] < 1.0 and max(with_sep, without) > 7.0                         # the other target is far worse
        assert pair["score"] == pytest.approx(pair["loss"] + 8.0, abs=2e-3)
    assert rows[0]["top5"][0]["kind"] == rows[1]["top5"][0]["kind"] == "pair"


def test_the_better_target_is_chosen_for_the_ensemble_not_for_each_model(tmp_path):
    """When the models disagree about the separator, the loss is that of the better target under the ensemble's
    mean loss, not the mean of each model's own better target (which would hide the disagreement)."""
    ens = _toy_ensemble(tmp_path, [("ka", "K A"), ("ta", "T A")])
    n = 14
    outs, T = _handmade(ens, [["K A _ T A"], ["K A T A"]], n)           # model 0 reads the separator, model 1 does not
    row = ens.read_batch([np.zeros((64, 8), np.uint8)], pair_penalty=8.0)[0]
    pair = next(e for e in row["top5"] if e["kind"] == "pair")
    with_sep, without = ("K A _ T A", "K A T A")
    mean_with = _loss(ens, outs, 0, n, with_sep)
    mean_without = _loss(ens, outs, 0, n, without)
    per_model_best = np.mean([min(_loss(ens, [lp], 0, n, with_sep), _loss(ens, [lp], 0, n, without)) for lp in outs])
    assert row["free_models_agree"] is False
    assert pair["loss"] == pytest.approx(min(mean_with, mean_without), abs=2e-3)
    assert pair["loss"] > 3.0 and per_model_best < 1.0 and pair["loss"] > per_model_best + 3.0


def test_a_pair_that_does_not_fit_the_image_has_no_score(tmp_path):
    ens = _toy_ensemble(tmp_path, [("kata", "K A T A"), ("as", "A S"), ("s", "S"), ("k", "K")])
    # five frames, one sign each: "K A T A S" fits, but a pair of K A T A with A S has six signs (seven with the
    # separator) and cannot be aligned to five frames
    _handmade(ens, [["K A T A S"], ["K A T A S"]], 5, blanks=False)
    row = ens.read_batch([np.zeros((64, 8), np.uint8)], k=20, pair_penalty=8.0)[0]
    assert row["free_tokens"] == "K A T A S"
    pairs = {e["tokens"]: e for e in row["top5"] if e["kind"] == "pair"}
    assert set(pairs) == {"K A T A _ S", "K A T A _ A S"} and row["n_pair_proposals"] == 2
    fits, too_long = pairs["K A T A _ S"], pairs["K A T A _ A S"]
    assert fits["score"] is not None and fits["loss"] is not None and fits["p_rel"] > 0
    assert too_long["score"] is None and too_long["loss"] is None and too_long["p_rel"] == 0.0
    assert row["top5"][-1] == too_long                        # no score: last
    assert len(row["top5"]) == 6                              # four words and two pairs, all shown
    assert sum(e["p_rel"] for e in row["top5"]) == pytest.approx(1.0, abs=1e-3)


def test_default_rows_have_exactly_the_fields_they_always_had(tmp_path):
    ens = _toy_ensemble(tmp_path, WORDS)
    _handmade(ens, [["K A _ T A"], ["K A _ T A"]], 14)
    arrs = [np.zeros((64, 8), np.uint8)]
    row = ens.read_batch(arrs)[0]
    assert list(row) == ["free_tokens", "free_roman", "free_models_agree", "confidence", "confidence_nonblank",
                         "frames", "top5", "free_best"]
    assert all(list(e) == ["headword", "also", "tokens", "score", "p_rel"] for e in row["top5"])
    assert ens.read_batch(arrs, pair_penalty=None)[0] == row and ens.read_batch(arrs, 5, None)[0] == row
    # the same words in the same order as the pair reading's list words, with the same losses
    paired = ens.read_batch(arrs, pair_penalty=1e9)[0]
    assert [e["tokens"] for e in paired["top5"]] == [e["tokens"] for e in row["top5"]]
    assert [e["score"] for e in paired["top5"]] == [e["score"] for e in row["top5"]]
    # with the penalty out of reach, p_rel is the softmax over the words alone: exactly the default values
    assert [e["p_rel"] for e in paired["top5"]] == pytest.approx([e["p_rel"] for e in row["top5"]], abs=1.5e-4)


def test_the_word_list_with_an_extra_list_appended(tmp_path):
    base = [("ka", "K A"), ("ta", "T A")]
    extra = tmp_path / "extra.tsv"
    extra.write_text("headword\tbest_conf\ttokens\tsource\nS.T.\tA\tS T\tbrief_forms.yaml:ST\n"
                     "low\tC\tL O\tx\n"                       # not an A/B row: left out
                     "ka again\tB\tK A\tx\n", encoding="utf-8")   # same tokens as a main-list word: a second headword
    ens = _toy_ensemble(tmp_path, base, extra=[str(extra)])
    assert ens.keys == ["K A", "T A", "S T"] and ens.N == 3
    assert ens.heads == {"K A": ["ka", "ka again"], "T A": ["ta"], "S T": ["S.T."]}
    assert ens.cand_classes[2] == [ens.token_class["S"], ens.token_class["T"]]
    plain = _toy_ensemble(tmp_path, base)
    assert plain.keys == ["K A", "T A"]
    # two extra lists come after the main one, in the order given
    second = tmp_path / "second.tsv"
    second.write_text("headword\tbest_conf\ttokens\nmama\tA\t\n", encoding="utf-8")        # no tokens column value: spelled
    ens2 = _toy_ensemble(tmp_path, base, extra=[str(extra), str(second)])
    assert ens2.keys == ["K A", "T A", "S T", "M A M A"]


def test_the_brief_forms_list_has_exactly_the_one_row_that_was_tested():
    path = os.path.join(REPO, "data", "lexicon", "brief_forms.tsv")
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    assert rows == [["headword", "best_conf", "tokens", "source"], ["S.T.", "A", "S T", "brief_forms.yaml:ST"]]
    assert open(path, "rb").read().endswith(b"\n") and b"\r" not in open(path, "rb").read()
    import yaml
    brief = yaml.safe_load(open(os.path.join(REPO, "data", "signs", "brief_forms.yaml"), encoding="utf-8"))
    st = next(item for item in brief["items"] if item["id"] == "ST")           # the row's source exists
    assert "God" in st["expansion"]


def test_the_scorer_reads_the_brief_forms_list_as_an_extra_list(tmp_path):
    pytest.importorskip("torch")
    from chinukpipa.text import gtrows
    path = os.path.join(REPO, "data", "lexicon", "brief_forms.tsv")
    main = tmp_path / "main.tsv"
    main.write_text("headword\tbest_conf\ttokens\nka\tA\tK A\n", encoding="utf-8")
    assert gtrows.lexicon_keys(path) == {"S T"}
    assert gtrows.lexicon_keys(str(main)) == {"K A"}
    assert gtrows.lexicon_keys(str(main), [path]) == {"K A", "S T"}


# ---------------------------------------------------------------------------------------- command line


class _SerialPool:
    """multiprocessing.Pool without processes, for the tests."""

    def __init__(self, n):
        pass

    def imap(self, f, tasks):
        return map(f, tasks)

    imap_unordered = imap

    def close(self):
        pass

    def join(self):
        pass


def _stage_a_page(pages, book="bk", leaf=1, widths=(120, 90, 150)):
    """A made-up stage-A page: three word crops on one text line (`_words.npz`, `_seg.json`)."""
    os.makedirs(pages, exist_ok=True)
    pad, arrs, boxes, x0 = 5, [], [], 10
    for k, w in enumerate(widths):
        box = [x0, 10, x0 + w, 50]
        crop = np.full((40 + 2 * pad, w + 2 * pad), 255, np.uint8)
        crop[18:28, 8:w] = 25 + 10 * k                          # a bar and two uprights: something for the reader to see
        crop[8:44, 12:20] = 25
        crop[8:44, w - 20:w - 12] = 25
        arrs.append(crop)
        boxes.append(box)
        x0 += w + 20
    np.savez_compressed(os.path.join(pages, f"{book}_{leaf}_words.npz"),
                        flat=np.concatenate([a.ravel() for a in arrs]),
                        shapes=np.array([a.shape for a in arrs], dtype=np.int32),
                        boxes=np.array(boxes, dtype=np.int32), line=np.ones(len(arrs), dtype=np.int32),
                        index=np.arange(1, len(arrs) + 1, dtype=np.int32))
    seg = {"stats": {"h_med": 40, "word_gap_threshold_px": 14, "gap_stats": {"mode_large_px": 30}},
           "lines": [{"kind": "text", "number": 1, "words": [{"kind": "word", "bbox": b} for b in boxes]}]}
    with open(os.path.join(pages, f"{book}_{leaf}_seg.json"), "w", encoding="utf-8") as fh:
        json.dump(seg, fh)


def _random_model_files(tmp_path, n=2):
    import torch
    from chinukpipa.htr.data import vocab
    from chinukpipa.htr.model import CRNN
    paths = []
    for seed in range(n):
        torch.manual_seed(seed)
        m = CRNN(len(vocab()), hidden=16)
        path = tmp_path / f"m{seed}.pt"
        torch.save(m.state_dict(), path)
        paths.append(str(path))
    return paths


def _word_list(tmp_path):
    """Single-sign words and a few longer ones, so that the models' readings can be cut into list words."""
    from chinukpipa.htr.data import vocab
    lines = [f"{t.lower()}\tA\t{t}" for t in vocab() if t != "_"] + ["kata\tA\tK A T A", "mama\tA\tM A M A"]
    path = tmp_path / "list.tsv"
    path.write_text("headword\tbest_conf\ttokens\n" + "\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _run_readcrops(monkeypatch, argv):
    from chinukpipa.text import readcrops
    monkeypatch.setattr(readcrops, "Pool", _SerialPool)
    monkeypatch.setattr(sys, "argv", ["readcrops"] + argv)
    readcrops.main()


def test_readcrops_options_reach_the_ensemble_and_the_files(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pages = str(tmp_path / "pages")
    _stage_a_page(pages)
    models, lex = _random_model_files(tmp_path), _word_list(tmp_path)
    brief = os.path.join(REPO, "data", "lexicon", "brief_forms.tsv")
    common = ["--pages", pages, "--models", *models, "--lexicon", lex, "--device", "cpu", "--scale", "0.8",
              "--workers", "1"]
    _run_readcrops(monkeypatch, common + ["--out-dir", str(tmp_path / "plain")])
    _run_readcrops(monkeypatch, common + ["--extra-lexicon", brief, "--pair-penalty", "8", "--out-dir",
                                          str(tmp_path / "pair")])
    plain = json.load(open(tmp_path / "plain" / "bk_1_read.json", encoding="utf-8"))
    pair = json.load(open(tmp_path / "pair" / "bk_1_read.json", encoding="utf-8"))
    # without the options: the file keeps its old shape
    assert list(plain) == ["item", "leaf", "scale", "models", "n_candidates", "reading", "words"]
    assert all("n_pair_proposals" not in w and all("kind" not in e for e in w["top5"]) for w in plain["words"])
    # with them: the penalty and the extra list are recorded, the candidates grow by the S.T. row
    assert list(pair) == ["item", "leaf", "scale", "models", "n_candidates", "extra_lexicons", "pair_penalty",
                          "reading", "words"]
    assert pair["pair_penalty"] == 8.0 and pair["extra_lexicons"] == ["brief_forms.tsv"]
    assert pair["n_candidates"] == plain["n_candidates"] + 1
    for w in pair["words"]:
        assert isinstance(w["n_pair_proposals"], int) and 0 <= w["n_pair_proposals"] <= 400
        assert all(e["kind"] in ("word", "pair") for e in w["top5"])
        scores = [e["score"] for e in w["top5"] if e["score"] is not None]
        assert scores == sorted(scores)
    # the rest of the reading does not depend on the options
    for a, b in zip(plain["words"], pair["words"]):
        for key in ("line", "index", "bbox", "free_tokens", "free_roman", "free_models_agree", "confidence",
                    "confidence_nonblank", "frames", "free_best"):
            assert a[key] == b[key]
    assert os.path.exists(tmp_path / "pair" / "bk_book.json")           # the summary still works with pair entries


def test_remerge_must_be_run_with_the_stage_b_penalty_and_records_it(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from chinukpipa.text import remerge
    pages = str(tmp_path / "pages")
    _stage_a_page(pages)
    models, lex = _random_model_files(tmp_path), _word_list(tmp_path)
    brief = os.path.join(REPO, "data", "lexicon", "brief_forms.tsv")
    read_args = ["--pages", pages, "--models", *models, "--lexicon", lex, "--device", "cpu", "--scale", "0.8",
                 "--workers", "1"]
    _run_readcrops(monkeypatch, read_args + ["--pair-penalty", "8", "--extra-lexicon", brief,
                                             "--out-dir", str(tmp_path / "read8")])
    _run_readcrops(monkeypatch, read_args + ["--out-dir", str(tmp_path / "read")])
    _run_readcrops(monkeypatch, read_args + ["--extra-lexicon", brief, "--out-dir", str(tmp_path / "readb")])
    # a negative penalty puts pairs first wherever there is a proposal: the joins must still not change
    _run_readcrops(monkeypatch, read_args + ["--pair-penalty", "-20", "--extra-lexicon", brief,
                                             "--out-dir", str(tmp_path / "read-20")])

    def run(read_dir, out_dir, *flags):
        monkeypatch.setattr(sys, "argv", ["remerge", "--pages", pages, "--read", str(read_dir), "--out-dir",
                                          str(out_dir), "--models", *models, "--lexicon", lex, "--device", "cpu",
                                          *flags])
        remerge.main()

    with pytest.raises(SystemExit) as e:                    # stage B with a penalty, stage C without: refused
        run(tmp_path / "read8", tmp_path / "m_bad", "--extra-lexicon", brief)
    assert "pair penalty" in str(e.value) and "8.0" in str(e.value)
    with pytest.raises(SystemExit) as e:                    # another value: refused as well
        run(tmp_path / "read8", tmp_path / "m_bad2", "--extra-lexicon", brief, "--pair-penalty", "4")
    assert "pair penalty" in str(e.value)
    with pytest.raises(SystemExit) as e:                    # stage B without, stage C with: refused
        run(tmp_path / "read", tmp_path / "m_bad3", "--pair-penalty", "8")
    assert "pair penalty" in str(e.value)

    run(tmp_path / "read8", tmp_path / "merged8", "--extra-lexicon", brief, "--pair-penalty", "8")
    run(tmp_path / "read", tmp_path / "merged")
    run(tmp_path / "readb", tmp_path / "mergedb", "--extra-lexicon", brief)
    m8 = json.load(open(tmp_path / "merged8" / "bk_1_merged.json", encoding="utf-8"))
    m0 = json.load(open(tmp_path / "merged" / "bk_1_merged.json", encoding="utf-8"))
    mb = json.load(open(tmp_path / "mergedb" / "bk_1_merged.json", encoding="utf-8"))
    assert m8["pair_penalty"] == 8.0 and m8["extra_lexicons"] == ["brief_forms.tsv"]
    assert "best_word_score" in m8["merge_rule"]
    # two-word readings take no part in the join decisions: the same word list without pairs joins the same boxes,
    # and best_word_score is the first-choice score of the reading without pairs
    assert [w["boxes"] for w in m8["words"]] == [w["boxes"] for w in mb["words"]] and m8["joins"] == mb["joins"]
    run(tmp_path / "read-20", tmp_path / "merged-20", "--extra-lexicon", brief, "--pair-penalty", "-20")
    mn = json.load(open(tmp_path / "merged-20" / "bk_1_merged.json", encoding="utf-8"))
    assert any(w["top5"] and w["top5"][0]["kind"] == "pair" for w in mn["words"])     # pairs do come first here
    assert [w["boxes"] for w in mn["words"]] == [w["boxes"] for w in mb["words"]] and mn["joins"] == mb["joins"]
    r8 = json.load(open(tmp_path / "read8" / "bk_1_read.json", encoding="utf-8"))
    rb = json.load(open(tmp_path / "readb" / "bk_1_read.json", encoding="utf-8"))
    assert [w["best_word_score"] for w in r8["words"]] == [w["top5"][0]["score"] for w in rb["words"]]
    assert all("best_word_score" not in w for w in rb["words"])
    assert "pair_penalty" not in m0 and "extra_lexicons" not in m0
    assert m0["merge_rule"] == "join if gap < gap_stats.mode_large_px and best loss(joined) < sum of best losses"
    assert list(m0) == ["item", "leaf", "scale", "models", "lexicon", "merge_rule", "gap_max_px", "joins", "reading",
                        "words"]
    assert all("kind" in w["top5"][0] for w in m8["words"] if w["top5"])


class _FakeEnsemble:
    """Stands in for readcrops.Ensemble in remerge: a joined image reads as a pair first (effective score 10.0, loss
    2.0 + penalty 8.0), with the best list word's score WORD; it records how it was asked."""

    N = 100
    word = 12.0
    calls: list = []

    def __init__(self, models, lexicon, device, extra_lexicons=()):
        _FakeEnsemble.calls = []

    def read_batch(self, arrs, k=5, pair_penalty=None):
        _FakeEnsemble.calls.append(pair_penalty)
        pair = {"headword": "x + y", "also": [], "tokens": "X _ Y", "score": 10.0, "p_rel": 0.5, "kind": "pair",
                "loss": 2.0}
        word = {"headword": "w", "also": [], "tokens": "W", "score": self.word, "p_rel": 0.1, "kind": "word",
                "loss": self.word}
        if pair_penalty is None:      # without the option the reader knows no pairs: the list word is first
            return [{"free_tokens": "", "top5": [{k_: word[k_] for k_ in ("headword", "also", "tokens", "score",
                                                                          "p_rel")}], "free_best": None}]
        return [{"free_tokens": "", "top5": [pair, word], "free_best": None, "n_pair_proposals": 1,
                 "best_word_score": self.word}]


def test_stage_c_decides_joins_on_list_word_scores_only(tmp_path, monkeypatch):
    """remerge joins two boxes when the joined image's best list-word score is lower than the sum of the pieces'.
    With a pair penalty a two-word reading takes no part in that decision (here the joined image reads as a pair with
    effective score 10.0, lower than 6.0 + 5.0, but its best list word scores 12.0: no join); the unit that is joined
    keeps its reading with the pairs. A stage-B file without best_word_score is refused."""
    pytest.importorskip("torch")
    from chinukpipa.text import readcrops, remerge
    pages = str(tmp_path / "pages")
    _stage_a_page(pages, widths=(100, 100))

    def piece(index, score, penalty):
        w = {"line": 1, "index": index, "bbox": [0, 0, 1, 1], "free_tokens": "", "top5": [
            {"headword": "w", "also": [], "tokens": "W", "score": score, "p_rel": 0.9}], "free_best": None}
        if penalty is not None:
            w["top5"][0].update({"kind": "word", "loss": score})
            w.update({"n_pair_proposals": 0, "best_word_score": score})
        return w

    def run(first, second, penalty, word=12.0, drop_field=False):
        tag = f"{first}_{second}_{penalty}_{word}_{drop_field}"
        read = tmp_path / f"read_{tag}"
        read.mkdir()
        words = [piece(1, first, penalty), piece(2, second, penalty)]
        if drop_field:
            del words[1]["best_word_score"]
        (read / "bk_1_read.json").write_text(json.dumps({
            "item": "bk", "leaf": 1, "scale": 0.8, "models": ["m"], "n_candidates": 100,
            **({} if penalty is None else {"pair_penalty": penalty}), "words": words}), encoding="utf-8")
        out = tmp_path / f"out_{tag}"
        monkeypatch.setattr(readcrops, "Ensemble", _FakeEnsemble)
        monkeypatch.setattr(_FakeEnsemble, "word", word)
        monkeypatch.setattr(sys, "argv", ["remerge", "--pages", pages, "--read", str(read), "--out-dir", str(out),
                                          "--models", "m", "--lexicon", "unused.tsv", "--device", "cpu"]
                            + ([] if penalty is None else ["--pair-penalty", str(penalty)]))
        remerge.main()
        return json.load(open(out / "bk_1_merged.json", encoding="utf-8"))

    apart = run(6.0, 5.0, 8.0)             # pair 10.0 < 11.0, but list word 12.0 < 11.0 is false: two words
    assert apart["joins"] == 0 and len(apart["words"]) == 2
    assert _FakeEnsemble.calls == [8.0]    # the joined image was read with the penalty
    assert run(6.0, 5.0, None, word=12.0)["joins"] == 0          # the same decision without the option
    joined = run(6.0, 5.0, 8.0, word=10.5)   # list word 10.5 < 11.0: joined, and the unit keeps its pair reading
    assert joined["joins"] == 1 and len(joined["words"]) == 1 and joined["words"][0]["boxes"] == ["1.1", "1.2"]
    assert joined["words"][0]["top5"][0]["kind"] == "pair" and joined["words"][0]["best_word_score"] == 10.5
    assert run(6.0, 5.0, None, word=10.5)["joins"] == 1 and _FakeEnsemble.calls == [None]
    with pytest.raises(SystemExit) as e:
        run(6.0, 5.0, 8.0, drop_field=True)
    assert "best_word_score" in str(e.value)


def test_ranking_uses_the_scores_as_written_and_a_tie_goes_to_the_list_word():
    """Unrounded, the pair below would win by 0.0003 nats (2.0001 + 8 < 10.0004). The files show 10.0 for both, and
    the penalty was chosen on such written scores, so the reader ranks on them: 2.0 + 8.0 is exactly 10.0, a tie,
    which goes to the list word. p_rel is still computed from the unrounded losses."""
    from chinukpipa.text import readcrops
    ens = readcrops.Ensemble.__new__(readcrops.Ensemble)      # only the word list is needed here
    ens.keys = ["K A", "T A", "S E L"]
    ens.heads = {"K A": ["ka"], "T A": ["ta"], "S E L": ["sel"]}
    top = ens._merged_top([10.0004, 11.0], [2, 0], [(0, 1, 2.0001)], 8.0, logz=-9.5, k=3)
    assert [(e["kind"], e["tokens"], e["score"]) for e in top] == [
        ("word", "S E L", 10.0), ("pair", "K A _ T A", 10.0), ("word", "K A", 11.0)]
    assert top[0]["p_rel"] == round(math.exp(-10.0004 + 9.5), 4) != round(math.exp(-10.0 + 9.5), 4)
    assert top[1]["loss"] == 2.0 and top[1]["headword"] == "ka + ta"
