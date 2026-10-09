"""Two-word readings of one word image: proposing "A _ B" from the models' free readings (v0.4, experimental).

The word-list reading of `readcrops` can only return one list word per image, but Le Jeune sometimes writes two
words as one outline. This module holds the part of the two-word decoding that needs no PyTorch: given the free
readings of the models (token strings) and the word list, which pairs of list words are worth scoring? The scoring
itself (CTC loss under the models, plus a penalty) is done by `readcrops.Ensemble.read_batch(pair_penalty=...)`.

    >>> from chinukpipa.text.twoword import DeletionIndex, propose_pairs
    >>> keys = [["K", "A"], ["T", "A"], ["K", "A", "T", "A"]]     # three list words, as token lists
    >>> index = DeletionIndex(keys)
    >>> propose_pairs([["K", "A", "_", "T", "A"]], index)         # (word 0, word 1, edit distances 0 + 0)
    [(0, 1, 0)]

How proposals are made (the same rules for every word image):

1. Every distinct free reading (one per model) is taken with its word-space tokens "_" dropped.
2. Every split point of the reading gives a left and a right part, each of at least one token.
3. Each part is matched to the list words within a token edit distance (insertions, deletions and substitutions
   count one each) that depends on the part's length: 0 for one or two tokens, 1 for three or four, 2 for five or
   more (`max_edit`).
4. Every pair (A, B) of matches is a proposal; the same pair found twice keeps its smaller distance sum. The
   proposals are ranked by that sum, ties by (index of A, index of B), and at most `MAX_PROPOSALS` (400) are kept.

The matching uses a symmetric-deletion index (the idea of the SymSpell spelling corrector): every list word is
stored under all the token strings that remain after deleting up to two of its tokens; a part is looked up under
its own deletion variants, and each hit is then checked with a real edit distance. Two token strings within edit
distance d always share a variant when up to d tokens may be deleted from each, so the index misses nothing; the
check removes the hits that merely share a variant (for example two tokens swapped).
"""
from __future__ import annotations

from collections.abc import Sequence

WORD_SPACE = "_"          # the word-space token (data/signs/tokens.yaml: PUNCT_WORD_SPACE)
MAX_DELETIONS = 2         # the index holds deletion variants up to this many deletions; the largest edit distance asked
MAX_PROPOSALS = 400       # most pairs kept per word image


def edit_distance(a: Sequence, b: Sequence) -> int:
    """Levenshtein distance between two sequences (tokens, or characters): the fewest insertions, deletions and
    substitutions that turn `a` into `b`."""
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[-1]


def max_edit(n_tokens: int) -> int:
    """Largest edit distance allowed between a part of `n_tokens` tokens and a list word: 0 for one or two tokens,
    1 for three or four, 2 for five or more."""
    return 0 if n_tokens <= 2 else 1 if n_tokens <= 4 else 2


def deletion_variants(tokens: Sequence[str], max_deletions: int) -> set[tuple[str, ...]]:
    """All distinct token tuples obtained by deleting at most `max_deletions` tokens (the tokens themselves, with
    no deletion, included)."""
    first = tuple(tokens)
    seen = {first}
    frontier = {first}
    for _ in range(max_deletions):
        step = {s[:i] + s[i + 1:] for s in frontier for i in range(len(s))} - seen
        seen |= step
        frontier = step
    return seen


class DeletionIndex:
    """Symmetric-deletion index over token sequences (see the module docstring).

    `keys` are the word list's candidates as token lists, in the order of the word list; `query` returns positions
    in that list."""

    def __init__(self, keys: Sequence[Sequence[str]], max_deletions: int = MAX_DELETIONS):
        self.keys = [tuple(k) for k in keys]
        self.max_deletions = max_deletions
        self._table: dict[tuple[str, ...], list[int]] = {}
        for i, key in enumerate(self.keys):
            for variant in deletion_variants(key, max_deletions):
                self._table.setdefault(variant, []).append(i)
        lengths = [len(k) for k in self.keys]
        self._shortest, self._longest = (min(lengths), max(lengths)) if lengths else (0, -1)

    def query(self, tokens: Sequence[str], max_dist: int) -> list[tuple[int, int]]:
        """The keys within edit distance `max_dist` of `tokens`, as (position in `keys`, distance), nearest first and
        by position within the same distance. `max_dist` is at most the index's `max_deletions`."""
        if max_dist > self.max_deletions:
            raise ValueError(f"the index holds up to {self.max_deletions} deletions; asked for distance {max_dist}")
        q = tuple(tokens)
        if not self.keys or len(q) < self._shortest - max_dist or len(q) > self._longest + max_dist:
            return []
        hits = {i for v in deletion_variants(q, max_dist) for i in self._table.get(v, ())}
        found = [(i, d) for i in hits if (d := edit_distance(q, self.keys[i])) <= max_dist]
        found.sort(key=lambda m: (m[1], m[0]))
        return found


def propose_pairs(readings: Sequence[Sequence[str]], index: DeletionIndex,
                  max_proposals: int = MAX_PROPOSALS) -> list[tuple[int, int, int]]:
    """Pairs of list words worth scoring for one word image, from the models' free readings (token strings).

    Returns [(position of A, position of B, edit distance of A + edit distance of B)], best first (smallest sum, ties
    by the two positions), at most `max_proposals` of them. A reading with fewer than two tokens (after dropping the
    word-space tokens) gives none."""
    cache: dict[tuple[str, ...], list[tuple[int, int]]] = {}

    def match(part: tuple[str, ...]) -> list[tuple[int, int]]:
        if part not in cache:
            cache[part] = index.query(part, max_edit(len(part)))
        return cache[part]

    best: dict[tuple[int, int], int] = {}
    done: set[tuple[str, ...]] = set()
    for reading in readings:
        toks = tuple(t for t in reading if t != WORD_SPACE)
        if len(toks) < 2 or toks in done:
            continue
        done.add(toks)
        for cut in range(1, len(toks)):
            left = match(toks[:cut])
            right = match(toks[cut:]) if left else []
            for ia, da in left:
                for ib, db in right:
                    if (ia, ib) not in best or da + db < best[(ia, ib)]:
                        best[(ia, ib)] = da + db
    ranked = sorted((d, ia, ib) for (ia, ib), d in best.items())
    return [(ia, ib, d) for d, ia, ib in ranked[:max_proposals]]


def merge_ranked(singles: Sequence[tuple[float, int]], pairs: Sequence[tuple[int, int, float]], penalty: float,
                 k: int = 5) -> list[tuple[str, int, int | None, float, float]]:
    """The k best readings of one word image by effective score: the loss for a list word, the loss plus `penalty`
    for a pair.

    `singles`: [(loss, position in the word list)], the best ones, any order. `pairs`: [(position of A, position of
    B, loss)] in proposal order. Returns [(kind, position of A or of the word, position of B or None, loss,
    effective score)], lowest effective score first ("word" before "pair" and earlier before later when scores are
    equal; infinite scores last)."""
    entries = [("word", j, None, loss, loss) for loss, j in singles]
    entries += [("pair", ia, ib, loss, loss + penalty) for ia, ib, loss in pairs]
    entries.sort(key=lambda e: e[4])          # stable: ties keep the order above
    return entries[:k]
