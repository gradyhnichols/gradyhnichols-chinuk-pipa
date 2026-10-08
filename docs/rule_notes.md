# Notes on the spelling-to-sign rules

The transliterator (`chinukpipa/translit.py`, rules in `data/signs/tokens.yaml`) predicts which shorthand signs
go with each of Le Jeune's Roman spellings. Those predictions are hypotheses. This file records where the
ground-truth images have led to a change, and questions still open. Nothing here has been reviewed by a
specialist or by any person yet; "viewed" below means viewed by the AI (Claude) on the image crops.

**How leads are found.** The recognizer (`chinukpipa/htr/`) is trained on rule-labelled words and then reads
words it has never seen. Where its readings and the rule labels disagree in the same spelling context across
many different words, either the recognizer has a weakness or the rule is wrong; the word images are then
viewed. The counts below come from the five runs in `results/sweep5/base_v*.json` (each holds out a different
random 20% of the words, so a word can be counted in more than one run). They were made with the rules as they
stood before the revision below.

## Revised (provisional): *z* → the S sign (2026-10-08)

- **Old rule:** `z` → TS, following the *z = ts* entry of the English row in Le Jeune's 1898 sign table
  (LJ1898 p. 2, as recorded in `data/signs/signs.yaml`). **New rule:** `z` → S. The change applies to every
  *z*; in the ground truth *z* occurs only at the end of a word (*Aïaz*, *Legliz*, *Noz*, *Snaz*, *Spoz*,
  *Tanaz*, *Tamanoaz*).
- **Recognizer:** at the 12 held-out occurrences of a word-final *z* (7 distinct spellings), labelled TS, the
  free reading has S in 8, TS in 1 and nothing in 3. All three gaps are in *Legliz Katolik*, which was misread
  as a whole.
- **Images viewed:** *Aïaz*, *Tanaz* and *Noz* (1892 list), *tanaz'* (1898), *Tamanoaz* (1892) and *tamano'az*
  (1898). Each ends in a curve like the fonts' S sign. None shows the dot that the fonts draw in the TS sign
  (U+1BC25 DUPLOYAN LETTER S WITH DOT).
- **Caveat:** the 1898 table entry concerns English; these are Chinook words. Whether *z* inside a word, or in
  other texts, behaves the same way is untested.
- **Effect:** 10 rows of `data/gt/vocab_annotations.jsonl` and 7 of `data/gt/copy_annotations.jsonl` were
  relabelled.

## Open questions

- **An *i* next to another vowel** (*Aïlan*, *Maika*, *Heloïma*, *Kiutan*, *Lapiosh* …). The rules write it
  as the E sign. Where the *i* has a vowel letter on at least one side, the recognizer misread that E in 39 of 67
  held-out occurrences (25 dropped; 43 distinct spellings). Where it has no vowel on either side, it misread 23 of
  93. Reproduce with `python -m chinukpipa.htr.contexts "results/sweep5/base_v*.json" --token E --latin i`.
  The images have not settled this yet. Possibilities include a diphthong written as one sign, or a form of *i*
  the rules do not model.
- ***tsh*** (*Tsher*, *Tshok*, *Klootshmen*, *Patlatsh*). The rules write TS + H. The TS of *tsh* was misread
  in 7 of 8 held-out occurrences, against 5 of 16 for a word-initial *ts* before a vowel. In the three 1892
  crops viewed (*Tshok*, *Tsher*, *Tshiken*), no separate H dot is visible. There are too few clear examples to
  decide.

Corrections and checks of individual words are welcome as issues.
