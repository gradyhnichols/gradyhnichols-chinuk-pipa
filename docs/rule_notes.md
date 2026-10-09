# Notes on the spelling-to-sign rules

The transliterator (`chinukpipa/translit.py`, rules in `data/signs/tokens.yaml`) predicts which shorthand signs
go with each of Le Jeune's Roman spellings. Those predictions are hypotheses. This file records where evidence
has led to a change, and questions still open. Nothing here has been reviewed by a specialist or by any person
yet. "Viewed" below means viewed by the AI (Claude) on the scans; page references to Le Jeune's sign tables are
the AI's readings of those scans.

**Four kinds of evidence are used.**
1. *Sources:* Le Jeune's own sign tables (1891 *Elements of Shorthand*, `Ayer_PM846_L47_1891`; 1892 list,
   `cihm_15474`; 1896 *Wawa Shorthand Instructor*, `wawashorthandins00leje`; 1898 *Rudiments*, `cihm_15465`)
   and Robertson (2011), whose normalized spellings we use as a guide to how the signs are read.
2. *Recognizer, free readings:* where readings of held-out words and the rule labels disagree in the same
   spelling context across many words (`chinukpipa/htr/contexts.py` on `results/sweep5/base_v*.json`; each run
   holds out a different random 20% of the words, so a word can be counted more than once). These counts were made
   with the rules in force before the revision; re-running `contexts.py` with the revised rules gives different
   counts, because readings whose stored rule tokens no longer match the rules are skipped.
3. *Recognizer, rule tests:* `chinukpipa/htr/hypotheses.py` scores the rule's token sequence and an alternative
   for each held-out word with models that never saw it (the "cal" configuration, three random seeds per
   split, five splits; output in `results/hypotheses_rules_v1.json`). The models were trained with the old
   labels, which biases them toward the old rules.
   "Control" alternatives that delete one sign the rules have no reason to doubt win 9-17% of the time; that is
   the base rate a deletion has to beat.
4. *Recognizer, rule tests repeated with the revised labels* (2026-10-08): the same test with models trained on
   the revised labels (three random seeds per split, five splits; output in `results/hypotheses_rules_v2.json`).
   Here the old rules are the alternatives. These models are biased toward the revised rules, so this is a
   consistency check; the test in 3, whose models were biased the other way, is the stronger evidence. Controls
   won 8-15% of the time (dropping the first A: 21 of 257; the first O: 14 of 95; an *i* with no vowel beside it:
   12 of 79).

## Revised rules (provisional, 2026-10-08)

| rule | old | new | rows relabelled |
|---|---|---|---|
| *z* | TS | S | 17 |
| *tsh*, *tch* | TS + H, T + CH | CH | 11 |
| *iu*, *yu*, *yoo* | E + U / E + OO | U | 6 |
| *aw* before a consonant or at the end of a word | A + OO | OW | 1 |

(Rows relabelled: in `vocab_annotations.jsonl` and `copy_annotations.jsonl` together.)

**z → S.**
- Sources: the *z = ts* entries are in Le Jeune's tables for English (1896 p. 6, 1898 p. 2). Robertson (2011,
  p. 56 n. 62) describes shorthand *ts* for English [z] as a convention of Le Jeune's literary norm, and
  normalizes the Chinook words with *s* (*tanas*, *spos*, *aias*).
- Images viewed: *Aïaz*, *Tanaz*, *Noz* (1892), *tanaz'*, *tamano'az*, *snaz* (1898) end in a plain curve like
  the fonts' S, without the dot of the TS sign (U+1BC25 DUPLOYAN LETTER S WITH DOT).
- Recognizer: free readings had S at 8 of the 12 held-out final-*z* positions labelled TS. In the rule test, TS
  beat S in 0 of 12 (and again in 0 of 12 with the revised-label models).
- In the ground truth *z* occurs only at the end of a word; *z* elsewhere is untested.

**tsh, tch → CH.**
- Sources: Le Jeune's tables give one sign for *ch* / *j*, an arch with a dot inside (1896 p. 6; 1898 p. 2).
  Robertson normalizes these words with one affricate, t͡ʃ (*t͡ʃok* 'water', *klut͡ʃmin* 'woman',
  *patlat͡ʃ*, *nanit͡ʃ*).
- Images viewed: *Tshok*, *Tsher*, *Tshiken*, *Patlatsh*, *Nanitsh*, *Klootshmen* (1892) and *klootchmin* (1898)
  show an arch with a dot and no separate H dot.
- Recognizer rule test: dropping the H of *tsh* won in 7 of 8 held-out occurrences (4 words). Replacing TS + H by
  CH won in 5 of 8; the models cannot separate the two dotted curves reliably with this little data. With the
  revised-label models, the old TS + H reading won in 0 of 8.

**iu, yu, yoo → U.**
- Sources: the U sign is labelled "u, as in use" (1898 p. 6). Robertson writes /ju/ with one letter (*aj͡u*,
  *kj͡utan*, *stj͡uil*).
- Images viewed: *a'yoo* (1898), *Ayoo* and *Kiutan* (1892) show no separate E hook.
- Recognizer rule test: U alone won in 4 of 4 held-out occurrences (2 words). With the revised-label models, the
  old readings won in 0 of 4 (*iu*, *yu*) and 0 of 2 (*yoo*).

**aw → OW** (before a consonant or at the end of a word).
- Sources and images viewed: *Khaw* (1892) and *kow* (1898) are written with a circle with a central dot, the OW
  sign. Robertson writes a͡w as one letter (*k'a͡w* 'to tie', p. 44).
- One ground-truth word is affected; not tested by the recognizer.

## Open questions

- **An *i* next to another vowel** (*ai*, *oi*, *ia* …: *Aïlan*, *Maika*, *Heloïma*, *Lapiosh*). Le Jeune's
  tables show it written with the E hook beside the other vowel (1896 p. 9; 1898 pp. 5-6), so the rules keep E.
  The recognizer often misses it: in the rule test, dropping that E won in 44 of 81 held-out occurrences (53
  words), against 9-17% for the controls. With the revised-label models it won in 35 of 75 (49 words; 14 of 29
  before a vowel, 21 of 46 after one), against 8-15% for the controls; the median difference in log-likelihood between the
  two readings is small (0.9 in favour of keeping E, against 8-11 for the controls). It may be a weakness of the
  recognizer (a small hook joined to a circle) or a rule error; it is not settled.
- ***stiw'ilh*** may contain the U sign (Robertson writes *stj͡uil*); one word, low confidence.

Corrections and checks of individual words are welcome as issues.
