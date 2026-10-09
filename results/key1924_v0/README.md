# Second running-text test: the 1924 exercises (version 0)

**What it is.** Machine reading of the 823 Chinook words with rule tokens in
`data/gt/rudiments1924_text_annotations.jsonl` (narrative text from the exercises and key of the 1924 *Chinook
Rudiments*, printed one word per line with the Roman spelling beside each outline). The word boxes are the ground
truth's own boxes, so this measures reading, not segmentation. The rows were not used for training.

**How it was run.**
```
python -m chinukpipa.corpus.internet_archive download --ids Ayer_PM843_L45_1924 --dest corpus/lejeune --kinds orig_jp2 scandata
python scripts/build_lj_wordlist.py --sources LJ1892 LJ1898 --out lex_no1924.tsv
python scripts/build_lj_wordlist.py --sources LJ1892 LJ1898 LJ1924 --out lex_all.tsv \
    --gt data/gt/vocab_annotations.jsonl data/gt/copy_annotations.jsonl data/gt/rudiments1924_annotations.jsonl
python -m chinukpipa.gt.make_crops data/gt/rudiments1924_text_annotations.jsonl corpus/lejeune gt_crops/
python -m chinukpipa.text.gtrows pack data/gt/rudiments1924_text_annotations.jsonl gt_crops pages --book key1924
python -m chinukpipa.text.readcrops --pages pages --models <five models> --lexicon <word list> --out-dir read
python -m chinukpipa.text.gtrows score data/gt/rudiments1924_text_annotations.jsonl pages read --book key1924 \
    --lexicon <word list> --out score.json
```
The scale was chosen by `readcrops`' own rule without labels (0.8 for this book). The crops are those of
`make_crops` (cut from the page with a margin and border cleaning), not the masked crops of stage A; the free
reading is the first model's.

| file | models | word list | first choice | top five | first choice, words in the list | free reading: token error rate |
|---|---|---|---|---|---|---|
| `score_unseen_book.json` | five models never trained on the 1924 book (`synth3` + batch 128 configuration, all 1892/1898 words, 20 epochs, seeds 0–4) | `lexicon_merged.tsv` + Le Jeune's 1892 and 1898 spellings (`scripts/build_lj_wordlist.py --sources LJ1892 LJ1898`) | 637 of 823 (77%) | 83% | 637 of 693 (92%) | 0.17 |
| `score_book_vocabulary_known.json` | five models trained also on the 475 vocabulary rows of the same book (`synth3`, batch 64) | + the 1924 vocabulary spellings | 685 of 823 (83%) | 86% | 685 of 717 (96%) | 0.08 |

**Reading these numbers.**
- Most errors are words that are not in the word list under the spelling Le Jeune used here: within the list,
  92–96% are read right.
- The 823 outlines (64 of them are two words on one outline: 52 written with a space, 12 hyphenated) are 261
  distinct token strings; 624 of the 823 outlines (125 of the 261 strings) share a token string with the 1892 and
  1898 training words, so the first row is "a book not seen in training" but mostly words that were. `lexicon_merged.tsv` itself contains headwords parsed from a 1924
  edition of the *Rudiments* (see its sources), so even the first word list is not fully independent of this book.
- The two rows differ in batch size as well as in training data.
- The 35 outlines containing the abbreviation S.T. are not scored here (no rule tokens); on the Creation page they
  count as misses.
- The labels are AI readings checked by a second AI model, with no human review; token labels follow the
  spelling rules, which are hypotheses.
- The model weights are not included.
