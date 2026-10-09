# Running text (experimental)

Tools that take scanned pages of running Chinuk Pipa text, cut them into lines and word images, and read each
word image with the word recognizer of [`chinukpipa/htr/`](../htr/README.md). The output is a machine reading of
every word box, with the reading's scores, so that it can be checked, scored and improved. It is an early experiment,
not a transcription system.

## What the pipeline does

Three stages. Each writes files that the next one reads, and each can be restarted: stages A and B skip pages
whose output files already exist.

**Stage A: segment pages and save word images** (`segcrops`, built on `segment` and `pagesrc`; CPU, no PyTorch).
A page of an Internet Archive item is loaded upright (raw captures are rotated and cropped with the item's
`scandata.xml`; processed derivatives are cut to the sheet of paper), binarized, and deskewed. Long rules, dotted
page frames, stanza separators and braces are set aside; page numbers and, in the hymn-book layout, headings are
labelled and left out. Text lines are found per column, and the dark pieces of each line are chained from left to
right into words. A gap starts a new word if it is wider than a threshold taken from the valley between the two
modes of the page's own gap sizes (pen lifts inside words, spaces between words; there are fallbacks for pages
without two clear modes). Since v0.4 a valley with fewer than a quarter of the page's gaps above it is rejected and
Otsu's threshold on the same gaps is used instead (the quarter was set by hand from the spread of that share over
the pages). This happened on 48 of the 478 pages read so far; on those looked at, the valley had fallen between the
pen lifts and a few very wide gaps (between columns, around a picture), and whole lines had become one word box. An
AI check of 10 of them, blind to which version was new, judged the new boxes better on 9 (one tie); some dense or
faint pages are still cut badly. Small marks (dots, vowel circles)
join the nearest word; punctuation and runs of Roman letters get their own labels and are not read. The other
thresholds are fixed multiples of quantities measured on the page (the stroke width, the line spacing and the
height of a typical word piece); the multipliers were set by hand while developing on pages of these books. On a
page where no column rule or gutter is found, a text line runs across all columns. For every word box, one grey crop is cut (the word's own ink, with ink of neighbouring words
whitened) and all crops of a page are saved in one `.npz` file.

**Stage B: read every word image** (`readcrops`; GPU or CPU). Needs one to several trained recognizer models
(`model.pt` from `chinukpipa.htr.train`; with several, they are read as an ensemble).

- *Scale calibration, per book.* The recognizer was trained with a per-book scale factor applied after stroke-width
  normalization (see the `--real-scales` option in the recognizer's README). For a new book the factor is not known.
  The rule used here: try each scale of 0.4, 0.5, 0.63, 0.8, 1.0 and 1.25 on a random sample of the book's word
  images (seed 0, up to `--calib-words`, default 600) and keep the scale with the highest mean per-frame maximum
  class probability (averaged over frames, words and models). It is a heuristic for choosing a scale without any
  labels; it is not checked against correct readings. The table is saved in `<book>_calibration.json`, and later
  runs reuse it unless `--force` is given. `--scale S` sets the scale by hand.
- *Free reading.* Greedy CTC decoding of the first model's output: a sequence of sign tokens, and the same as Roman
  letters (the first Latin spelling listed for each token in `data/signs/tokens.yaml`).
- *Word-list reading.* Every candidate word of the word list (the A/B-grade headwords of
  `data/lexicon/lexicon_merged.tsv`, turned into tokens by the project's spelling rules; about 1,600 distinct token
  strings) is scored by the CTC loss (negative log-likelihood) of its tokens given the image, averaged over the
  models. The five lowest are kept. A candidate that is too long for the image gets no score and ranks last.
  `p_rel` is the softmax of the negative scores over all candidates: it says how much the best candidate stands out
  among the candidates, not how likely it is to be right.
- *Free best.* Of the models' own free readings, the one with the lowest ensemble loss. Its score can be compared
  with the word-list scores, for example to tell when a word is probably not in the list.
- Words go through the network in batches of similar width, and each word is read as if it were alone (the models
  were trained on batches of similar width and read badly through long padding). `--selftest` checks this against
  reading one word at a time.

**Stage C: join word pieces** (`remerge`; CPU is enough). The segmenter sometimes cuts one word in two, when its own
signs are spaced a little wider than usual. Walking each text line from left to right, the current group of boxes
and the next box are joined when (1) the gap between them is smaller than the page's typical between-word gap
(`gap_stats.mode_large_px` of the page's segmentation, a page statistic, not tuned; on a page without that
statistic, 2.2 times the page's word-gap threshold) and (2) the joined image reads
better than the pieces: the best word-list loss of the joined image is lower than the sum of the pieces' best
losses. Groups have at most `--max-group` boxes (default 3). The joined image is rebuilt from the stage-A crops
(the darkest pixel of overlapping crops wins) and read again with the same models, lexicon and scale as in stage B.

## Running it

```bash
pip install -e ".[text,net]"      # or: pip install torch opencv-python-headless pillow numpy scipy scikit-image pyyaml requests

# page images: an Internet Archive item folder with <id>_orig_jp2.tar (+ <id>_scandata.xml) or <id>_jp2.zip
python -m chinukpipa.corpus.internet_archive download --ids ITEM --dest corpus/lejeune --kinds orig_jp2 scandata

# A: segment pages (leaf numbers are 0-based positions in the item's sorted image list)
python -m chinukpipa.text.segcrops corpus/lejeune/ITEM --leaves 15-126,130 --out-dir out/pages   # --shard K/N splits the work

# B: read all word images of the pages found in --pages
python -m chinukpipa.text.readcrops --pages out/pages --models runs/a/model.pt runs/b/model.pt runs/c/model.pt \
    --out-dir out/read --device cuda          # --device cpu without a GPU; --books, --leaves select part of the work

# C: join pieces that read better together (same models and lexicon as in B)
python -m chinukpipa.text.remerge --pages out/pages --read out/read --out-dir out/merged \
    --models runs/a/model.pt runs/b/model.pt runs/c/model.pt
```

All three take `--lexicon TSV` (default `data/lexicon/lexicon_merged.tsv` of this repository). A lexicon that also has
a `tokens` column gives those rows' tokens directly instead of computing them from the spelling; `scripts/build_lj_wordlist.py`
builds such a list by adding the Roman spellings and rule tokens of ground-truth sources:

```bash
python scripts/build_lj_wordlist.py --sources LJ1892 LJ1898 LJ1924 --out out/lex_lj_all.tsv \
    --gt data/gt/vocab_annotations.jsonl data/gt/copy_annotations.jsonl data/gt/rudiments1924_annotations.jsonl
```

Other modules:

```bash
python -m chinukpipa.text.segment corpus/lejeune/ITEM 15 --out-dir out/seg      # one page: _seg.json and an overlay image
python -m chinukpipa.text.read_page corpus/lejeune/ITEM 15 16 --models runs/a/model.pt --out-dir out/rp
                                                              # A and B for a few pages on the CPU, with review sheets
python -m chinukpipa.text.pagesrc corpus/lejeune/ITEM 15 page15.png --max-side 1600   # save one upright page as PNG
python -m chinukpipa.text.dumpcrops out/pages/ITEM_15_words.npz out/crops             # the crops of a page as PNGs
python -m chinukpipa.text.score_alignment ALIGNMENT.tsv out/read/ITEM_15_read.json out/merged/ITEM_15_merged.json
```

`read_page` is the one-step route for a few pages; it ignores a `tokens` column in the word list. `score_alignment`
compares readings with an alignment of the page's Roman text to its word boxes (columns `unit_kind`, `main_box`,
`box_ids`, `roman_tokens`), if one exists for the page, with a strict rule: a Roman word counts as read right only if
it has a word box of its own (or, for a merged file, one output word covers exactly its boxes) and the tokens are
equal. See the module docstring. The options `--pair-penalty`, `--extra-lexicon` (stages B and C), `--st-labels`
(`gtrows score`) and `--merge-rule` (`score_alignment`) are experimental and described under "Two-word decoding and
brief forms" below.

## What the outputs contain

Names use `<stem>` = `<item>_<leaf>`; in stages B and C the item is called the book.

| File | Stage | Contents |
|---|---|---|
| `<stem>_seg.json` | A | Page statistics (stroke width, typical height, line spacing, gap statistics, rules, frames), the lines and their word boxes with kind (`word`, `roman`, `punct`) and the components in each, and the role of every component (text, frame, rule, brace, ...). Coordinates are pixels of the deskewed page. |
| `<stem>_overlay.png` | A | The page with word boxes, line numbers and set-aside furniture drawn on it, for checking the segmentation by eye. |
| `<stem>_words.npz` | A | `flat` (all crops as one uint8 array), `shapes` (height, width of each), `boxes` (x0, y0, x1, y1), `line` and `index` (the word's line number and position in the line; the key of a box is "line.index"). |
| `<book>_calibration.json` | B | The scale table and the chosen scale. |
| `<stem>_read.json` | B | `item`, `leaf`, `scale`, `models`, `n_candidates`, a `reading` note and `words`: for each word box `line`, `index`, `bbox`, `free_tokens`, `free_roman`, `free_models_agree`, `confidence` and `confidence_nonblank` (mean per-frame maximum class probability, over all frames and over non-blank frames), `frames`, `top5` (each with `headword`, `also`, `tokens`, `score`, `p_rel`) and `free_best` (`tokens`, `roman`, `score`). With the experimental `--pair-penalty` there is more: see below. |
| `<book>_book.json` | B | A summary over the book's pages read so far: how often the models agree, how often the free reading equals the list's first choice, quantiles of `p_rel` and `confidence`, the most frequent first choices. |
| `<stem>_merged.json` | C | Like `_read.json`, with the words after joining: each has `boxes` (the box keys it covers, e.g. `["5.2", "5.3"]`), the union `bbox`, and the same reading fields. Also `merge_rule`, `gap_max_px` and `joins`. |

`read_page` writes, per page, `_read.json` (words with `crop` file names), the crops as PNG files and review
sheets (crop, free reading, top three candidates) next to the segmentation files.

## Limits

- Every reading is a machine reading. No person has checked them, and the stage B and C files carry a note saying
  so. A reading that agrees with a word-list entry is not thereby correct.
- Segmentation errors are the main loss. A word cut in two or two words in one box cannot be read right by a
  recognizer that reads one box at a time, whatever the model does (except, experimentally, two words in one box with
  `--pair-penalty`; see below); stage C repairs only some splits and can join
  words that belong apart. Marks such as braces or numerals can be boxed as words, and words can be missed. The
  segmenter handles the layouts it was written for (columns, rules, dotted frames, stanza separators, page numbers,
  headings in the hymn-book layout); other layouts may be cut wrongly. Look at the overlay of each page.
- The word-list reading can only return words that are in the list, and the list's token strings come from
  spelling rules that are themselves hypotheses (see `docs/rule_notes.md`). The free reading has no such limit
  but also no spelling check, and is less often right.
- The recognizer was trained on synthetic words and on word-list entries of two Kamloops books (its README gives the
  numbers and their limits). Running text, other hands and other printings can look different, and the per-book scale
  is only a guess made from the page images.
- The recognizer was trained on Chinook Jargon words only. Pages in other languages (the Salish languages that also
  appear in Le Jeune's books) are outside what it was trained for; as in the rest of this repository, transcriptions
  of Salish-language texts are not published.

## Two-word decoding and brief forms (v0.4, experimental)

Two additions to the word-list reading, both off unless asked for. Without the options below, `readcrops`, `remerge`,
`gtrows score` and `score_alignment` do what they did before and write the same files. They address two kinds of
miss seen in the tests below: an outline that holds two words (64 of the 823 scored rows of the 1924 exercises are
two words on one outline, 52 written with a space and 12 hyphenated; on the Creation page five boxes hold two words
each, and one more pair of words is cut across two boxes), and the abbreviation S.T. (*Sahale Taye*, "God"), which
the word lists do not have.

**Two-word decoding**, in plain words. The word-list reading can only return one list word per image. With
`--pair-penalty`, two-word readings "A _ B" (A and B words of the list, `_` the word-space token) compete with it:

1. Each model's free reading of the image is taken as it is, with its `_` tokens dropped.
2. Every place where it can be cut in two gives a left part and a right part. Each part is looked up in the word
   list: a part of one or two tokens has to equal a list word's tokens, a part of three or four may differ from one
   by one token (an insertion, a deletion or a substitution), a part of five or more by two. The lookup is a table of
   the list's token strings with up to two tokens deleted (the idea of the SymSpell spelling corrector), followed by
   a real edit-distance check of every hit.
3. Each left match and right match together are a proposal "A _ B". At most 400 proposals are kept per image: those
   with the fewest token differences first.
4. A proposal is scored like a list word: the mean over the models of the CTC loss (in nats) of its tokens, as
   `A _ B` or as `A B`, whichever is lower.
5. A list word's *effective score* is its loss; a pair's is its loss plus the penalty. The first choice is the
   lowest effective score, so a pair is chosen only if it fits the image better than the best list word by more than
   the penalty.

The proposals and their ranking are in `twoword.py` (no PyTorch); the scoring is in `readcrops.Ensemble.read_batch`.

```bash
python -m chinukpipa.text.readcrops --pages out/pages --models runs/a/model.pt runs/b/model.pt runs/c/model.pt \
    --out-dir out/read --device cuda --pair-penalty 8
python -m chinukpipa.text.remerge --pages out/pages --read out/read --out-dir out/merged \
    --models runs/a/model.pt runs/b/model.pt runs/c/model.pt --pair-penalty 8
```

- `--pair-penalty NATS` exists in `readcrops` and `remerge` (not in `read_page`). Without it the result is the word-list
  reading described above. `remerge` refuses stage-B files read with another value. Pages that already have a
  `_read.json` are skipped by `readcrops` whatever options they were read with, so use a new `--out-dir` (or `--force`)
  when an option changes; `gtrows score` refuses readings made with different penalties.
- Stage C decides its joins as it does without the option: it compares the best list word's score of the joined
  image with the sum of the pieces' (`best_word_score`), so two-word readings do not change which boxes are joined;
  each unit's reading then includes them. This is the path that was tested on the Creation page (joins on list-word
  scores, then two-word decoding of the units). The `merge_rule` note in the `_merged.json` file says so.
- In `top5`, each entry also has `kind` (`"word"` or `"pair"`) and `loss` (the raw loss); `score` is the effective
  score. A pair's `headword` is "A + B" (the first headword of each part) and its `tokens` are `A _ B`. `p_rel` is
  the softmax of the negative effective scores over all list words and all proposals, so it is not comparable with a
  run without the option. The ranking uses the scores as written (3 decimals), as in the experiments: when a list
  word and a pair show the same score, the list word usually comes first, but the floating-point sum of a pair's loss
  and the penalty can fall a hair below it and put the pair first. Each word has `n_pair_proposals` and `best_word_score` (the best list word's score), and the `_read.json` and
  `_merged.json` files record `pair_penalty` (and `extra_lexicons`).
- A pair is a machine reading like any other: it says that the outline looks like those two words, not that it is
  right.

**Where the penalty 8 comes from.** The 1924 exercises (`data/gt/rudiments1924_text_annotations.jsonl`) were split
by page: pages 25–31 of the scan (printed pages 21–27) to choose the penalty, pages 32–37 (rows exist on 32, 35, 36
and 37) to test it. A short list of values between 0 and 15 was tried on the first group, and 8 gave the most first
choices right (counting the rows that `--st-labels` labels). It was then tested on the second group and on the Creation page. The Creation page had been used to
check several earlier changes, so it is no longer an untouched test. The penalty was tuned on one book; another
book may want another value.

**Brief forms.** `--extra-lexicon TSV` (repeatable; in `readcrops`, `remerge` and `gtrows score`) appends the rows of
another word list to the main one, with the same rules (A/B rows; the `tokens` column, if filled, gives the tokens).
`data/lexicon/brief_forms.tsv` has one row, the abbreviation S.T. with the tokens `S T`; its source is the
entry `ST` of `data/signs/brief_forms.yaml` (Le Jeune's note in the 1898 *Rudiments*, p. 13). Only S.T. has been
tested. The other brief forms in that file are not in the TSV, and each one would need its own test. `gtrows score`
needs the same `--extra-lexicon` as the reading, and stops if the number of candidates differs from the reading's.

**Two scoring options** (defaults unchanged):
- `gtrows score --st-labels` also scores the rows that have no rule tokens, when their Roman spelling gives
  tokens: "S.T." or "S.T" is `S T`, any other word goes through the spelling rules, several words are joined with
  ` _ `, and a row containing "·" stays unscored. The result reports the rows with and without label tokens
  separately. These labels are made when scoring, from the AI-read Roman spelling; no person has checked them, they are
  not written to the ground-truth files and are not published as labels.
- `score_alignment --merge-rule` also scores the units that hold k Roman words (`merge`: one box; `merge+split`:
  several boxes): such a unit is right if
  one output word covers exactly that unit's boxes and its first-choice tokens equal the k words' tokens joined by
  ` _ `, and then each of the k words counts as right. Match and split units are scored as before. A first choice
  can only equal such a token string if it contains a word-space token: a two-word reading, or a phrase that is itself
  in the word list (a headword with a space or a hyphen gets a word-space token from the spelling rules), so the rule
  matters mostly together with `--pair-penalty`.

**Limits.**
- A pair is only proposed when at least one model's free reading comes close enough to both words (step 2), and
  only from words of the list. An error beyond that in either half, or a word that is not in the list, means no
  proposal.
- More work per image: up to 400 more candidates, each scored with and without the word space.
- The Roman spellings behind the `--st-labels` labels are AI readings of the printed book, and the token strings come
  from spelling rules that are hypotheses (see `docs/rule_notes.md`).
- The stage-A rule assumes that at least a quarter of a page's gaps are spaces between words. On a page whose words
  are drawn in many separate strokes (about four pieces per word or more), a true word-gap valley would be rejected
  as well.
- `read_page` has neither option.

**Results.** Every number below was reproduced with this code from the stored models (word list built with the
`build_lj_wordlist.py` command under "Running it": 1,780 candidates, 1,781 with `brief_forms.tsv`). The experiments'
own scripts had measured all of them except the first two rows of the Creation table, and gave the same counts. The
models are not published.

The 1924 exercises, read from their correct word boxes (`gtrows`, scale 0.8), by five models trained on the 1892 and
1898 words and the book's vocabulary rows but not on the exercises (the second row of the table in
[`results/key1924_v0/`](../../results/key1924_v0/README.md)):

| options | rows with rule tokens (823) | with `--st-labels` (858) |
|---|---|---|
| none | 685 | 685 |
| `--extra-lexicon brief_forms.tsv` | 685 | 705 |
| both, `--pair-penalty 8` | 717 | 750 |

On the pages kept back when the penalty was chosen (pages 32–37, 424 rows with `--st-labels`), the two-word readings
took the count from 354 to 383, and no row that was right before became wrong.

The Creation page (stage B + C), read by five models trained like the models above plus 858 rows of the 1924
exercises (the 823 with rule tokens and the 35 S.T. rows, labelled by the `--st-labels` rule):

| options | strict | `--merge-rule` |
|---|---|---|
| none | 156 | 156 |
| `--extra-lexicon brief_forms.tsv` | 167 | 167 |
| both, `--pair-penalty 8` | 167 | 173 |

With both options, three of the six two-word units (five boxes, and one pair cut across two) are read as their two
words, and all 10 S.T. are right. These
models trained on all of the 1924 exercises, so the exercises are no test for them; and the Creation page has now
been used to judge about ten changes, so it is no longer an untouched test either. A second scan of the same page
(Internet Archive `cihm_14939`, leaf 9; its boxes matched to the Roman words by geometry, by an AI agent) was cut
into whole lines by the old stage-A rule (0 of 208 right); the stage-A rule above was written after that, so the
134 it then gets with both options (152 with `--merge-rule`) is not a fresh test.

## Results

**First test: one page with a Roman parallel.** The Creation chapter of the *Chinook Bible History* (Paul Durieu, 1899; Internet Archive
`chinookbiblehist00duri`, leaf 15, left column, paragraphs 1–5) has the same Chinook wording as the "First Lesson
in Chinook" printed in Roman letters in the 1898 *Rudiments* (`cihm_15465`, leaves 15–17). An AI agent aligned
the page's word boxes with the lesson's 208 Roman words (a dynamic-programming alignment on token sequences, checked
by eye on many lines; the Roman text was taken from the OCR and corrected against the page images). The alignment
is in [`results/creation_v0/`](../../results/creation_v0/). No person has checked it.

Scored strictly: a Roman word counts as read right only if it has a word box of its own (after stage C: one output
word covers exactly its boxes) and the reading's tokens equal the tokens of its Roman spelling.

| pipeline | word list: first choice right (of 208) |
|---|---|
| stage B, three models trained on the 1892 and 1898 words, word list `lexicon_merged.tsv` | 105 (50%) |
| same, with Le Jeune's own spellings from the ground truth added to the word list (`scripts/build_lj_wordlist.py`, 1892 and 1924 sources) | 137 (66%) |
| same, plus stage C | 152 (73%) |
| stage C with five models trained also on the 1924 vocabulary rows (`synth3` configuration) and Le Jeune's spellings from all three sources (readings in `results/creation_v0/`) | 153 (74%) |

Where the first run lost its words (one cause per word): segmentation 45 (30 words cut into two or three boxes,
12 words sharing a box with a neighbour, 3 words whose ink was classed as a brace or punctuation and not read),
spelling 37 (the word list had the word only under another
spelling, e.g. *iaka*/*yahka* 12 times, *ookook*/*okook* 8 times), not in the word list 18 (10 of them the
abbreviation S.T., which has no token of its own here), recognizer confusions 3. Of the 108 words that had a clean
box of their own and a spelling in the list, 105 were read right.

Limits of this number:
- It is one page, aligned by AI. On this page the segmenter did not separate the two columns: each text line runs
  across the page, and only the left column's boxes are aligned and scored.
- The 208 words are 67 distinct token strings, so frequent words weigh heavily, and 179 of them have the same
  token string as a word of the 1892 and 1898 lists that the recognizer was trained on (97 of the first run's 105
  right words).
- The added spellings come from Le Jeune's own vocabularies, which cover many of the words of this lesson (the
  *Rudiments* vocabularies of 1898 and 1924 go with its lessons); on other texts the gain may be smaller. The most frequent repairs (*iaka*, *ookook*) are common
  words, though.
- Stage C was written after looking at this page's errors. Its gap limit is a statistic of each page and its
  decision rule has no tuned parameter, but it has not yet been tested on another page. No second passage with a
  Roman parallel was found in the 1898 *Rudiments*; the 1924 exercises below are the second test.
- Taking the free reading (`free_best`) instead of the list's choice when its loss is lower by a margin changed at
  most four words, at a margin chosen on this page; it is not used.

**A second test, from correct word boxes.** The exercises of the 1924 *Rudiments* print narrative text one word
per line, with Le Jeune's Roman spelling beside each outline. Their 823 Chinook outlines with rule tokens
(`data/gt/rudiments1924_text_annotations.jsonl`) were read with `chinukpipa.text.gtrows` and `readcrops`
([`results/key1924_v0/`](../../results/key1924_v0/README.md)): the word list's first choice was right for 77% (five
models never trained on the 1924 book, Le Jeune's 1892 and 1898 spellings in the list) and 83% (models trained also
on the book's vocabulary pages, its spellings in the list; these models also differ in batch size); for words whose
spelling was in the list, 92% and 96%. Most of these words also occur in the training books (624 of the 823 share
a token string with a word of the 1892 and 1898 lists), the 35 outlines containing the abbreviation S.T. are not
scored here (they count as misses on the Creation page), and "right" means agreeing with the rule tokens of Le
Jeune's printed spelling.
Because the boxes are given, this measures reading alone; on the Creation page, word-boundary errors were the
largest single cause of misses (45 of 103 in the first run). The stage-C rule was not tuned on this set, but stage C was not needed for it either.

The pipeline has also been run over 478 pages of running text in 8 Internet Archive items (about 247,000 word
boxes). Those readings have no checked sample yet and are not published.
