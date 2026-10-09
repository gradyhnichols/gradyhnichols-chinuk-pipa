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
without two clear modes). Small marks (dots, vowel circles)
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
equal. See the module docstring.

## What the outputs contain

Names use `<stem>` = `<item>_<leaf>`; in stages B and C the item is called the book.

| File | Stage | Contents |
|---|---|---|
| `<stem>_seg.json` | A | Page statistics (stroke width, typical height, line spacing, gap statistics, rules, frames), the lines and their word boxes with kind (`word`, `roman`, `punct`) and the components in each, and the role of every component (text, frame, rule, brace, ...). Coordinates are pixels of the deskewed page. |
| `<stem>_overlay.png` | A | The page with word boxes, line numbers and set-aside furniture drawn on it, for checking the segmentation by eye. |
| `<stem>_words.npz` | A | `flat` (all crops as one uint8 array), `shapes` (height, width of each), `boxes` (x0, y0, x1, y1), `line` and `index` (the word's line number and position in the line; the key of a box is "line.index"). |
| `<book>_calibration.json` | B | The scale table and the chosen scale. |
| `<stem>_read.json` | B | `item`, `leaf`, `scale`, `models`, `n_candidates`, a `reading` note and `words`: for each word box `line`, `index`, `bbox`, `free_tokens`, `free_roman`, `free_models_agree`, `confidence` and `confidence_nonblank` (mean per-frame maximum class probability, over all frames and over non-blank frames), `frames`, `top5` (each with `headword`, `also`, `tokens`, `score`, `p_rel`) and `free_best` (`tokens`, `roman`, `score`). |
| `<book>_book.json` | B | A summary over the book's pages read so far: how often the models agree, how often the free reading equals the list's first choice, quantiles of `p_rel` and `confidence`, the most frequent first choices. |
| `<stem>_merged.json` | C | Like `_read.json`, with the words after joining: each has `boxes` (the box keys it covers, e.g. `["5.2", "5.3"]`), the union `bbox`, and the same reading fields. Also `merge_rule`, `gap_max_px` and `joins`. |

`read_page` writes, per page, `_read.json` (words with `crop` file names), the crops as PNG files and review
sheets (crop, free reading, top three candidates) next to the segmentation files.

## Limits

- Every reading is a machine reading. No person has checked them, and the stage B and C files carry a note saying
  so. A reading that agrees with a word-list entry is not thereby correct.
- Segmentation errors are the main loss. A word cut in two or two words in one box cannot be read right by a
  recognizer that reads one box at a time, whatever the model does; stage C repairs only some splits and can join
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
