# Word recognizer (experimental)

A small recognizer that reads one Chinuk Pipa word image as a sequence of sign tokens (the token set in
`data/signs/tokens.yaml`). It is an early experiment, built to find out what is possible with the little data
that exists, and to help check the spelling-to-sign rules. It is not a working OCR system.

## How it works

- **Model** (`model.py`): a convolutional network followed by a bidirectional LSTM, trained with CTC loss
  (a "CRNN", a standard design for reading lines and words).
- **Synthetic words** (`chinukpipa/synth.py`): Chinook words from the merged word list and random syllables,
  turned into tokens by the transliterator, drawn with Duployan fonts and roughened (slant, wobble, blur, noise).
  Renderings of the held-out test words are removed before training.
- **Real words** (`data.py`): the ground-truth crops in `data/gt/`, labelled with `tokens_rule`, i.e. with the
  tokens the spelling rules predict, not with checked readings. Each word image is scaled so that its strokes
  are about 3 pixels wide and its paper is white.
- **Training** (`train.py`): batches mix synthetic and real words (40% real in the runs below).
- **Evaluation** (`evaluate.py`): on real words held out *by word* (a word in the test set never appears in
  training, as a real image or as a synthetic one). Two readings are scored:
  - *free reading*: the model's own token sequence, scored as token error rate (TER: edits needed to turn the
    reading into the rule tokens, divided by the number of rule tokens) and as exact-word accuracy;
  - *word-list reading*: every candidate from the word list (about 1,600 words, including the test words) is
    scored by how well it fits the image, and the best is taken. This measures "pick the right word from a
    known vocabulary", which is easier than open reading.

## Results so far

### With the revised spelling rules (October 8, 2026)

The rule revisions in `docs/rule_notes.md` changed the labels of 35 ground-truth rows, so these numbers are not
comparable with the older table below. Five splits (`--split-seed v0` … `v4`), 472 held-out words in all (85–106
per split), one model per split (random seed 0) unless noted; word-list scores with test-time rescaling as before.
The base configuration is the "rescaling" row of the older table (160,000 synthetic words, 40% real words,
`--real-scales LJ1892=0.5,LJ1898=1.0 --scale-aug 0.3`), with batch 64, 1,000 steps per epoch and 15 epochs.
Mean ± standard deviation over the five splits (population standard deviation; the older table below uses the
sample standard deviation; per-word readings in `results/cv_rules_v2/`):

| configuration | free reading: TER | free reading: exact word | word list: right word first | word list: in top 5 |
|---|---|---|---|---|
| base | 0.29 ± 0.02 | 30% ± 4 | 68% ± 4 | 87% ± 4 |
| base, three models combined (seeds 0–2) | | | 72% ± 3 | 88% ± 3 |
| base + 160,000 more synthetic words (`synth3`, below) | 0.27 ± 0.01 | 33% ± 2 | 72% ± 4 | 88% ± 3 |
| same, three models combined | | | 73% ± 4 | 90% ± 2 |
| base with batch 128 (`--bs 128 --lr 2.8e-3 --steps-per-epoch 500`) | 0.28 ± 0.03 | 31% ± 5 | 72% ± 4 | 89% ± 2 |
| `synth3` and batch 128 | 0.29 ± 0.03 | 30% ± 5 | 68% ± 4 | 87% ± 4 |
| `synth3` and batch 128, plus the 1924 rows (ground truth v0.4) | 0.27 ± 0.02 | 33% ± 5 | 71% ± 4 | 90% ± 2 |
| `synth3`, plus the 1924 rows | 0.28 ± 0.02 | 33% ± 3 | 70% ± 4 | 89% ± 3 |

(For combined models the free reading is the first model's, so it is not repeated.)

- `synth3` is 16 × 10,000 words drawn with the revised rules, in the style of the second set below
  (`--engine hb --jitter 1.0 --seed 300` … `315`). It lowered the token error rate in all five splits (by
  0.015–0.034). Synthetic labels are fixed when the words are drawn: the base sets were drawn with the rules as
  they stood before the revision, `synth3` with the revised rules, so part of this gain may come from labels that
  agree with the real words' labels rather than from more data.
- The other differences are within the noise of these small test sets: with one model per split, the per-split
  differences in first choice between two configurations range from −10 to +9 points. Batch 128 trains about
  30% faster per word on the GPU used here (an 8 GB laptop GPU).
- Adding the 475 rows of the 1924 *Rudiments* (`data/gt/rudiments1924_annotations.jsonl`, scale 0.63 for that
  source) mainly helps on words from that book: on its held-out words, word-list first choice rose from 75% to 86%
  (batch-128 configuration, 440 words over five splits) and from 80% to 86% (batch-64 configuration; means over
  the five splits), with the token error rate falling from about 0.26 to 0.16 in both. On the 472 older test words its effect was small and not
  consistent: +2 points with batch 128, −2 with batch 64 (in four of five splits).

**A book the recognizer had not seen.** Three models trained on all the 1892 and 1898 words (base configuration,
20 epochs) read all 415 Chinook words of the 1924 *Rudiments*, a later edition (image comparisons suggest its
shorthand was written out anew; see `data/gt/README.md`): word-list first choice 85%, top five 96%, token error rate 0.15. For the 316 words whose token strings were
among the training words (from the other two books) the first choice was right for 93%; for the 99 words new to
the recognizer, for 60% (top five 86%, token error rate 0.34). The scale for this book (0.63) was chosen with
`calibrate.py` on the 1924 words of a training split, using a model that had not seen the book; as everywhere
here, the candidate list contains the test words.

### Before the rule revision

Five runs, each holding out a different random 20% of the words (`--split-seed v0` … `v4`; 85–106 test words
per run), ground truth v0.3, with the spelling rules as they stood before the *z* revision in
`docs/rule_notes.md`. Training used 160,000 synthetic images (8 × 10,000 with `--seed 100`…`107`, and
8 × 10,000 with `--engine hb --jitter 1.0 --seed 200`…`207`, drawn with Noto Sans Duployan and Rawnd Musmus
Duployan) and the real training words of each split, including the second-copy images. The per-word readings
are in `results/sweep5/` and `results/sweep6/`. Mean ± standard deviation over the five runs:

| configuration | free reading: TER | free reading: exact word | word list: right word first | word list: in top 5 |
|---|---|---|---|---|
| synthetic + 40% real, 15 epochs | 0.32 ± 0.02 | 24% ± 4 | 60% ± 5 | 83% ± 4 |
| same, plus GPU warping (`--gpu-aug 0.5`) | 0.36 ± 0.01 | 19% ± 5 | 54% ± 4 | 78% ± 3 |
| same as the first, plus rescaling (`--real-scales LJ1892=0.5,LJ1898=1.0 --scale-aug 0.3`) | 0.28 ± 0.01 | 29% ± 6 | 62% ± 6 | 85% ± 3 |

Words from the 1898 book were read better than words from the 1892 list (word list, first choice: 65% against
57% for the plain configuration). The word-list scores use test-time rescaling (`--tta 0.85,1,1.2`).

The rescaling row shrinks the 1892 words to about the size of the synthetic words (after stroke-width
normalization they come out roughly twice as large; the factor 0.5 was chosen with `calibrate.py` on the
training words of split `v0`, using an earlier model trained on synthetic words only) and rescales every training
word by a random factor between 1/1.3 and 1.3. It lowered the token error rate in all five
splits; its effect on the word-list scores was smaller and not consistent from split to split. On one split
(`v0`), combining three models trained with different random seeds (`evaluate.py` with three model files)
raised the word-list first choice from 53% to 64% (`results/sweep6/ens3_v0.json`, plain configuration).

GPU warping did not help once real words were part of training.

**Limits.** The test sets are small (85–106 words per split), so differences of a few points are within noise.
The labels are rule predictions, so a "wrong" reading may be a rule error rather than a model error. All words
come from word lists; for running text see [`chinukpipa/text/`](../text/README.md).

## Running

```bash
pip install torch pillow numpy scipy scikit-image pyyaml uharfbuzz fonttools
# fonts are not included: put Duployan fonts (e.g. Noto Sans Duployan, Rawnd Musmus Duployan; both SIL OFL)
# in fonts/ first
python -m chinukpipa.synth synth/part0 --n 10000 --engine hb
python -m chinukpipa.gt.make_crops data/gt/vocab_annotations.jsonl corpus/lejeune gt_crops/
python -m chinukpipa.gt.make_crops data/gt/copy_annotations.jsonl corpus/lejeune gt_crops/
python -m chinukpipa.htr.train --synth synth/part0 --crops gt_crops --out runs/x --real-train-weight 0.4 --epochs 15
python -m chinukpipa.htr.evaluate runs/x/model.pt --crops gt_crops --out runs/x/eval.json
python -m chinukpipa.htr.confusions runs/*/eval.json        # where readings and rules disagree
```

With the 1924 rows (ground truth v0.4) and the faster input pipeline, as in the last row of the first table:

```bash
python -m chinukpipa.synth synth3/part0 --n 10000 --engine hb --jitter 1.0 --seed 300   # ... part15, seed 315
python -m chinukpipa.gt.make_crops data/gt/rudiments1924_annotations.jsonl corpus/lejeune gt_crops/
python -m chinukpipa.htr.train --synth "synth*/part*" --crops gt_crops --out runs/y \
    --real data/gt/vocab_annotations.jsonl data/gt/copy_annotations.jsonl data/gt/rudiments1924_annotations.jsonl \
    --real-scales LJ1892=0.5,LJ1898=1.0,LJ1924=0.63 --scale-aug 0.3 --real-train-weight 0.4 \
    --bs 64 --lr 2e-3 --epochs 15 --amp --workers 2 --bucket 64      # --all-real: train on every real word
python -m chinukpipa.htr.bench <the same train arguments>    # where the training time goes
```
