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
come from two word lists; text pages (sentences, prayers, the newspaper) have not been tried.

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
