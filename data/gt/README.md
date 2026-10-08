# Ground truth: Chinuk Pipa word images with readings

## vocab_annotations.jsonl (v0.3, October 2026)

**What it is.** 486 rows from two word lists produced at Kamloops. Each row gives a shorthand word and, printed
beside it, the same word in Roman letters and an English gloss:

| source | item (Internet Archive) | pages (index in `<item>_orig_jp2.tar`) | rows |
|---|---|---|---|
| `LJ1898` | *Chinook and Shorthand Rudiments* (Kamloops, 1898), "Chinook Vocabulary": `cihm_15465` | 11, 12, 14 | 192 |
| `LJ1892` | *Chinook Vocabulary, Chinook–English* (title page: "From the Original of Rt. Rev. Bishop Durieu O.M.I. With the Chinook Words in Phonography By J.M.R. Le Jeune O.M.I. Second Edition. Mimeographed at Kamloops.", dated October 1892): `cihm_15474` | 7–22 | 294 |

By language: 426 Chinook Jargon rows (`chn`; 332 distinct spellings when case, accents and apostrophes are
ignored, 94 of them in both lists), 32 French
loanwords printed in French spelling (`fr`), and 28 English loanwords printed in English spelling (`en`). Rows
whose crop was damaged or overlapped a neighbouring line were left out. On pages 19 and 21 of the 1892 scan the
Roman-word column is hidden in the binding; for those rows the Roman word was read from a second copy (below).

**Fields.**
| field | meaning |
|---|---|
| `id` | row identifier |
| `frame` | how to reproduce the page frame the boxes refer to: `{"deskew_deg": a}` (rotation), `{"affine": [6 numbers], "size": [w, h], "note": ...}` (PIL affine transform) or `{"perspective": [8 numbers], "size": [w, h], "note": ...}` (PIL perspective transform); `note` records how it was derived |
| `crop_clean` | if `true`, ink touching the border of the shorthand crop (pieces of neighbouring lines) is whitened when the crop is cut |
| `bbox_shorthand`, `bbox_latin`, `bbox_gloss` | `[x0, y0, x1, y1]` in that frame's pixels |
| `latin`, `gloss_en` | the Roman spelling and English gloss as printed |
| `tokens_rule` | shorthand sign tokens predicted from `latin` by the rules in `chinukpipa/translit.py`. This is a hypothesis, not a reading of the image. It is `null` for the 60 French- or English-spelled rows and the 1 low-confidence row. |
| `tokens_verified` | tokens checked sign by sign against the image (`null` = not done yet) |
| `reading_confidence` | confidence in the Roman/gloss reading: high / med / low |
| `reviewed_by`, `notes` | who reviewed the row, and remarks |

**How it was made.** Pages were segmented by code in `chinukpipa/gt/`. The Roman spellings and glosses were then
read from the image crops by AI models (Claude), with OCR used only as a hint:
- **1898 list:** for 181 rows, two models (Claude Opus and Claude Sonnet) read each one separately. They agreed
  on 180. The one disagreement was settled using Le Jeune's own spelling of the same word in his other books.
  The other 11 rows were read by Sonnet and checked by Opus.
- **1892 list, pages 7–18 (v0.1):** one model (Sonnet) read it, and the other (Opus) checked each row against the image.
- **1892 list, additions on page 18 and pages 20 and 22 (v0.2, 41 rows):** rows were segmented by
  `chinukpipa/gt/rule_columns.py`, which uses the dotted rules between the columns (on pages 20 and 22 their
  positions were given by hand: `--rules 20:1180:1470,22:1250:1600`). Opus read the rows and Sonnet read them
  separately. The Roman spellings were identical for 39 of 41 rows; the two differences (a
  final letter cut off by the box, and word spacing) were settled on a wider crop. The glosses matched apart
  from punctuation, except where a gloss crop was cut off or showed a neighbouring line; those glosses were
  read from the page image, as noted in `notes`.
- **1892 list, pages 19 and 21 (v0.3, 21 rows):** the Roman words were read from the Newberry Library copy (see
  below), whose page was registered to this scan; the shorthand boxes were then fitted to the ink of this scan.
  Opus read the rows and Sonnet read them separately: the Roman spellings were identical for 20 of 21 rows
  apart from punctuation, and the one difference (a possible diacritic) was settled on an enlarged crop.

**No human has reviewed these readings yet.** Agreement between two AI models is a consistency check, not a
verification.

**Images are not included.** To regenerate them:
```
pip install requests pyyaml pillow numpy scipy scikit-image
python -m chinukpipa.corpus.internet_archive download --ids cihm_15465 cihm_15474 --dest corpus/lejeune --kinds orig_jp2
python -m chinukpipa.gt.make_crops data/gt/vocab_annotations.jsonl corpus/lejeune gt_crops/
```
In testing, the 1,395 crops of the v0.1 and v0.2 rows regenerated pixel for pixel identical to the crops the
readings were made from. (For the v0.3 rows the Roman words were read on the second copy; see below.)

## rudiments1924_annotations.jsonl (v0.4, October 2026)

**What it is.** 475 rows from the vocabulary pages of Le Jeune's *Chinook Rudiments* (36 pages; the Internet Archive
record gives the year 1924 and no place of publication), in the Newberry Library's copy on the Internet Archive
(`Ayer_PM843_L45_1924`). The layout is that of the 1898 list:
a shorthand word, its Roman spelling and an English gloss, all lettered by hand. Source code `LJ1924`.

| pages (index in `<item>_orig_jp2.tar`; printed page = index − 4) | rows |
|---|---|
| 10, 11, 13, 15, 16, 17, 18, 19, 21, 22 | 64, 38, 33, 34, 76, 67, 59, 42, 26, 36 |

By language: 415 Chinook Jargon rows (`chn`), 54 French loanwords in French spelling (`fr`) and 6 English (`en`);
`tokens_rule` is `null` for the 60 French and English rows. Not used: indices 12 and 20 (shorthand beside English
words, with no Roman Chinook spelling), the running exercise text (indices 29–40), entries wrapped over two lines,
headings, and three entries whose shorthand touches a neighbour or whose reading was too uncertain.

The book was photographed turned on its side, so every row has an `affine` frame that turns its page upright (the
same form as the 1892 rows), and `crop_clean` is set.

**How it was made.** Boxes were found by a version of the row segmenter in `chinukpipa/gt/` adapted to ragged
hand-lettered rows, then checked on contact sheets and page overlays; 28 were corrected by hand (each has a note).
The boxes are stored, so `make_crops` regenerates the crops without that step. Readings, all by AI models:
- Claude Sonnet read the Roman spelling and gloss of each row from the crops.
- Claude Opus then read the Roman words again without seeing the first reading. The two readings were identical,
  character for character, for 384 of the 475 rows. For the 415 Chinook rows they implied the same sign tokens in
  411; the other 4 (word spacing in three, a crossed *t* in one) were settled on enlarged crops in favour of the
  first reading. Where the second reading differs (mostly capitals, accents and apostrophes) it is given in `notes`.
- Glosses were read once.

**No human has reviewed these readings yet.**

**Relation to the 1898 list.** About 270 rows (about 250 of them Chinook) share a Roman spelling with an 1898 row
when case, accents and apostrophes are ignored, and about 40 of these have a differently worded gloss (the exact
counts depend on how spellings and glosses are compared). The shorthand appears to have been written out again
rather than reprinted: on the 250
same-word pairs, an overlay of the blurred ink of the 1924 and 1898 images correlates at a median of 0.43, against 0.64 for
the same word written twice within the 1924 book and 0.04 for different words (the heavier strokes and looser boxes of the
1898 scan may lower these scores). A few simple words look nearly identical in both books.

**Images are not included.** To regenerate them:
```
python -m chinukpipa.corpus.internet_archive download --ids Ayer_PM843_L45_1924 --dest corpus/lejeune --kinds orig_jp2 scandata
python -m chinukpipa.gt.make_crops data/gt/rudiments1924_annotations.jsonl corpus/lejeune gt_crops/
```
In testing this gave 1,425 crops (shorthand, Roman word and gloss for each row); spot checks matched, pixel for
pixel, the crops the readings were made from.

**Rights.** The Internet Archive record gives the Newberry's terms: available "for any lawful purpose, commercial
or non-commercial, without licensing or permission fees to the library, subject to these terms and conditions".
The book is dated 1924, so it is in the public domain in the United States. The scans are not
redistributed here; the annotations are CC BY 4.0.

## copy_annotations.jsonl: the same words in a second copy of the 1892 list

The 1892 list was mimeographed, so its copies should carry the same handwriting, each inked and aged
differently; the second copy bears this out (its pages register onto the first scan, and side-by-side crops
viewed by the AI match). The Internet Archive also holds the Newberry Library's copy (`Ayer_PM848_L4_1892`; the Internet Archive record
gives the Newberry's terms: available "for any lawful purpose, commercial or non-commercial, without licensing
or permission fees to the library, subject to these terms and conditions"). Its pages 5–7 and 8–18 correspond
to pages 7–9 and 12–22 of `cihm_15474`; pages 10 and 11 of `cihm_15474` have no counterpart in it.

`copy_annotations.jsonl` (245 rows) was made with `chinukpipa/gt/register_copy.py`: each annotated page was
matched to the Newberry page (ORB keypoints, RANSAC homography, recorded as a `perspective` frame) and each
row's boxes were refined by template matching. Rows keep the reading of the row named in `copy_of`; rows with a
weak match (correlation below 0.5) were dropped. These rows are extra images of the same words, not new words.
The recognizer uses them for training only; test scores are computed on the `vocab_annotations.jsonl` rows.

    pip install opencv-python-headless          # register_copy only; make_crops does not need it
    python -m chinukpipa.corpus.internet_archive download --ids Ayer_PM848_L4_1892 --dest corpus/lejeune --kinds orig_jp2
    python -m chinukpipa.gt.make_crops data/gt/copy_annotations.jsonl corpus/lejeune gt_crops/

**Scans used.** Internet Archive copies of the CIHM microfilm (`cihm_15465`, `cihm_15474`). The Internet
Archive records list the contributor as Canadiana.org and the sponsor as University of Alberta Libraries. The
second copy of the 1892 list is the Newberry Library's (`Ayer_PM848_L4_1892`). The scans are not redistributed
here.

**Licence.** Annotations CC BY 4.0. The underlying texts were published in 1892 and 1898, so they are in the public domain in the United States.
