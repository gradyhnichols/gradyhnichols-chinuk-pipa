# Ground truth: Chinuk Pipa word images with readings

## vocab_annotations.jsonl (v0.1, October 2026)

**What it is.** 424 rows from two word lists printed at Kamloops. Each row gives a shorthand word and, printed
beside it, the same word in Roman letters and an English gloss:

| source | item (Internet Archive) | pages (index in `<item>_orig_jp2.tar`) | rows |
|---|---|---|---|
| `LJ1898` | *Chinook and Shorthand Rudiments* (Kamloops, 1898), "Chinook Vocabulary": `cihm_15465` | 11, 12, 14 | 192 |
| `LJ1892` | *Chinook Vocabulary, Chinook–English* (Kamloops, 1892; "from the original of Rt. Rev. Bishop Durieu"): `cihm_15474` | 7–18 | 232 |

By language: 371 Chinook Jargon words (`chn`, about 300 distinct words, many appearing in both lists), 32
French loanwords printed in French spelling (`fr`), and 21 English loanwords printed in English spelling
(`en`). Rows whose crop was damaged or overlapped a neighbouring line were left out.

**Fields.**
| field | meaning |
|---|---|
| `id` | row identifier |
| `frame` | how to reproduce the page frame the boxes refer to: `{"deskew_deg": a}` (rotation) or `{"affine": [6 numbers], "size": [w, h], "note": ...}` (PIL affine transform; `note` records how it was derived) |
| `bbox_shorthand`, `bbox_latin`, `bbox_gloss` | `[x0, y0, x1, y1]` in that frame's pixels |
| `latin`, `gloss_en` | the Roman spelling and English gloss as printed |
| `tokens_rule` | shorthand sign tokens predicted from `latin` by the rules in `chinukpipa/translit.py`. This is a hypothesis, not a reading of the image. It is `null` for the 53 French- or English-spelled rows and the 1 low-confidence row. |
| `tokens_verified` | tokens checked sign by sign against the image (`null` = not done yet) |
| `reading_confidence` | confidence in the Roman/gloss reading: high / med / low |
| `reviewed_by`, `notes` | who reviewed the row, and remarks |

**How it was made.** Pages were segmented by code in `chinukpipa/gt/`. The Roman spellings and glosses were then
read from the image crops by AI models (Claude), with OCR used only as a hint:
- **1898 list:** for 181 rows, two models (Claude Opus and Claude Sonnet) read each one separately. They agreed
  on 180. The one disagreement was settled using Le Jeune's own spelling of the same word in his other books.
  The other 11 rows were read by Sonnet and checked by Opus.
- **1892 list:** one model (Sonnet) read it, and the other (Opus) checked each row against the image.

**No human expert has reviewed these readings yet.**

**Images are not included.** To regenerate them:
```
pip install requests pyyaml pillow numpy scipy scikit-image
python -m chinukpipa.corpus.internet_archive download --ids cihm_15465 cihm_15474 --dest corpus/lejeune --kinds orig_jp2
python -m chinukpipa.gt.make_crops data/gt/vocab_annotations.jsonl corpus/lejeune gt_crops/
```
In testing, all 1,272 crops (424 rows × 3 fields) regenerated pixel for pixel identical to the crops the readings were made from.

**Scans used.** Internet Archive copies of the CIHM microfilm. The Internet Archive records list the
contributor as Canadiana.org and the sponsor as University of Alberta Libraries. The scans are not
redistributed here.

**Licence.** Annotations CC BY 4.0. The underlying texts were published in 1892 and 1898, so they are in the public domain in the United States.
