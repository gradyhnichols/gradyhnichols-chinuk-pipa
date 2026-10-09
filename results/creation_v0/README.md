# A first running-text test page: the Creation (version 0)

**What it is.** An alignment between the word boxes of one page of running Chinuk Pipa and the Roman text of the
same words, used to score the reading pipeline in [`chinukpipa/text/`](../../chinukpipa/text/README.md).

- Shorthand: Paul Durieu, *Chinook Bible History* (Kamloops, 1899; shorthand by Le Jeune; Smithsonian Libraries copy), Internet Archive `chinookbiblehist00duri`, leaf 15, left
  column, text lines 5–29 (paragraphs 1–5 of the Creation chapter).
- Roman text: "First Lesson in Chinook", *Chinook and Shorthand Rudiments* (Kamloops, 1898), Internet Archive
  `cihm_15465`, leaves 15–17 (printed pages 11–13). 208 Chinook words.

**How it was made.** By an AI agent (Claude), with no human check:
- The Roman words were taken from the Internet Archive OCR and corrected against the page images
  (`ocr_corrections.txt`; about 66 OCR slips). The OCR gives each column's Chinook and English words as separate
  blocks, so the word order was rebuilt from the images (`roman_words.json`).
- Word boxes come from `chinukpipa.text.segment` on that leaf (box key "line.index", where index counts every entry
  of the line's `words` list, from 1). Boxes were aligned with the Roman words in reading order by dynamic
  programming on token edit distance, allowing a word to span two or three boxes and a box to hold two words, then
  checked by eye; 16 rows were corrected by eye (`alignment_check`).
- Checks reported by the agent: two alignment variants agreed on 192 of 208 rows; the same scoring against
  shuffled Roman text gives 4–4.5% first-choice accuracy; repeated words show the same outline each time.

**Files.**
- `alignment.tsv`: one row per Roman word. The alignment itself is in `roman_printed`, `roman_spelling`,
  `roman_flag` (`abbr` abbreviation, `resp` English loan printed with a respelling, which was used; `loan` English
  loan printed in English spelling), `roman_tokens`, `box_ids`, `main_box`, `unit_id` and `unit_kind` (`match`: one
  box, one word; `split`: several boxes, one word; `merge`: one box, several words; `merge+split`). The other
  columns are the first run's readings (three models trained on the 1892 and 1898 words, word list
  `lexicon_merged.tsv`, stage B only) and the agent's classification of each miss (`cause`, `cause_detail`).
- `roman_words.json`: the 208 words with their printed form, spelling used, tokens and word-list status.
- `ocr_corrections.txt`: how the OCR was corrected and which spellings were normalized.
- `readings_final6_stageC.json`: the stage-C output for this leaf from the five models trained also on the 1924
  vocabulary rows, with Le Jeune's spellings in the word list (153 of 208 right by the strict rule). The
  intermediate rows of the table in `chinukpipa/text/README.md` (137 and 152) were not archived.
- `boxes_leaf15.json`: the segmentation's text lines and word boxes for the leaf (keys, kinds, `bbox` in pixels of the
  deskewed page), so that the box keys can be checked without re-running the segmenter. Two rows refer to ink the
  segmenter set aside as braces (`brace532`, `brace1278`, component numbers of the full segmentation); those have
  no key here.
- On this page the segmenter did not separate the two columns: text lines run across the page. Only boxes of the
  left column (x below about 1,200 pixels) are aligned.

Score readings with `python -m chinukpipa.text.score_alignment results/creation_v0/alignment.tsv <read or merged
JSON>`. The box keys refer to the segmentation of this leaf by `chinukpipa.text.segment` as of October 2026; a
change to the segmenter can renumber boxes.

**Limits.** One page; AI readings and AI alignment; English loanwords printed in English spelling have rule tokens
that are not meaningful. The abbreviation S.T. (10 times) is read here as the letters S, T.
