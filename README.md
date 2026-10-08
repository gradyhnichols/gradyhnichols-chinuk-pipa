# chinuk-pipa

Open tools and open data toward reading **Chinuk Pipa**, the shorthand that Father J.-M.-R. Le Jeune (1855–1930)
adapted from French Duployan stenography to write Chinook Jargon (Chinuk Wawa) in Kamloops, British Columbia,
in the 1890s.

> **This is an amateur project, and the maintainer's first.** It was started in October 2026 and is carried
> out with AI research tools (see [How this was made](#how-this-was-made)). It is not affiliated with any
> university, archive or language community, and it is not an authority on the language or its history.
> Please treat everything here as a working draft. Corrections are very welcome.

## Why

According to David Robertson's 2011 dissertation (page numbers in brackets):

- He located about 600 texts that Indigenous people wrote in the script between 1891 and 1912 [p. 12].
- He estimates that 50–75% of all documented Chinuk Wawa is written in it [pp. 11, 25; p. 50 gives "about 75%"].
- Le Jeune's newspaper, the *Kamloops Wawa*, was written in it for Indigenous readers from 1891 [p. 12].

We could not find any published software that reads the script from page images (OCR or handwriting
recognition), though our search may have missed something. This project builds small, open, checkable pieces
toward that, and shares them freely.

## What's here (v0.1, October 2026)

| Piece | What it is | Notes |
|---|---|---|
| **Ground truth** [`data/gt/`](data/gt/) | 424 shorthand word images from two word lists printed at Kamloops in 1892 and 1898. Each is paired with the Roman spelling and English gloss printed beside it. | Stored as page coordinates plus readings. Images regenerate from Internet Archive scans. Readings were made by AI and have not yet been reviewed by a human expert. |
| **Transliterator** [`chinukpipa/translit.py`](chinukpipa/translit.py) | Le Jeune's Roman spelling → shorthand sign tokens → Unicode Duployan text. | The spelling-to-sign rules are hypotheses, to be tested against the ground truth. |
| **Sign data** [`data/signs/`](data/signs/) | 81 sign entries (70 with a Unicode code point) and 58 abbreviations/logograms, each with its sound value, stroke description, joining notes and sources. | Compiled by AI agents from Robertson (2011), Le Jeune's printed sign tables, the Unicode documents and the Kaltash Wawa guide. Not yet reviewed by a specialist. |
| **Chinook word list** [`data/lexicon/`](data/lexicon/) | About 2,000 headwords parsed from 10 public-domain dictionaries and word lists (1863–1924). Each row cites its source. | Parsed from OCR text, so expect errors. Each row carries a quality grade. Glosses are quoted as printed in 1863–1924 and include terms now considered offensive. They are kept as historical evidence, not endorsed. |
| **Page tools** [`chinukpipa/gt/`](chinukpipa/gt/) | Deskewing, column and row segmentation, crop regeneration, review sheets. | |

Planned: a small recognition model trained on these examples, and research notes once they have been checked.

## Quick start

```bash
pip install pyyaml pillow numpy scipy scikit-image   # add requests (downloads) and pytesseract (harvest_vocab) as needed
python -m chinukpipa.translit          # prints a few example words as tokens and Unicode
```

```python
from chinukpipa.translit import latin_to_tokens, tokens_to_unicode
latin_to_tokens("kamooks")             # ['K', 'A', 'M', 'OO', 'K', 'S']
```

To display Duployan text you need a font that supports it, such as Noto Sans Duployan, or David Corbett's
[Rawnd Musmus Duployan](https://github.com/dscorbett/duployan-font).

## Built on the work of others

- **David D. Robertson**: *Kamloops Chinúk Wawa, Chinuk Pipa, and the Vitality of Pidgins* (PhD dissertation,
  University of Victoria, 2011) and his blog [chinookjargon.com](https://chinookjargon.com).
- **Kaltash Wawa**: the *Chinuk Pipa Guide* (2024).
- **David Corbett**: the Rawnd Musmus Duployan font and online Duployan keyboard.
- **Van Anderson** (with later contributions from Michael Everson and others): the proposals that brought
  Duployan, with its Chinook letters, into Unicode 7.0.
- **Canadiana.org / CRKN** (the CIHM microfilm program), **University of Alberta Libraries**, the **Newberry Library** and the **Internet Archive**, for the digitized books.

Any mistakes here are ours, not theirs.

## Respect for the languages and communities

Chinuk Wawa is a living language with its own communities of speakers and learners. The Salish languages that
also appear in Le Jeune's books belong to their nations. This repository does not publish transcriptions or
analysis of Salish-language texts. The sign data only records the Unicode letters that are tagged as Salishan
and a few place-name abbreviations. If you speak for a community and want something changed or removed,
please open an issue.

## Sources and rights

- The ground truth comes from *Chinook Vocabulary, Chinook–English* (Kamloops, 1892, "from the original of
  Rt. Rev. Bishop Durieu") and Le Jeune's *Chinook and Shorthand Rudiments* (Kamloops, 1898). We used the
  Internet Archive copies of the CIHM microfilm, items `cihm_15474` and `cihm_15465`. The Internet Archive
  records list the contributor as Canadiana.org and the sponsor as University of Alberta Libraries.
- Page images are **not** stored in this repository. Annotations point into those scans by item, page and
  pixel coordinates, and `python -m chinukpipa.gt.make_crops` regenerates the crops.
- The word list cites, for each row, the public-domain source it was parsed from. All were published before 1931.

## Licences

- Code: [Apache-2.0](LICENSE).
- Data and annotations created here: [CC BY 4.0](DATA_LICENSE.md).

## Contributing

The most useful help is **checking readings**: confirming or correcting what a shorthand word says. See
[CONTRIBUTING.md](CONTRIBUTING.md). You don't need to write code.

## How this was made

This is an amateur project. AI research agents (Anthropic's Claude) found and read sources, wrote the code,
read the word images, and compiled the sign data. Most claims in the data carry an evidence tag (VERIFIED /
COMPUTED / INFERRED / UNVERIFIED) and a source. VERIFIED means an AI read the cited passage, not that a person
confirmed it. AI tools can be confidently wrong, so please check anything you rely on against the cited
source, and tell us when we got it wrong.
