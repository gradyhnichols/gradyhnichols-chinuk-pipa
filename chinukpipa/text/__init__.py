"""Reading running text: split page images into lines and words, and read the words with the recognizer.

The pipeline has three stages (see README.md in this folder): A, `segcrops` (segment pages, save word images);
B, `readcrops` (read every word image); C, `remerge` (join word pieces that read better together).
"""
from pathlib import Path

# the merged word list of this repository, used when --lexicon is not given
DEFAULT_LEXICON = str(Path(__file__).resolve().parents[2] / "data" / "lexicon" / "lexicon_merged.tsv")
