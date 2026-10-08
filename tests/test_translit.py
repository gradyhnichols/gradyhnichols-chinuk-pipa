import pytest

from chinukpipa.translit import latin_to_tokens, tokens_to_unicode, unicode_to_tokens, normalize_latin


@pytest.mark.parametrize("word,tokens", [
    ("kamooks", "K A M OO K S"),
    ("kal'kala", "K A L K A L A"),
    ("wa'wa", "WA WA"),
    ("khell", "KH E L"),          # transliterator default: doubled consonant letters -> one sign (a hypothesis)
    ("ka'namoxt", "K A N A M O K S T"),   # default x -> K S (a hypothesis)
    ("tanke son", "T A N K E _ S O N"),
    ("Tanaz", "T A N A S"),        # z -> S (revised rule, docs/rule_notes.md)
    ("Tshok", "CH O K"),           # tsh -> CH (revised rule)
    ("kiu'tan", "K U T A N"),      # iu -> U (revised rule)
    ("Khaw", "KH OW"),             # aw before a consonant or at the end -> OW (revised rule)
    ("Lawagin", "L A WA G E N"),   # aw before a vowel is unchanged
])
def test_latin_to_tokens(word, tokens):
    assert " ".join(latin_to_tokens(word)) == tokens


def test_normalize_strips_stress_and_diacritics():
    assert normalize_latin("Kla'howyam") == "klahowyam"
    assert normalize_latin("enataï") == "enatai"


def test_unicode_roundtrip():
    toks = latin_to_tokens("kla'howyam")
    assert unicode_to_tokens(tokens_to_unicode(toks)) == toks


def test_unknown_letter_raises():
    with pytest.raises(ValueError):
        latin_to_tokens("q9")
