import enum
from typing import Set, FrozenSet
import re


class ArabicDiacritics(enum.Enum):
    """All possible Arabic diacritics (standard + extended)."""

    NO_DIACRITIC = ""
    SUKOON = "ْ"
    SHADDA = "ّ"
    DAMMA = "ُ"
    FATHA = "َ"
    KASRA = "ِ"
    TANWEEN_DAMMA = "ٌ"
    TANWEEN_FATHA = "ً"
    TANWEEN_KASRA = "ٍ"
    # Combinations where order may vary in Unicode
    SHADDA_PLUS_DAMMA = "ُّ"  # normalize to 'shadda then vowel'
    SHADDA_PLUS_FATHA = "َّ"
    SHADDA_PLUS_KASRA = "ِّ"
    SHADDA_PLUS_TANWEEN_DAMMA = "ٌّ"
    SHADDA_PLUS_TANWEEN_FATHA = "ًّ"
    SHADDA_PLUS_TANWEEN_KASRA = "ٍّ"

    # Quranic / orthographic additions
    DAGGER_ALEF = "ٰ"  # superscript Alef (dagger)
    MADDA = "ٓ"  # Maddah
    WASLA = "ٱ"  # Hamzat Wasl (technically letter with mark)

    @classmethod
    def chars(cls) -> Set[str]:
        """Return set of atomic (single-character) diacritics."""
        return {
            cls.SUKOON.value,
            cls.SHADDA.value,
            cls.DAMMA.value,
            cls.FATHA.value,
            cls.KASRA.value,
            cls.TANWEEN_DAMMA.value,
            cls.TANWEEN_FATHA.value,
            cls.TANWEEN_KASRA.value,
            cls.DAGGER_ALEF.value,
            cls.MADDA.value,
        }

    @classmethod
    def valid_combinations(cls) -> Set[str]:
        """Return full set of valid diacritic combinations."""
        return {
            cls.NO_DIACRITIC.value,
            # Singles
            cls.SUKOON.value,
            cls.DAMMA.value,
            cls.FATHA.value,
            cls.KASRA.value,
            cls.TANWEEN_DAMMA.value,
            cls.TANWEEN_FATHA.value,
            cls.TANWEEN_KASRA.value,
            cls.DAGGER_ALEF.value,
            cls.MADDA.value,
            # Shadda combos
            cls.SHADDA_PLUS_DAMMA.value,
            cls.SHADDA_PLUS_FATHA.value,
            cls.SHADDA_PLUS_KASRA.value,
            cls.SHADDA_PLUS_TANWEEN_DAMMA.value,
            cls.SHADDA_PLUS_TANWEEN_FATHA.value,
            cls.SHADDA_PLUS_TANWEEN_KASRA.value,
        }

    @classmethod
    def is_valid_diacritic(cls, diacritic: str) -> bool:
        return diacritic in cls.valid_combinations()


# Character sets
WORD_SEPARATOR = " "


# Arabic letters base Unicode block (0600–06FF covers standard Arabic letters)
ARABIC_LETTERS_BASE = [chr(c) for c in range(0x0621, 0x064B)]
# Extended Arabic letters (found in borrowed words, Persian/Urdu usage)
ARABIC_LETTERS_EXTENDED_BLOCK = [
    "ى",  # Alef Maqsura
    "ة",  # Taa Marbuta
    "پ",
    "چ",
    "ڤ",
    "گ",  # Persian/Urdu additions
]
ALEF_VARIANTS = {"ا", "أ", "إ", "آ"}

# Merge all letters
ARABIC_LETTERS = frozenset(
    ARABIC_LETTERS_BASE + ARABIC_LETTERS_EXTENDED_BLOCK + list(ALEF_VARIANTS)
)

# Punctuation
PUNCTUATIONS = frozenset(
    {
        ".",
        "،",
        ":",
        "؛",
        "-",
        "؟",
        "!",
        "(",
        ")",
        "[",
        "]",
        '"',
        "«",
        "»",
        "/",
        ";",
        ",",
        "…",
        "ـ",  # ellipsis + tatweel
    }
)
SENTENCE_DELIMITERS = {".", "؟", "!", "،", ":", "؛", "…"}
WORD_DELIMITERS = {WORD_SEPARATOR, *SENTENCE_DELIMITERS}

# Diacritics sets
DIACRITIC_CHARS = ArabicDiacritics.chars()
ALL_VALID_DIACRITICS = ArabicDiacritics.valid_combinations()

# All valid characters
VALID_ARABIC_CHARS = {WORD_SEPARATOR, *ARABIC_LETTERS, *PUNCTUATIONS, *DIACRITIC_CHARS}

# Text normalization (fixes diacritic ordering inconsistencies)
INVALID_SEQUENCES = {
    # Normalize to canonical "SHADDA first, VOWEL after"
    "َّ": "َّ",  # fatha + shadda → shadda + fatha
    "ِّ": "ِّ",  # kasra + shadda → shadda + kasra
    "ُّ": "ُّ",  # damma + shadda → shadda + damma
    "ًّ": "ًّ",  # tanween fatha
    "ٍّ": "ٍّ",  # tanween kasra
    "ٌّ": "ٌّ",  # tanween damma
    # Punctuation spacing corrections
    " ،": "،",
    " .": ".",
    " ؟": "؟",
    " ؛": "؛",
    " …": "…",
}

# Regex for Arabic letters
ARABIC_LETTERS_REGEX = re.compile(f'[{"".join(ARABIC_LETTERS)}]+')
