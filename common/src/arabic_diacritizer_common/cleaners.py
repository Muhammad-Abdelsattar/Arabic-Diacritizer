import re
from typing import List, Tuple
from .constants import (
    VALID_ARABIC_CHARS,
    DIACRITIC_CHARS,
    INVALID_SEQUENCES,
    ALL_VALID_DIACRITICS,
    ArabicDiacritics,
)

# Whitespace regex
_whitespace_re = re.compile(r"\s+")


class TextCleaner:
    """Modular text cleaning utilities"""

    @staticmethod
    def collapse_whitespace(text: str) -> str:
        """Collapse multiple whitespace characters into a single space"""
        return re.sub(_whitespace_re, " ", text).strip()

    @staticmethod
    def filter_valid_arabic(text: str) -> str:
        """Keep only valid Arabic characters, punctuation, and diacritics"""
        return "".join(char for char in text if char in VALID_ARABIC_CHARS)

    @staticmethod
    def remove_diacritics(text: str) -> str:
        """Remove all diacritic characters from text"""
        return "".join(ch for ch in text if ch not in DIACRITIC_CHARS)

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize common text irregularities & diacritic order"""
        # Canonicalize diacritic order (make sure Shadda always comes first)
        for invalid, correct in INVALID_SEQUENCES.items():
            text = text.replace(invalid, correct)

        # Normalize alef variants to bare alef (optional, safer for training)
        normalize_map = {"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"}
        for k, v in normalize_map.items():
            text = text.replace(k, v)

        # Remove Tatweel (ـ) since it is purely decorative
        text = text.replace("ـ", "")
        return text

    @staticmethod
    def clean_text(
        text: str, keep_valid_only: bool = True, normalize: bool = False
    ) -> str:
        """Complete cleaning pipeline: normalize → optional filtering → collapse ws"""
        if normalize:
            text = TextCleaner.normalize_text(text)
        if keep_valid_only:
            text = TextCleaner.filter_valid_arabic(text)
        return TextCleaner.collapse_whitespace(text)


class DiacriticValidator:
    """Handles diacritic validation and extraction"""

    @staticmethod
    def extract_diacritics(text: str) -> Tuple[str, List[str]]:
        """
        Extract base text and list of diacritics.
        Each base character gets an associated diacritic string (possibly multiple).
        Example:
          "بَّ" → ("ب", ["َّ"])
        """
        base_chars = []
        diacritics = []

        i = 0
        while i < len(text):
            char = text[i]
            if char in DIACRITIC_CHARS:
                # attach to previous base character if exists
                if base_chars:
                    # Append this diacritic to most recent slot
                    diacritics[-1] = diacritics[-1] + char
                else:
                    # Stray diacritic at beginning — skip or treat as invalid
                    pass
            else:
                # New base char: allocate diacritic slot
                base_chars.append(char)
                diacritics.append("")
            i += 1

        # Normalize combined diacritics to canonical representations
        normalized_diacritics = []
        for d in diacritics:
            if d in ALL_VALID_DIACRITICS:
                normalized_diacritics.append(d)
            else:
                # try to reorder if contains shadda + vowel
                if "ّ" in d:
                    # move shadda to front
                    d = "ّ" + "".join(c for c in d if c != "ّ")
                # keep only known chars
                d = "".join(c for c in d if c in DIACRITIC_CHARS)
                normalized_diacritics.append(d)
        return "".join(base_chars), normalized_diacritics

    @staticmethod
    def validate_diacritics(
        text: str, require_any: bool = False, strict: bool = False
    ) -> str:
        """
        Validate that text diacritics are well-formed.
        - require_any: if True, reject sentences with no diacritics at all.
        - strict: if True, reject unknown/malformed diacritics, else sanitize them.
        Returns text if valid, otherwise "".
        """
        try:
            base_text, diacritics_list = DiacriticValidator.extract_diacritics(text)

            # Optionally require that at least one diacritic is present
            if require_any:
                if not any(
                    d
                    for d in diacritics_list
                    if d != ArabicDiacritics.NO_DIACRITIC.value
                ):
                    return ""

            # In strict mode, reject any diacritic not in valid set
            if strict:
                for d in diacritics_list:
                    if d not in ALL_VALID_DIACRITICS:
                        return ""
            return text
        except Exception:
            return ""

