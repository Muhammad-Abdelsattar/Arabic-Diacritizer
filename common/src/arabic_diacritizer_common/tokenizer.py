import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from .constants import ArabicDiacritics, ARABIC_LETTERS, VALID_ARABIC_CHARS
from .cleaners import DiacriticValidator, TextCleaner


class CharTokenizer:
    """
    Character-level tokenizer for Arabic diacritization.

    Input: bare characters (without diacritics)
    Output: per-character diacritic labels (including NO_DIACRITIC)
    """

    def __init__(
        self,
        char2id: Optional[Dict[str, int]] = None,
        diacritic2id: Optional[Dict[str, int]] = None,
        include_punct: bool = True,
        extra_chars: Optional[List[str]] = None,
    ):
        """
        If no vocab mappings are provided, builds defaults from constants.py
        """
        if char2id is None or diacritic2id is None:
            # Base vocabulary from constants
            vocab_chars = list(ARABIC_LETTERS)
            if include_punct:
                vocab_chars += [
                    c for c in VALID_ARABIC_CHARS if c not in ARABIC_LETTERS
                ]
            if extra_chars:
                vocab_chars += extra_chars
            vocab_chars = sorted(set(vocab_chars))

            # Char vocab (+PAD, +UNK)
            char2id = {"<PAD>": 0, "<UNK>": 1}
            char2id.update({ch: idx + 2 for idx, ch in enumerate(vocab_chars)})

            # Diacritic vocab (includes NO_DIACRITIC "")
            diacritic2id = {
                d: i
                for i, d in enumerate(sorted(ArabicDiacritics.valid_combinations()))
            }

        self.char2id = char2id
        self.id2char = {i: c for c, i in char2id.items()}
        self.diacritic2id = diacritic2id
        self.id2diacritic = {i: d for d, i in diacritic2id.items()}

    def save(self, path: str):
        Path(path).write_text(
            json.dumps(
                {"char2id": self.char2id, "diacritic2id": self.diacritic2id},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["char2id"], data["diacritic2id"])

    def encode(self, text: str) -> Tuple[List[int], List[int]]:
        """
        Encode a diacritized string → (input_ids, diacritic_labels)
        """
        # clean_text = TextCleaner.clean_text(text, keep_valid_only=True)
        base_text, diacritics = DiacriticValidator.extract_diacritics(text)

        input_ids = [self.char2id.get(ch, self.char2id["<PAD>"]) for ch in base_text]
        label_ids = [
            self.diacritic2id.get(
                d, self.diacritic2id[ArabicDiacritics.NO_DIACRITIC.value]
            )
            for d in diacritics
        ]
        return input_ids, label_ids

    def encode_for_inference(self, text: str) -> Tuple[List[int], List[int]]:
        """
        Encode a diacritized string → (input_ids, diacritic_labels)
        """
        # clean_text = TextCleaner.clean_text(text, keep_valid_only=True)

        input_ids = [self.char2id.get(ch, self.char2id["<PAD>"]) for ch in base_text]
        label_ids = [
            self.diacritic2id.get(
                d, self.diacritic2id[ArabicDiacritics.NO_DIACRITIC.value]
            )
            for d in diacritics
        ]
        return input_ids, label_ids

    def decode(
        self, input_ids: List[int], label_ids: List[int], cleanup_mode: str = "clean"
    ) -> str:
        """
        Decode (input_ids, label_ids) -> string with diacritics.

        Args:
            input_ids: List of character IDs.
            label_ids: List of predicted diacritic IDs.
            cleanup_mode (str): Determines the post-processing strategy.
                - "clean": (Default) Removes diacritics from non-Arabic letters (e.g., punctuation, spaces).
                - "raw": Returns the raw model output without any cleanup.

        Returns:
            The reconstructed, diacritized string.
        """
        if cleanup_mode not in {"clean", "raw"}:
            raise ValueError("cleanup_mode must be either 'clean' or 'raw'.")

        chars = [self.id2char.get(i, "<UNK>") for i in input_ids]
        diacs = [self.id2diacritic.get(i, "") for i in label_ids]

        if cleanup_mode == "raw":
            return "".join(ch + d for ch, d in zip(chars, diacs))

        # Default is "clean" mode
        cleaned_output = []
        for char, diac in zip(chars, diacs):
            # Only attach a diacritic if the character is a valid Arabic letter
            if char in ARABIC_LETTERS:
                cleaned_output.append(char + diac)
            else:
                cleaned_output.append(
                    char
                )  # Append the character without the predicted diacritic

        return "".join(cleaned_output)

    def decode_inference(
        self,
        text_list: list,
        label_ids: list,
        cleanup_mode: str = "clean",
    ):
        """
        Decode (input_ids, label_ids) -> string with diacritics.

        Args:
            text_list: List of chars in the original text without diacritics.
            label_ids: List of predicted diacritic IDs.
            cleanup_mode (str): Determines the post-processing strategy.
                - "clean": (Default) Removes diacritics from non-Arabic letters (e.g., punctuation, spaces).
                - "raw": Returns the raw model output without any cleanup.

        Returns:
            The reconstructed, diacritized string.
        """
        if cleanup_mode not in {"clean", "raw"}:
            raise ValueError("cleanup_mode must be either 'clean' or 'raw'.")

        diacs = [self.id2diacritic.get(i, "") for i in label_ids]
        cleaned_output = []
        for char, diac in zip(text_list, diacs):
            # Only attach a diacritic if the character is a valid Arabic letter
            if char in ARABIC_LETTERS:
                cleaned_output.append(str(char) + str(diac))
            else:
                cleaned_output.append(
                    char
                )  # Append the character without the predicted diacritic
        return "".join(cleaned_output)
