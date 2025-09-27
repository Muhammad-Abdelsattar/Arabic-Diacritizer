from .constants import DIACRITIC_CHARS, ArabicDiacritics
from .cleaners import DiacriticValidator, TextCleaner


class Postprocessor:
    """
    Provides rule-based post-processing to correct common linguistic errors
    in model-generated diacritized text.
    """

    @staticmethod
    def postprocess(text: str) -> str:
        text = Postprocessor._correct_tanween_fatha_placement(text)
        text = Postprocessor._remove_diacritics_from_alifs(text)
        return text

    @staticmethod
    def _correct_tanween_fatha_placement(text: str) -> str:
        """
        Corrects the placement of Tanween Fatha (ً) from a final Alif (ا)
        to the preceding character. This is a common model error.

        Example: "مَرْحَبَاً" (incorrect) -> "مَرْحَبًا" (correct)
        """
        words = text.split(" ")
        corrected_words = []
        tanween_fatha = ArabicDiacritics.TANWEEN_FATHA.value

        for word in words:

            # Ensure the word is long enough to have a character before a final Alif.
            if len(word) < 2:
                corrected_words.append(word)
                continue

            base_chars, diacritics = DiacriticValidator.extract_diacritics(word)

            if (
                len(base_chars) > 1
                and base_chars[-1] == "ا"
                and diacritics[-1] == tanween_fatha
            ):

                diacritics[-2] = tanween_fatha

                # Ensure the final Alif is left with no diacritic.
                diacritics[-1] = ""

                corrected_word = "".join(
                    [c + d for c, d in zip(base_chars, diacritics)]
                )
                corrected_words.append(corrected_word)
            else:
                corrected_words.append(word)

        return " ".join(corrected_words)

    @staticmethod
    def _remove_diacritics_from_alifs(text: str) -> str:
        """
        Removes any diacritics from plain Alif (ا) and Alif Maqsura (ى)
        anywhere within a word. These characters should not carry short vowels.

        Example:
            - "عَلَىَ" -> "عَلَى"
            - "اِسْم" -> "اِسْم" (This is correct; the model predicted a base letter, not a diacritic)
            - "كِتَابُ" -> "كِتَاب" (Incorrect model output gets corrected)
        """
        words = text.split(" ")
        corrected_words = []
        for word in words:
            base_chars, diacritics = DiacriticValidator.extract_diacritics(word)

            for i in range(len(base_chars)):
                if base_chars[i] == "ا" or base_chars[i] == "ى":
                    diacritics[i] = ""

            corrected_word = "".join([c + d for c, d in zip(base_chars, diacritics)])
            corrected_words.append(corrected_word)

        return " ".join(corrected_words)
