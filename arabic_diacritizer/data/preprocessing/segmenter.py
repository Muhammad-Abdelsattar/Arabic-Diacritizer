import re
from typing import List
from .cleaners import TextCleaner
from .constants import DIACRITIC_CHARS

def grapheme_length(text: str) -> int:
    """Return logical length of text, counting base characters only (ignore diacritics)."""
    return sum(1 for ch in text if ch not in DIACRITIC_CHARS)

class TextSegmenter:
    """Handles text segmentation and sentence splitting"""

    # Regex for sentence boundaries (includes multi-char delimiters)
    # Treats sequences like "؟!" or "..." as a single delimiter
    SENTENCE_BOUNDARY_RE = re.compile(r"(؟!|!|\?|\.{2,}|…|،|؛)")

    @staticmethod
    def segment_sentences(max_chars: int, line: str) -> List[str]:
        """
        Segment long lines into shorter sentences with max length constraint.
        """
        line = line.strip()
        if not line:
            return []

        if grapheme_length(line) <= max_chars:
            return [TextCleaner.collapse_whitespace(line)]

        # Perform segmentation
        return TextSegmenter._do_segment_sentences(line, max_chars)

    @staticmethod
    def _do_segment_sentences(line: str, max_chars: int) -> List[str]:
        """
        Internal recursive sentence segmentation logic.
        """
        # Split based on boundary regex (keeps delimiters)
        parts = []
        last_idx = 0
        for match in TextSegmenter.SENTENCE_BOUNDARY_RE.finditer(line):
            start, end = match.span()
            segment = line[last_idx:start].strip()
            delimiter = match.group()
            if segment:
                parts.append(segment + delimiter)
            last_idx = end

        if last_idx < len(line):
            remainder = line[last_idx:].strip()
            if remainder:
                parts.append(remainder)

        # Now filter by length
        results: List[str] = []
        for sent in parts:
            sent = TextCleaner.collapse_whitespace(sent)
            if not sent:
                continue
            if grapheme_length(sent) <= max_chars:
                results.append(sent)
            else:
                # Recursive split if still too long
                subsegments = TextSegmenter._recursive_split(sent, max_chars)
                results.extend(subsegments)

        return results

    @staticmethod
    def _recursive_split(text: str, max_chars: int) -> List[str]:
        """
        Splits oversized text recursively by words if necessary.
        """
        words = text.split()
        if not words:
            return []

        segments = []
        cur_segment = []
        cur_len = 0

        for w in words:
            if cur_len + grapheme_length(w) + 1 > max_chars:
                if cur_segment:
                    segments.append(" ".join(cur_segment))
                cur_segment = [w]
                cur_len = grapheme_length(w)
            else:
                cur_segment.append(w)
                cur_len += grapheme_length(w) + 1

        if cur_segment:
            segments.append(" ".join(cur_segment))

        return segments