import logging
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from arabic_diacritizer_common import (
   TextCleaner,
   DiacriticValidator,
   TextSegmenter,
)


_LOGGER = logging.getLogger("arabic_diacritizer.preprocessor")
_LOGGER.setLevel(logging.INFO)
if not _LOGGER.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
    )
    handler.setFormatter(formatter)
    _LOGGER.addHandler(handler)


class DatasetPreprocessor:
    """
    Memory-efficient Preprocessor:
    - Processes raw corpus in configurable chunks.
    - Can output to file or list.
    """

    def __init__(
        self,
        max_chars: int = 600,
        validate: bool = True,
        filter_non_arabic: bool = True,
        chunk_size: int = 200,
    ):
        """
        Args:
            max_chars: maximum sentence length before segmentation
            validate: whether to validate diacritic correctness
            filter_non_arabic: remove non-Arabic characters
            chunk_size: number of lines to load and process in one chunk (default=200)
        """
        self.max_chars = max_chars
        self.validate = validate
        self.filter_non_arabic = filter_non_arabic
        self.chunk_size = chunk_size

    def _process_line(self, line: str) -> List[str]:
        line = TextCleaner.clean_text(line, keep_valid_only=self.filter_non_arabic)
        if not line.strip():
            return []

        # Segment
        segments = TextSegmenter.segment_sentences(self.max_chars, line)
        output_segments = []

        for seg in segments:
            if self.validate:
                seg = DiacriticValidator.validate_diacritics(
                    seg, require_any=False, strict=False
                )
                if not seg:
                    continue
            output_segments.append(seg.strip())

        return output_segments

    def process_corpus_to_file(
        self,
        corpus_paths: List[str],
        output_file: str,
        overwrite: bool = True,
        n_lines: Optional[int] = None,
    ) -> str:
        """
        Stream corpus in chunks and write cleaned text to a file (memory-safe).
        """
        output_path = Path(output_file)
        if output_path.exists() and not overwrite:
            _LOGGER.warning(
                f"{output_file} already exists. Skipping because overwrite=False."
            )
            return str(output_path)

        total_written = 0
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as fout:
            for path in corpus_paths:
                p = Path(path)
                if not p.exists():
                    _LOGGER.warning(f"Corpus file not found: {p}")
                    continue

                with p.open("r", encoding="utf-8") as fin:
                    # Estimate total lines if possible
                    try:
                        total_lines = sum(1 for _ in p.open("r", encoding="utf-8"))
                    except Exception:
                        total_lines = None

                    fin.seek(0)
                    pbar = tqdm(
                        total=total_lines, desc=f"Processing {p.name}", unit=" lines"
                    )

                    chunk = []
                    for line in fin:
                        chunk.append(line)
                        if len(chunk) >= self.chunk_size:
                            for l in chunk:
                                if n_lines and total_written >= n_lines:
                                    break
                                for seg in self._process_line(l):
                                    fout.write(seg + "\n")
                                    total_written += 1
                            chunk = []
                            if n_lines and total_written >= n_lines:
                                break
                        pbar.update(1)

                    # Process any leftovers
                    for l in chunk:
                        if n_lines and total_written >= n_lines:
                            break
                        for seg in self._process_line(l):
                            fout.write(seg + "\n")
                            total_written += 1
                        pbar.update(1)

                    pbar.close()

        _LOGGER.info(f"Saved {total_written} cleaned lines to {output_path}")
        return str(output_path)

    def process_corpus_to_list(
        self, corpus_paths: List[str], n_lines: Optional[int] = None
    ) -> List[str]:
        """
        Process corpus and return cleaned lines as a list.
        ⚠️ Not recommended for >100 MB files.
        """
        results = []
        total_processed = 0

        for path in corpus_paths:
            p = Path(path)
            if not p.exists():
                _LOGGER.warning(f"Corpus file not found: {p}")
                continue

            with p.open("r", encoding="utf-8") as fin:
                total_lines = sum(1 for _ in p.open("r", encoding="utf-8"))
                fin.seek(0)
                pbar = tqdm(
                    total=total_lines, desc=f"Processing {p.name}", unit=" lines"
                )

                chunk = []
                for line in fin:
                    chunk.append(line)
                    if len(chunk) >= self.chunk_size:
                        for l in chunk:
                            if n_lines and total_processed >= n_lines:
                                break
                            for seg in self._process_line(l):
                                results.append(seg)
                                total_processed += 1
                        chunk = []
                        if n_lines and total_processed >= n_lines:
                            break
                    pbar.update(1)

                for l in chunk:
                    if n_lines and total_processed >= n_lines:
                        break
                    for seg in self._process_line(l):
                        results.append(seg)
                        total_processed += 1
                    pbar.update(1)
                pbar.close()

        return results
