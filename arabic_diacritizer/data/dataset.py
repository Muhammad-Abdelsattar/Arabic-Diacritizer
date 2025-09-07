import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple, Optional, Union
from pathlib import Path
import pickle
import logging

from .tokenizer import CharTokenizer

# Logger setup
_LOGGER = logging.getLogger("arabic_diacritizer.dataset")
_LOGGER.setLevel(logging.INFO)
if not _LOGGER.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
    )
    handler.setFormatter(formatter)
    _LOGGER.addHandler(handler)


class DiacritizationDataset(Dataset):
    """
    PyTorch Dataset for Arabic diacritization.

    Supports 3 caching modes:
    - "pickle": Cache all encoded data in memory (fast, but large RAM).
    - "npz"   : Disk-backed cache using numpy.memmap (scales to huge datasets).
    - "none"  : No cache, tokenize on the fly (slowest).
    """

    def __init__(
        self,
        file_paths: Union[str, List[str]],
        tokenizer: CharTokenizer,
        cache_dir: Optional[str] = None,
        cache_format: str = "pickle",
        preload: bool = False,
        max_length: Optional[int] = None,
    ):
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        self.file_paths = [Path(fp) for fp in file_paths]
        self.tokenizer = tokenizer
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_format = cache_format.lower()
        self.max_length = max_length

        for path in self.file_paths:
            if not path.exists():
                raise FileNotFoundError(f"Dataset file not found: {path}")

        # Load all lines
        self.lines: List[str] = []
        for path in self.file_paths:
            file_lines = path.read_text(encoding="utf-8").splitlines()
            self.lines.extend(file_lines)
            _LOGGER.info(f"Loaded {len(file_lines)} lines from {path}")

        self.size = len(self.lines)

        # Storage
        self._data = None
        self._inputs_npz = None
        self._labels_npz = None

        # Initialize according to cache format
        if self.cache_dir and self.cache_format != "none":
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            stem_combo = "+".join(path.stem for path in self.file_paths)

            if self.cache_format == "pickle":
                cache_file = self.cache_dir / f"{stem_combo}.pkl"
                self._init_pickle_cache(cache_file, preload)

            elif self.cache_format == "npz":
                inputs_file = self.cache_dir / f"{stem_combo}_inputs.npy"
                labels_file = self.cache_dir / f"{stem_combo}_labels.npy"
                self._init_npz_cache(inputs_file, labels_file)

        elif preload and self.cache_format == "none":
            _LOGGER.info("Preloading in memory without cache...")
            self._data = [self._encode_line(line) for line in self.lines]

        _LOGGER.info(
            f"Dataset initialized with {self.size} samples (cache_format={self.cache_format})."
        )

    # ------------------------------
    # Initialization subroutines
    # ------------------------------
    def _init_pickle_cache(self, cache_file: Path, preload: bool):
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                self._data = pickle.load(f)
            _LOGGER.info(
                f"Loaded pickle cached dataset: {cache_file} ({len(self._data)} samples)"
            )
        else:
            _LOGGER.info("Encoding dataset for pickle caching...")
            self._data = [self._encode_line(line) for line in self.lines]
            with open(cache_file, "wb") as f:
                pickle.dump(self._data, f)
            _LOGGER.info(f"Saved pickle cache to {cache_file}")

        if not preload:
            # Optional: keep only reference to lines if not preloading
            _LOGGER.info("Dataset cached with pickle; data fully in RAM by default.")

    def _init_npz_cache(self, inputs_file: Path, labels_file: Path):
        max_len = self.max_length or max(len(line) for line in self.lines)  # fallback
        if inputs_file.exists() and labels_file.exists():
            _LOGGER.info("Loading existing NPZ cache with memmap...")
            self._inputs_npz = np.memmap(
                inputs_file, dtype=np.int32, mode="r", shape=(self.size, max_len)
            )
            self._labels_npz = np.memmap(
                labels_file, dtype=np.int32, mode="r", shape=(self.size, max_len)
            )
        else:
            _LOGGER.info("Encoding dataset for NPZ memmap caching...")
            # Create memmap files
            self._inputs_npz = np.memmap(
                inputs_file, dtype=np.int32, mode="w+", shape=(self.size, max_len)
            )
            self._labels_npz = np.memmap(
                labels_file, dtype=np.int32, mode="w+", shape=(self.size, max_len)
            )
            for i, line in enumerate(self.lines):
                input_ids, label_ids = self._encode_line(line)
                L = min(len(input_ids), max_len)
                self._inputs_npz[i, :L] = input_ids[:L]
                self._labels_npz[i, :L] = label_ids[:L]
                if L < max_len:
                    self._inputs_npz[i, L:] = 0
                    self._labels_npz[i, L:] = 0
            _LOGGER.info(f"Saved NPZ cache: {inputs_file}, {labels_file}")
            # Reopen in readonly mode to avoid locks
            self._inputs_npz = np.memmap(
                inputs_file, dtype=np.int32, mode="r", shape=(self.size, max_len)
            )
            self._labels_npz = np.memmap(
                labels_file, dtype=np.int32, mode="r", shape=(self.size, max_len)
            )

    def _encode_line(self, text: str) -> Tuple[List[int], List[int]]:
        input_ids, label_ids = self.tokenizer.encode(text)
        if self.max_length:
            input_ids = input_ids[: self.max_length]
            label_ids = label_ids[: self.max_length]
        return input_ids, label_ids

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        if self.cache_format == "pickle" and self._data is not None:
            return self._data[idx]
        elif self.cache_format == "npz" and self._inputs_npz is not None:
            return (self._inputs_npz[idx].tolist(), self._labels_npz[idx].tolist())
        elif self._data is not None:  # preload with no cache
            return self._data[idx]
        else:
            return self._encode_line(self.lines[idx])
