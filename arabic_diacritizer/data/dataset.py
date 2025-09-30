import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple, Optional, Union
from pathlib import Path
import pickle
import logging

from arabic_diacritizer_common import CharTokenizer

# Logger setup (remains the same)
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
        diacritic_keep_prob: float = 0.2,
    ):
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        self.file_paths = [Path(fp) for fp in file_paths]
        self.tokenizer = tokenizer
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_format = cache_format.lower()
        self.max_length = max_length
        self.keep_prob = diacritic_keep_prob

        for path in self.file_paths:
            if not path.exists():
                raise FileNotFoundError(f"Dataset file not found: {path}")

        self.lines: List[str] = []
        for path in self.file_paths:
            file_lines = path.read_text(encoding="utf-8").splitlines()
            self.lines.extend(file_lines)
            _LOGGER.info(f"Loaded {len(file_lines)} lines from {path}")

        self.size = len(self.lines)

        self._data = None
        self._inputs_npz = None
        self._labels_npz = None
        self._lengths_npz = None

        if self.cache_dir and self.cache_format != "none":
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            stem_combo = "+".join(path.stem for path in self.file_paths)

            if self.cache_format == "pickle":
                cache_file = self.cache_dir / f"{stem_combo}.pkl"
                self._init_pickle_cache(cache_file, preload)

            elif self.cache_format == "npz":
                inputs_file = self.cache_dir / f"{stem_combo}_inputs.npy"
                labels_file = self.cache_dir / f"{stem_combo}_labels.npy"
                lengths_file = self.cache_dir / f"{stem_combo}_lengths.npy"
                self._init_npz_cache(inputs_file, labels_file, lengths_file)

        elif preload and self.cache_format == "none":
            _LOGGER.info("Preloading in memory without cache...")
            self._data = [self._encode_line(line) for line in self.lines]

        _LOGGER.info(
            f"Dataset initialized with {self.size} samples (cache_format={self.cache_format})."
        )

    def _init_pickle_cache(self, cache_file: Path, preload: bool):
        # This function remains the same as before
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
            _LOGGER.info("Dataset cached with pickle; data fully in RAM by default.")

    def _init_npz_cache(self, inputs_file: Path, labels_file: Path, lengths_file: Path):
        """
        Initializes the NPZ cache, now including a separate file for sequence lengths.
        """
        print(self.max_length)
        print(self.max_length)
        print(self.max_length)
        if self.max_length is None:
            raise ValueError(
                "The `max_length` must be specified when using 'npz' cache format."
            )
        max_len = self.max_length

        if inputs_file.exists() and labels_file.exists() and lengths_file.exists():
            _LOGGER.info("Loading existing NPZ cache with memmap...")
            self._inputs_npz = np.memmap(
                inputs_file, dtype=np.int32, mode="r", shape=(self.size, max_len)
            )
            self._labels_npz = np.memmap(
                labels_file, dtype=np.int32, mode="r", shape=(self.size, max_len)
            )
            self._lengths_npz = np.memmap(
                lengths_file, dtype=np.int32, mode="r", shape=(self.size,)
            )
        else:
            _LOGGER.info("Encoding dataset for NPZ memmap caching...")
            # Create memmap files in write mode
            self._inputs_npz = np.memmap(
                inputs_file, dtype=np.int32, mode="w+", shape=(self.size, max_len)
            )
            self._labels_npz = np.memmap(
                labels_file, dtype=np.int32, mode="w+", shape=(self.size, max_len)
            )
            self._lengths_npz = np.memmap(
                lengths_file, dtype=np.int32, mode="w+", shape=(self.size,)
            )

            for i, line in enumerate(self.lines):
                input_ids, label_ids = self._encode_line(line)
                L = min(len(input_ids), max_len)

                # Write data to the memmapped arrays
                self._inputs_npz[i, :L] = input_ids[:L]
                self._labels_npz[i, :L] = label_ids[:L]
                self._lengths_npz[i] = L  # <-- BEST PRACTICE: Save the true length

                # Explicitly pad the rest of the sequence if necessary
                if L < max_len:
                    self._inputs_npz[i, L:] = 0
                    self._labels_npz[i, L:] = 0

            _LOGGER.info(
                f"Saved NPZ cache: {inputs_file}, {labels_file}, {lengths_file}"
            )

            # Reopen in readonly mode
            self._inputs_npz = np.memmap(
                inputs_file, dtype=np.int32, mode="r", shape=(self.size, max_len)
            )
            self._labels_npz = np.memmap(
                labels_file, dtype=np.int32, mode="r", shape=(self.size, max_len)
            )
            self._lengths_npz = np.memmap(
                lengths_file, dtype=np.int32, mode="r", shape=(self.size,)
            )

    def _encode_line(self, text: str) -> Tuple[List[int], List[int]]:
        # This function remains the same as before
        input_ids, label_ids = self.tokenizer.encode(text)
        if self.max_length:
            input_ids = input_ids[: self.max_length]
            label_ids = label_ids[: self.max_length]
        return input_ids, label_ids

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        MODIFIED: This now returns a tuple of FOUR items:
        - Padded input IDs (NumPy array)
        - PADDED HINT IDs (NumPy array) <--- NEW
        - Padded label IDs (NumPy array)
        - The true, unpadded length of the sequence (integer)
        """
        if self.cache_format == "npz" and self._inputs_npz is not None:
            inputs = self._inputs_npz[idx]
            labels = self._labels_npz[idx]
            length = self._lengths_npz[idx].item()
        else:  # Fallback for pickle cache or on-the-fly
            # This path is slower and requires converting to numpy
            if self.cache_format == "pickle" and self._data is not None:
                input_ids, label_ids = self._data[idx]
            elif self._data is not None:
                input_ids, label_ids = self._data[idx]
            else:
                input_ids, label_ids = self._encode_line(self.lines[idx])

            length = len(input_ids)
            # Create numpy arrays for consistency
            inputs = np.array(input_ids)
            labels = np.array(label_ids)

        # Generate diacritic hints
        # Get the ID for the "mask" token, which is "NO_DIACRITIC".
        no_diacritic_id = self.tokenizer.diacritic2id.get("", 0)

        # Set the probability of KEEPING a diacritic (a tunable hyperparameter)
        keep_prob = self.keep_prob

        # Create a random mask.
        if self.keep_prob == 0:
            hint_ids = np.full_like(labels, fill_value=no_diacritic_id)
        else:
            # Otherwise, use the original probabilistic masking logic.
            keep_prob = self.keep_prob
            mask = np.random.binomial(1, keep_prob, size=labels.shape)
            hint_ids = np.copy(labels)
            hint_ids[mask == 0] = no_diacritic_id

        # Create the hints tensor by masking the true labels.

        return inputs, hint_ids, labels, length
