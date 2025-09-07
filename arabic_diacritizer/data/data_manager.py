import logging
from typing import List, Optional, Union

import torch
from torch.utils.data import DataLoader, random_split
import lightning as L

from .dataset import CharTokenizer, DiacritizationDataset

_LOGGER = logging.getLogger("arabic_diacritizer.data_manager")
_LOGGER.setLevel(logging.INFO)
if not _LOGGER.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
    )
    handler.setFormatter(formatter)
    _LOGGER.addHandler(handler)


class DataManager(L.LightningDataModule):
    """
    Lightning DataModule to orchestrate datasets and loaders for Arabic Diacritization.
    Handles smart splitting behavior depending on provided files.
    """

    def __init__(
        self,
        train_files: Union[str, List[str]],
        val_files: Optional[Union[str, List[str]]] = None,
        test_files: Optional[Union[str, List[str]]] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        cache_dir: Optional[str] = None,
        cache_format: str = "npz",
        max_length: Optional[int] = None,
        val_split: float = 0.1,
        test_split: float = 0.1,
    ):
        super().__init__()
        self.train_files = (
            [train_files] if isinstance(train_files, str) else train_files
        )
        self.val_files = (
            [val_files]
            if val_files and isinstance(val_files, str)
            else (val_files or [])
        )
        self.test_files = (
            [test_files]
            if test_files and isinstance(test_files, str)
            else (test_files or [])
        )

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        self.cache_format = cache_format
        self.max_length = max_length
        self.val_split = val_split
        self.test_split = test_split

        self.tokenizer = CharTokenizer()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def collate_fn(self, batch):
        pad_idx = self.tokenizer.char2id["<PAD>"]
        input_seqs, label_seqs = zip(*batch)
        lengths = [len(seq) for seq in input_seqs]
        max_len = max(lengths)

        padded_inputs = [inp + [pad_idx] * (max_len - len(inp)) for inp in input_seqs]
        padded_labels = [lab + [pad_idx] * (max_len - len(lab)) for lab in label_seqs]

        return (
            torch.tensor(padded_inputs, dtype=torch.long),
            torch.tensor(padded_labels, dtype=torch.long),
            torch.tensor(lengths, dtype=torch.long),
        )

    def setup(self, stage: Optional[str] = None):
        _LOGGER.info(f"Setting up DataManager (stage={stage})")

        if stage == "fit" or stage is None:
            base_train_ds = DiacritizationDataset(
                file_paths=self.train_files,
                tokenizer=self.tokenizer,
                cache_dir=self.cache_dir,
                cache_format=self.cache_format,
                max_length=self.max_length,
            )

            # CASE 1: All three provided
            if self.val_files and self.test_files:
                _LOGGER.info("Train/Val/Test files all provided -> using as-is")
                self.train_dataset = base_train_ds
                self.val_dataset = DiacritizationDataset(
                    self.val_files,
                    self.tokenizer,
                    self.cache_dir,
                    self.cache_format,
                    self.max_length,
                )
                self.test_dataset = DiacritizationDataset(
                    self.test_files,
                    self.tokenizer,
                    self.cache_dir,
                    self.cache_format,
                    self.max_length,
                )

            # CASE 2: Only train provided
            elif not self.val_files and not self.test_files:
                _LOGGER.warning(
                    "Only train set provided. Splitting into train/val/test"
                )
                n_total = len(base_train_ds)
                n_val = max(1, int(n_total * self.val_split))
                n_test = max(1, int(n_total * self.test_split))
                n_train = n_total - n_val - n_test
                self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                    base_train_ds, [n_train, n_val, n_test]
                )
                _LOGGER.info(
                    f"Split dataset: train={n_train}, val={n_val}, test={n_test}"
                )

            # CASE 3: Train + Test provided, no Val
            elif self.test_files and not self.val_files:
                _LOGGER.warning(
                    "Validation set not provided. Splitting train set for validation"
                )
                self.test_dataset = DiacritizationDataset(
                    self.test_files,
                    self.tokenizer,
                    self.cache_dir,
                    self.cache_format,
                    self.max_length,
                )
                n_total = len(base_train_ds)
                n_val = max(1, int(n_total * self.val_split))
                n_train = n_total - n_val
                self.train_dataset, self.val_dataset = random_split(
                    base_train_ds, [n_train, n_val]
                )
                _LOGGER.info(
                    f"Split dataset: train={n_train}, val={n_val}, test={len(self.test_dataset)}"
                )

            # CASE 4: Train + Val provided, no Test
            elif self.val_files and not self.test_files:
                _LOGGER.warning(
                    "Test set not provided. Using provided Val as Test. Splitting train for new Val set"
                )
                self.test_dataset = DiacritizationDataset(
                    self.val_files,
                    self.tokenizer,
                    self.cache_dir,
                    self.cache_format,
                    self.max_length,
                )
                n_total = len(base_train_ds)
                n_val = max(1, int(n_total * self.val_split))
                n_train = n_total - n_val
                self.train_dataset, self.val_dataset = random_split(
                    base_train_ds, [n_train, n_val]
                )
                _LOGGER.info(
                    f"Split dataset: train={n_train}, new_val={n_val}, test={len(self.test_dataset)}"
                )

        # Ensure test dataset exists for "test" stage as well
        if stage == "test" or stage is None:
            if self.test_dataset is None and self.test_files:
                self.test_dataset = DiacritizationDataset(
                    self.test_files,
                    self.tokenizer,
                    self.cache_dir,
                    self.cache_format,
                    self.max_length,
                )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        if self.val_dataset:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )

    def test_dataloader(self):
        if self.test_dataset:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )
