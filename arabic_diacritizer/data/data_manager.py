import logging
from typing import List, Optional, Union, Tuple

import torch
from torch.utils.data import DataLoader, random_split, Subset
import lightning as L
import numpy as np  # Import numpy

from .dataset import CharTokenizer, DiacritizationDataset

# Logger setup (remains the same)
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
    # __init__ method remains the same
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

    def collate_fn(self, batch: List[Tuple[np.ndarray, np.ndarray, int]]):
        """
        Optimized collate_fn that leverages pre-padding and dynamic batch sizing.
        """
        # Batch is a list of (padded_input_np, padded_label_np, length) tuples
        inputs_list, labels_list, lengths = zip(*batch)

        # Find the max sequence length *within the current batch*
        max_len_in_batch = max(lengths)

        # Stack the numpy arrays into a single batch array
        # The arrays are already padded to max_length from the dataset
        inputs_np = np.stack(inputs_list, axis=0)
        labels_np = np.stack(labels_list, axis=0)

        # Slice the batch down to the max length found in this batch.
        # This is the crucial step that prevents sending unnecessary padding to the GPU.
        inputs_sliced = inputs_np[:, :max_len_in_batch]
        labels_sliced = labels_np[:, :max_len_in_batch]

        # Convert the sliced numpy arrays to PyTorch tensors.
        # torch.from_numpy is very fast as it avoids a data copy.
        inputs_tensor = torch.from_numpy(inputs_sliced).long()
        labels_tensor = torch.from_numpy(labels_sliced).long()
        lengths_tensor = torch.tensor(lengths, dtype=torch.long)

        return inputs_tensor, labels_tensor, lengths_tensor

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
            _LOGGER.info("Applying sortish bucketing to the training set...")

            # This can be memory intensive for huge datasets, but is fine for millions of lines.
            # It relies on our __getitem__ returning (data, labels, length).
            original_indices = np.arange(len(base_train_ds))
            lengths = [base_train_ds[i][2] for i in range(len(base_train_ds))]

            # Create a new sorted list of indices
            sorted_indices = [
                idx
                for _, idx in sorted(zip(lengths, original_indices), key=lambda x: x[0])
            ]

            # Wrap the original dataset in a Subset that uses the new sorted index order
            base_train_ds = Subset(base_train_ds, sorted_indices)
            _LOGGER.info("Sortish bucketing applied successfully.")

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
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        if self.val_dataset:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )

    def test_dataloader(self):
        if self.test_dataset:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )
