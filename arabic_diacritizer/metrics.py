import torch
from torchmetrics import Accuracy, F1Score, Metric, MetricCollection
from typing import Dict, Any


# Standard metrics - we'll handle reshaping externally
def get_metrics(num_classes, ignore_index, space_idx, no_diacritic_idx=None):
    """Simple metric collection that expects already-reshaped inputs"""
    return MetricCollection(
        {
            # "char_acc": Accuracy(
            #     task="multiclass",
            #     num_classes=num_classes,
            #     ignore_index=ignore_index,
            #     average="micro",
            # ),
            "f1_macro": ReshapingF1Score(
                task="multiclass",
                num_classes=num_classes,
                ignore_index=ignore_index,
                average="macro",
            ),
            "der": DiacriticErrorRate(
                ignore_index=ignore_index, no_diacritic_idx=no_diacritic_idx
            ),
            # "word_acc": WordAccuracy(ignore_index=ignore_index, space_idx=space_idx),
            # "wer": WordErrorRate(ignore_index=ignore_index, space_idx=space_idx),
        }
    )


class MetricsManager:
    """
    Manages metrics for a specific stage (train/val/test).
    """

    def __init__(self, metrics_config: dict, stage: str, device: torch.device):
        """
        Initialize metrics for a specific stage.

        Args:
            metrics_config: Dictionary of metric configurations
            stage: One of 'train', 'val', or 'test'
        """
        self.stage = stage
        self.metrics = {
            name: metric.to(device) for name, metric in metrics_config.items()
        }
        self.device = device

    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Update all metrics with the current batch.
        """
        for metric in self.metrics.values():
            metric.update(logits, labels)

    def compute(self) -> dict:
        """
        Compute all metrics and return as dictionary.
        """
        return {name: metric.compute() for name, metric in self.metrics.items()}

    def reset(self):
        """Reset all metrics to their initial state."""
        for metric in self.metrics.values():
            metric.reset()

    def get_log_dict(self) -> dict:
        """
        Get metrics in a format ready for Lightning logging.
        """
        metrics = self.compute()
        return {f"{self.stage}_{name}": value for name, value in metrics.items()}


class DiacriticErrorRate(Metric):
    """
    Diacritic Error Rate (DER) metric implementation.
    Calculates the error rate only on diacritic characters.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, ignore_index: int = 0, no_diacritic_idx: int = None):
        super().__init__()
        self.ignore_index = ignore_index
        self.no_diacritic_idx = no_diacritic_idx

        # States to track errors and total diacritic characters
        self.add_state("errors", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update metric state with new predictions and targets.

        Args:
            preds: Tensor of shape (Batch, SeqLen, NumClasses) with logits
            target: Tensor of shape (Batch, SeqLen) with ground truth labels
        """
        # Convert logits to predicted class indices
        preds = preds.argmax(dim=-1)

        # Flatten both tensors for easier processing
        preds_flat = preds.view(-1)
        target_flat = target.view(-1)

        # Create mask for valid characters (not padding and not no-diacritic if specified)
        valid_mask = target_flat != self.ignore_index
        if self.no_diacritic_idx is not None:
            valid_mask = valid_mask & (target_flat != self.no_diacritic_idx)

        # Count errors only on valid diacritic characters
        errors = (preds_flat != target_flat) & valid_mask

        # Update state
        self.errors += errors.sum()
        self.total += valid_mask.sum()

    def compute(self):
        """Compute the final DER value."""
        if self.total == 0:
            return torch.tensor(0.0)
        return self.errors.float() / self.total.float()


class ReshapingF1Score(F1Score):
    """
    F1 Score metric that handles sequence data by reshaping to (Batch*SeqLen, NumClasses).
    """

    def __init__(self, num_classes, ignore_index, average="macro", **kwargs):
        super().__init__(
            task="multiclass",
            num_classes=num_classes,
            ignore_index=ignore_index,
            average=average,
            **kwargs,
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update metric state with new predictions and targets.

        Args:
            preds: Tensor of shape (Batch, SeqLen, NumClasses) with logits
            target: Tensor of shape (Batch, SeqLen) with ground truth labels
        """
        # preds_reshaped = preds.reshape(-1, preds.shape[-1])
        # target_reshaped = target.reshape(-1)
        #
        # print(preds_reshaped.shape, target_reshaped.shape)

        # Call parent update method
        super().update(preds, target)
