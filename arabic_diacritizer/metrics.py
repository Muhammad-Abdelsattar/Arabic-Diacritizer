import torch
from torchmetrics import Accuracy, F1Score


def CharAccuracy(num_classes: int, ignore_index: int = 0):
    """
    Character-level accuracy using TorchMetrics.
    Accumulates over an epoch, handles distributed.
    """
    return Accuracy(
        task="multiclass",
        num_classes=num_classes,
        ignore_index=ignore_index,
        average="micro",
    )


def DiacriticF1(num_classes: int, ignore_index: int = 0, average: str = "macro"):
    """
    F1 score for diacritics using TorchMetrics.
    Good for handling class imbalance (rare diacritics).
    """
    return F1Score(
        task="multiclass",
        num_classes=num_classes,
        ignore_index=ignore_index,
        average=average,
    )


class DiacriticErrorRate:
    """
    DER = wrong diacritics / total diacritics (ignores 'no diacritic' and padding).
    """

    def __init__(self, ignore_index: int = 0, no_diacritic_idx: int = None):
        self.ignore_index = ignore_index
        self.no_diacritic_idx = no_diacritic_idx  # index for "no diacritic" label ("")

    def __call__(self, logits, labels):
        with torch.no_grad():
            preds = logits.argmax(-1)
            # mask: valid diacritic positions (exclude padding, exclude "no diacritic")
            mask = (labels != self.ignore_index)
            if self.no_diacritic_idx is not None:
                mask &= (labels != self.no_diacritic_idx)

            errors = ((preds != labels) & mask).sum().item()
            total = mask.sum().item()
        return errors / total if total > 0 else 0.0


class WordAccuracy:
    """
    Word-level accuracy: word correct if ALL diacritics in it are correct.
    """

    def __init__(self, ignore_index: int = 0):
        self.ignore_index = ignore_index

    def __call__(self, logits, labels):
        with torch.no_grad():
            preds = logits.argmax(-1)
            mask = labels != self.ignore_index

            # mark non-padded positions as "must match"
            correct_mask = (preds == labels) | ~mask
            word_correct = correct_mask.all(dim=1).sum().item()
            total = labels.size(0)
        return word_correct / total if total > 0 else 0.0


class WordErrorRate:
    """
    Word Error Rate (WER) for diacritization.
    Word counts as error if ANY diacritic in it is wrong.
    (essentially 1 - WordAccuracy).
    """

    def __init__(self, ignore_index: int = 0):
        self.ignore_index = ignore_index

    def __call__(self, logits, labels):
        with torch.no_grad():
            preds = logits.argmax(-1)
            mask = labels != self.ignore_index

            correct_mask = (preds == labels) | ~mask
            all_correct = correct_mask.all(dim=1)

            total = labels.size(0)
            errors = (~all_correct).sum().item()
        return errors / total if total > 0 else 0.0


# Metric Factory
def get_metrics(vocab_size: int, ignore_index: int = 0):
    """
    Build the full set of metrics we want to track.
    Returns a dict {metric_name: metric_callable}
    """
    return {
        "char_acc": CharAccuracy(num_classes=vocab_size, ignore_index=ignore_index),
        "f1_macro": DiacriticF1(
            num_classes=vocab_size, ignore_index=ignore_index, average="macro"
        ),
        "der": DiacriticErrorRate(ignore_index=ignore_index),
        "word_acc": WordAccuracy(ignore_index=ignore_index),
        "wer": WordErrorRate(ignore_index=ignore_index),
    }
