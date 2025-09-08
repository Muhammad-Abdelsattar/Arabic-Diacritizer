import torch
from torchmetrics import Accuracy, F1Score


def CharAccuracy(num_classes: int, ignore_index: int = 0):
    metric = Accuracy(
        task="multiclass",
        num_classes=num_classes,
        ignore_index=ignore_index,
        average="micro",
    )
    return lambda logits, labels: metric(
        logits.view(-1, logits.size(-1)), labels.view(-1)
    )


def DiacriticF1(num_classes: int, ignore_index: int = 0, average: str = "macro"):
    metric = F1Score(
        task="multiclass",
        num_classes=num_classes,
        ignore_index=ignore_index,
        average=average,
    )
    return lambda logits, labels: metric(
        logits.view(-1, logits.size(-1)), labels.view(-1)
    )


class DiacriticErrorRate:
    """
    DER = wrong diacritics / total diacritics (ignores 'no diacritic' and padding).
    """

    def __init__(self, ignore_index: int = 0, no_diacritic_idx: int = None):
        self.ignore_index = ignore_index
        self.no_diacritic_idx = no_diacritic_idx

    def __call__(self, logits, labels):
        with torch.no_grad():
            preds = logits.argmax(-1)
            mask = labels != self.ignore_index
            if self.no_diacritic_idx is not None:
                mask &= labels != self.no_diacritic_idx
            errors = ((preds != labels) & mask).sum().item()
            total = mask.sum().item()
        return errors / total if total > 0 else 0.0


class _WordLevelMetric:
    """
    Base class for word-level metrics. Not intended to be used directly.
    Handles the logic of identifying words and checking their correctness.
    """

    def __init__(self, ignore_index: int, space_idx: int):
        if space_idx is None:
            raise ValueError("space_idx must be provided for Word-level metrics.")
        self.ignore_index = ignore_index
        self.space_idx = space_idx

    def _calculate_stats(self, logits, labels):
        with torch.no_grad():
            preds = logits.argmax(-1)
            is_correct = preds == labels

            total_words = 0
            correct_words = 0

            # Iterate over each sentence in the batch
            for i in range(labels.size(0)):
                # Get the current sentence's labels and correctness flags
                sent_labels = labels[i]
                sent_correct = is_correct[i]

                # Create a mask to exclude padding
                pad_mask = sent_labels != self.ignore_index

                # Find indices of spaces within the non-padded part of the sentence
                space_indices = (sent_labels[pad_mask] == self.space_idx).nonzero(
                    as_tuple=True
                )[0]

                # Get the full sequence of non-padded correctness flags
                active_correct = sent_correct[pad_mask]

                start_idx = 0
                # Iterate through words using spaces as delimiters
                for space_idx in space_indices:
                    word = active_correct[start_idx:space_idx]
                    if word.numel() > 0:  # Check if the word is not empty
                        total_words += 1
                        if word.all():  # .all() is True if every element is True
                            correct_words += 1
                    start_idx = space_idx + 1

                # Handle the last word in the sentence (after the last space)
                last_word = active_correct[start_idx:]
                if last_word.numel() > 0:
                    total_words += 1
                    if last_word.all():
                        correct_words += 1

            return correct_words, total_words


class WordAccuracy(_WordLevelMetric):
    """
    Word-level accuracy: A word is correct if ALL its diacritics are correct.
    Calculates: (correctly diacritized words) / (total words)
    """

    def __call__(self, logits, labels):
        correct_words, total_words = self._calculate_stats(logits, labels)
        return correct_words / total_words if total_words > 0 else 0.0


class WordErrorRate(_WordLevelMetric):
    """
    Word Error Rate (WER): A word is an error if ANY of its diacritics are wrong.
    Calculates: (words with at least one error) / (total words)
    """

    def __call__(self, logits, labels):
        correct_words, total_words = self._calculate_stats(logits, labels)
        errors = total_words - correct_words
        return errors / total_words if total_words > 0 else 0.0


def get_metrics(
    num_classes: int, ignore_index: int, space_idx: int, no_diacritic_idx: int = None
):
    """
    Build the full set of metrics we want to track.
    Now requires space_idx for word-level metrics.
    """
    return {
        "char_acc": CharAccuracy(num_classes=num_classes, ignore_index=ignore_index),
        "f1_macro": DiacriticF1(
            num_classes=num_classes, ignore_index=ignore_index, average="macro"
        ),
        "der": DiacriticErrorRate(
            ignore_index=ignore_index, no_diacritic_idx=no_diacritic_idx
        ),
        "word_acc": WordAccuracy(ignore_index=ignore_index, space_idx=space_idx),
        "wer": WordErrorRate(ignore_index=ignore_index, space_idx=space_idx),
    }
