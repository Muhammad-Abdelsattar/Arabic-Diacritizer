from typing import Optional
import torch.nn as nn

LOSS_REGISTRY = {}


def register_loss(name):
    """Decorator to register a loss function into the global registry."""

    def wrapper(cls_or_fn):
        LOSS_REGISTRY[name] = cls_or_fn
        return cls_or_fn

    return wrapper


@register_loss("cross_entropy")
def build_cross_entropy(config: dict = None):
    cfg = config or {}
    ignore_index = cfg.get("ignore_index", -100)  # by default no ignore
    label_smoothing = cfg.get("label_smoothing", 0.0)
    reduction = cfg.get("reduction", "mean")
    return nn.CrossEntropyLoss(
        ignore_index=ignore_index, label_smoothing=label_smoothing, reduction=reduction
    )


@register_loss("nll_loss")
def build_nll_loss(config: dict = None):
    cfg = config or {}
    reduction = cfg.get("reduction", "mean")
    ignore_index = cfg.get("ignore_index", -100)
    return nn.NLLLoss(reduction=reduction, ignore_index=ignore_index)


def get_loss(config: Optional[dict] = None, name: Optional[str] = None, **kwargs):
    """
    Factory for loss functions.

    Args:
        config: dictionary such as {"name": "cross_entropy", "ignore_index": 0}
        name: optional, override config name
        kwargs: overrides

    Returns:
        nn.Module
    """
    if config is None and name is None:
        return build_cross_entropy(**kwargs)

    if isinstance(config, dict):
        loss_name = config.get("name", None) or name
        cfg = {**config}
        cfg.pop("name", None)
    else:
        loss_name = name
        cfg = {}

    if loss_name not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss: {loss_name}. Available: {list(LOSS_REGISTRY.keys())}"
        )

    return LOSS_REGISTRY[loss_name](cfg | kwargs)
