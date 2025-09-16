import torch
from typing import Optional

from torch.optim.lr_scheduler import LambdaLR

OPTIMIZER_REGISTRY = {}
SCHEDULER_REGISTRY = {}


def register_optimizer(name):
    """Decorator to register an optimizer builder."""

    def wrapper(fn):
        OPTIMIZER_REGISTRY[name] = fn
        return fn

    return wrapper


def register_scheduler(name):
    """Decorator to register a scheduler builder."""

    def wrapper(fn):
        SCHEDULER_REGISTRY[name] = fn
        return fn

    return wrapper


class _LinearWarmupDecay:
    """
    A callable class that implements the linear warmup, linear decay schedule.
    This is picklable and can be correctly saved and loaded by PyTorch Lightning.
    """

    def __init__(self, num_warmup_steps: int, num_training_steps: int):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps

    def __call__(self, current_step: int):
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        return max(
            0.0,
            float(self.num_training_steps - current_step)
            / float(max(1, self.num_training_steps - self.num_warmup_steps)),
        )


@register_optimizer("adam")
def build_adam(params, config: dict = None):
    cfg = config or {}
    return torch.optim.Adam(
        params,
        lr=cfg.get("lr", 1e-3),
        weight_decay=cfg.get("weight_decay", 0.0),
        betas=cfg.get("betas", (0.9, 0.999)),
        eps=cfg.get("eps", 1e-8),
    )


@register_optimizer("adamw")
def build_adamw(params, config: dict = None):
    cfg = config or {}
    return torch.optim.AdamW(
        params,
        lr=cfg.get("lr", 1e-3),
        weight_decay=cfg.get("weight_decay", 0.01),
        betas=cfg.get("betas", (0.9, 0.999)),
        eps=cfg.get("eps", 1e-8),
    )


@register_optimizer("sgd")
def build_sgd(params, config: dict = None):
    cfg = config or {}
    return torch.optim.SGD(
        params,
        lr=cfg.get("lr", 0.1),
        momentum=cfg.get("momentum", 0.9),
        weight_decay=cfg.get("weight_decay", 0.0),
        nesterov=cfg.get("nesterov", False),
    )


@register_scheduler("step_lr")
def build_step_lr(optimizer, config: dict = None):
    cfg = config or {}
    return torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.get("step_size", 10), gamma=cfg.get("gamma", 0.1)
    )


@register_scheduler("cosine")
def build_cosine_lr(optimizer, config: dict = None):
    cfg = config or {}
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.get("t_max", 50)
    )


@register_scheduler("reduce_on_plateau")
def build_plateau_lr(optimizer, config: dict = None):
    cfg = config or {}
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=cfg.get("mode", "min"),
        factor=cfg.get("factor", 0.1),
        patience=cfg.get("patience", 10),
    )


@register_scheduler("linear_warmup")
def build_linear_warmup(optimizer, config: dict = None):
    """Builds the linear warmup, linear decay scheduler using a picklable class."""
    cfg = config or {}
    if "num_training_steps" not in cfg:
        raise ValueError(
            "`num_training_steps` must be specified for linear_warmup scheduler."
        )

    # Instantiate our new picklable class
    lr_lambda_func = _LinearWarmupDecay(
        num_warmup_steps=cfg.get("num_warmup_steps", 500),
        num_training_steps=cfg.get("num_training_steps"),
    )

    return LambdaLR(optimizer, lr_lambda_func)


def get_optimizer(params, config: Optional[dict], scheduler_cfg: Optional[dict] = None):
    """
    Build optimizer (and scheduler if requested).

    # ... (docstring no change) ...
    """
    if config is None:
        raise ValueError("Optimizer config must be provided")

    opt_name = config.get("name", None)
    if opt_name not in OPTIMIZER_REGISTRY:
        raise ValueError(
            f"Unknown optimizer: {opt_name}. Available: {list(OPTIMIZER_REGISTRY.keys())}"
        )

    optimizer = OPTIMIZER_REGISTRY[opt_name](params, config)

    if scheduler_cfg:
        sched_name = scheduler_cfg.get("name")
        if sched_name not in SCHEDULER_REGISTRY:
            raise ValueError(
                f"Unknown scheduler: {sched_name}. Available: {list(SCHEDULER_REGISTRY.keys())}"
            )

        scheduler_instance = SCHEDULER_REGISTRY[sched_name](optimizer, scheduler_cfg)

        # We tell Lightning this by returning a dictionary with the interval.
        scheduler_dict = {
            "scheduler": scheduler_instance,
            "interval": scheduler_cfg.get(
                "interval", "epoch"
            ),  # Default to epoch for old schedulers
            "frequency": 1,
        }

        # For linear_warmup, scheduler, we want the interval to be 'step'.
        if sched_name == "linear_warmup":
            scheduler_dict["interval"] = "step"

        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

    return optimizer
