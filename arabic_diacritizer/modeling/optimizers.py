from typing import Optional
import torch

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


def get_optimizer(params, config: Optional[dict], scheduler_cfg: Optional[dict] = None):
    """
    Build optimizer (and scheduler if requested).

    Args:
        params: model parameters
        config: optimizer config dict {name: "adamw", lr: 1e-3, ...}
        scheduler_cfg: scheduler config dict {name: "step_lr", step_size: 5, ...}

    Returns:
        optimizer or dict with optimizer + scheduler (Lightning-compatible).
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
        scheduler = SCHEDULER_REGISTRY[sched_name](optimizer, scheduler_cfg)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    return optimizer
