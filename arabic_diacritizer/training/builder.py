import logging
from typing import Dict, Any
import torch
import lightning as L

from ..data import DataManager

from ..modeling import ModelingOrchestrator
from ..modeling import build_model
from ..modeling import get_loss
from ..metrics import get_metrics

from .loggers import get_loggers
from .callbacks import get_callbacks

_LOGGER = logging.getLogger(__name__)


def build_data_manager(data_config: Dict[str, Any]) -> DataManager:
    """Initializes the DataManager."""
    _LOGGER.info("Building DataManager...")
    datamodule = DataManager(**data_config)
    _LOGGER.info("DataManager built successfully.")
    return datamodule


def build_modeling_orchestrator(
    modeling_config: Dict[str, Any],
    datamodule: DataManager,
    reset_optimizer_and_scheduler: bool = False,
) -> ModelingOrchestrator:
    """Builds the core ModelingOrchestrator LightningModule."""
    _LOGGER.info("Building ModelingOrchestrator...")

    tokenizer = datamodule.tokenizer
    vocab_size = len(tokenizer.char2id)
    num_classes = len(tokenizer.diacritic2id)
    pad_idx = tokenizer.char2id.get("<PAD>", 0)
    space_idx = tokenizer.char2id.get(
        " ", -1
    )  # Use -1 as a sentinel if space is not in vocab
    no_diacritic_idx = tokenizer.diacritic2id.get("", None)
    _LOGGER.info(
        f"Derived from tokenizer: vocab_size={vocab_size}, num_classes={num_classes}, "
        f"pad_idx={pad_idx}, space_idx={space_idx}, no_diacritic_idx={no_diacritic_idx}"
    )

    # The builder no longer needs to know about specific models like BiLSTM.
    # It just delegates the task to the factory.
    architecture = build_model(
        config=modeling_config["architecture"],
        vocab_size=vocab_size,
        num_classes=num_classes,
        pad_idx=pad_idx,
    )

    loss_config = modeling_config.get("loss", {})
    loss_config["ignore_index"] = pad_idx
    loss_fn = get_loss(loss_config)
    _LOGGER.info(f"Built loss function: {loss_config.get('name', 'cross_entropy')}")

    metrics = get_metrics(
        num_classes=num_classes,
        ignore_index=pad_idx,
        space_idx=space_idx,
        no_diacritic_idx=no_diacritic_idx,
    )
    _LOGGER.info("Built metrics suite: char_acc, f1_macro, der, word_acc, wer")

    lightning_module = ModelingOrchestrator(
        model=architecture,
        loss_fn=loss_fn,
        metrics=metrics,
        optimizer_cfg=modeling_config["optimizer"],
        scheduler_cfg=modeling_config.get("scheduler"),
        reset_optimizer_and_scheduler=reset_optimizer_and_scheduler,
    )
    lightning_module.tokenizer = tokenizer
    _LOGGER.info(
        "Tokenizer has been attached to the LightningModule to be saved in the checkpoint."
    )

    _LOGGER.info("ModelingOrchestrator built successfully.")

    return lightning_module


def build_trainer(trainer_config: Dict[str, Any]) -> L.Trainer:
    """Builds the Lightning Trainer."""
    _LOGGER.info("Building Trainer...")

    config = trainer_config.copy()
    loggers = get_loggers(config.pop("loggers", None))
    callbacks = get_callbacks(config.pop("callbacks", None))

    trainer = L.Trainer(
        logger=loggers,
        callbacks=callbacks,
        **config,
    )
    _LOGGER.info("Trainer built successfully.")
    return trainer


def load_lightning_module_from_checkpoint(
    ckpt_path: str,
) -> ModelingOrchestrator:
    """
    Loads a trained ModelingOrchestrator from a checkpoint file in a self-contained way.

    This function first loads the configuration saved inside the checkpoint, then uses
    that configuration to reconstruct the necessary components (like the tokenizer)
    before loading the model weights.

    Args:
        ckpt_path: Path to the .ckpt file.

    Returns:
        The fully loaded ModelingOrchestrator instance, ready for inference or export.
    """
    _LOGGER.info(f"Loading Lightning module from checkpoint: {ckpt_path}")

    checkpoint = torch.load(
        ckpt_path, map_location=torch.device("cpu"), weights_only=False
    )

    try:
        config = checkpoint["hyper_parameters"]["config"]
    except KeyError:
        raise KeyError(
            "Could not find a 'config' dictionary in the checkpoint's hyperparameters. "
            "Ensure the config is saved via `lightning_module.hparams.config = self.config` during training."
        )

    # We use a dummy file list as we only need the tokenizer's vocabulary, not the data itself.
    datamodule = build_data_manager(config["data"])
    datamodule.setup(
        stage="predict"
    )  # 'predict' stage is lightweight and initializes the tokenizer

    # This ensures the architecture perfectly matches the saved weights.
    lightning_module_structure = build_modeling_orchestrator(
        modeling_config=config["modeling_config"], datamodule=datamodule
    )

    # We pass the newly constructed model object to the loader.
    loaded_module = ModelingOrchestrator.load_from_checkpoint(
        ckpt_path,
        map_location=torch.device("cpu"),
        # Strict is false to avoid issues with hparams like 'config' not being a simple type
        strict=False,
        # Pass the constructed objects to populate the loaded module
        model=lightning_module_structure.model,
        loss_fn=lightning_module_structure.loss_fn,
        metrics=lightning_module_structure.metrics,
    )

    loaded_module.tokenizer = datamodule.tokenizer
    # Also attach the config for convenience
    loaded_module.hparams.config = config

    _LOGGER.info("Successfully loaded model and tokenizer from checkpoint.")
    return loaded_module
