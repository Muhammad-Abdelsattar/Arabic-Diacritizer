import logging
from typing import Dict, Any

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
    modeling_config: Dict[str, Any], datamodule: DataManager
) -> ModelingOrchestrator:
    """Builds the core ModelingOrchestrator LightningModule."""
    _LOGGER.info("Building ModelingOrchestrator...")

    # 1. Get dynamic parameters from the data's tokenizer
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

    # 2. Build the neural network architecture using the model factory
    # The builder no longer needs to know about specific models like BiLSTM.
    # It just delegates the task to the factory.
    architecture = build_model(
        config=modeling_config["architecture"],
        vocab_size=vocab_size,
        num_classes=num_classes,
        pad_idx=pad_idx,
    )

    # 3. Build the loss function
    loss_config = modeling_config.get("loss", {})
    loss_config["ignore_index"] = pad_idx
    loss_fn = get_loss(loss_config)
    _LOGGER.info(f"Built loss function: {loss_config.get('name', 'cross_entropy')}")

    # 4. Build the metrics dictionary
    metrics = get_metrics(
        num_classes=num_classes,
        ignore_index=pad_idx,
        space_idx=space_idx,
        no_diacritic_idx=no_diacritic_idx,
    )
    _LOGGER.info("Built metrics suite: char_acc, f1_macro, der, word_acc, wer")

    # 5. Build the final LightningModule orchestrator
    lightning_module = ModelingOrchestrator(
        model=architecture,
        loss_fn=loss_fn,
        metrics=metrics,
        optimizer_cfg=modeling_config["optimizer"],
        scheduler_cfg=modeling_config.get("scheduler"),
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
