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
    ckpt_path: str, config: Dict[str, Any]
) -> ModelingOrchestrator:
    """
    Loads a trained ModelingOrchestrator from a checkpoint file using a config dict.

    This function rebuilds the model structure from the provided configuration
    and then loads the saved weights from the checkpoint file into it.

    Args:
        ckpt_path: Path to the .ckpt file.
        config: The configuration dictionary used during training.

    Returns:
        The fully loaded ModelingOrchestrator instance, ready for inference.
    """
    _LOGGER.info(f"Loading Lightning module from checkpoint: {ckpt_path}")

    # Build the DataManager just to get the tokenizer initialized.
    # We use a dummy file list as we only need the tokenizer's vocabulary,
    # which is built from constants, not the data itself.
    datamodule = build_data_manager(config["data"])
    datamodule.setup(stage="predict")  # 'predict' stage is lightweight

    # Build the model structure using the configuration
    lightning_module = build_modeling_orchestrator(
        modeling_config=config["modeling_config"], datamodule=datamodule
    )

    # Load the weights from the checkpoint into the model structure.
    # The 'map_location' is important for loading a GPU-trained model onto a CPU.
    # We must pass the newly constructed model object to the loader.
    loaded_module = ModelingOrchestrator.load_from_checkpoint(
        ckpt_path,
        map_location=torch.device("cpu"),
        # Strict is false to avoid issues with tokenizer not being in the checkpoint hparams
        strict=False,
        # Pass the constructed objects to populate the loaded module
        model=lightning_module.model,
        loss_fn=lightning_module.loss_fn,
        metrics=lightning_module.metrics,
    )

    # Manually attach the tokenizer, as it's not saved in the checkpoint
    loaded_module.tokenizer = datamodule.tokenizer

    _LOGGER.info("Successfully loaded model and tokenizer from checkpoint.")
    return loaded_module
