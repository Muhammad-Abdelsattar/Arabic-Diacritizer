import logging
from typing import List, Dict, Any, Optional

from lightning.pytorch.callbacks import (
    Callback,
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    DeviceStatsMonitor,
    RichProgressBar,
)

_LOGGER = logging.getLogger(__name__)

SUPPORTED_CALLBACKS = {
    "model_checkpoint": ModelCheckpoint,
    "early_stopping": EarlyStopping,
    "lr_monitor": LearningRateMonitor,
    "device_stats": DeviceStatsMonitor,
    "rich_progress_bar": RichProgressBar,
}


def get_callbacks(config: Optional[List[Dict[str, Any]]]) -> List[Callback]:
    """
    Builds a list of PyTorch Lightning callbacks from a configuration list.
    """
    if not config:
        _LOGGER.info("No callbacks configured.")
        return []

    callbacks: List[Callback] = []
    for callback_conf in config:
        conf = callback_conf.copy()
        callback_name = conf.pop("name", None)

        if not callback_name:
            _LOGGER.warning(
                "Found a callback configuration without a 'name' key. Skipping."
            )
            continue

        if callback_name not in SUPPORTED_CALLBACKS:
            raise ValueError(
                f"Unknown callback '{callback_name}'. "
                f"Supported callbacks are: {list(SUPPORTED_CALLBACKS.keys())}"
            )

        try:
            callback_class = SUPPORTED_CALLBACKS[callback_name]
            # Instantiate the callback with the remaining config as kwargs
            instance = callback_class(**conf)
            callbacks.append(instance)
            _LOGGER.info(
                f"Successfully instantiated callback: {callback_name} with params {conf}"
            )
        except Exception as e:
            _LOGGER.error(
                f"Failed to instantiate callback '{callback_name}' with config {conf}. Error: {e}"
            )
            raise

    return callbacks
