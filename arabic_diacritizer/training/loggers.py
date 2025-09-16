import logging
from typing import List, Dict, Any, Optional

from lightning.pytorch.loggers import Logger, TensorBoardLogger, WandbLogger, CSVLogger

# Configure a basic logger for this module
_LOGGER = logging.getLogger(__name__)

SUPPORTED_LOGGERS = {
    "tensorboard": TensorBoardLogger,
    "wandb": WandbLogger,
    "csv": CSVLogger,
}


def get_loggers(config: Optional[List[Dict[str, Any]]]) -> List[Logger]:
    """
    Builds a list of PyTorch Lightning loggers from a configuration list.

    Each item in the list is a dictionary that must contain a 'name' key, which
    corresponds to a supported logger, and the rest of the keys are passed as
    arguments to the logger's constructor.

    Args:
        config: A list of logger configurations. Can be None or empty.
            Example:
            [
                {"name": "tensorboard", "save_dir": "logs/", "version": "run_01"},
                {"name": "wandb", "project": "arabic-diacritizer"}
            ]

    Returns:
        A list of instantiated `lightning.pytorch.loggers.Logger` objects.
        Returns an empty list if the config is None or empty.

    Raises:
        ValueError: If a logger name specified in the config is not supported.
    """
    if not config:
        _LOGGER.info("No loggers configured. Using default Lightning loggers.")
        return []

    loggers: List[Logger] = []
    for logger_conf in config:
        conf = logger_conf.copy()
        logger_name = conf.pop("name", None)

        if not logger_name:
            _LOGGER.warning(
                "Found a logger configuration without a 'name' key. Skipping."
            )
            continue

        if logger_name not in SUPPORTED_LOGGERS:
            raise ValueError(
                f"Unknown logger '{logger_name}'. "
                f"Supported loggers are: {list(SUPPORTED_LOGGERS.keys())}"
            )

        try:
            logger_class = SUPPORTED_LOGGERS[logger_name]
            # Instantiate the logger with the remaining config as kwargs
            logger_instance = logger_class(**conf)
            loggers.append(logger_instance)
            _LOGGER.info(
                f"Successfully instantiated logger: {logger_name} with params {conf}"
            )
        except Exception as e:
            _LOGGER.error(
                f"Failed to instantiate logger '{logger_name}' with config {conf}. Error: {e}"
            )
            # Depending on desired behavior, you could re-raise or just skip it
            raise

    return loggers
