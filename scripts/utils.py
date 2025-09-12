import os
import sys
from typing import List, Optional

from omegaconf import OmegaConf, DictConfig

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def add_project_root_to_path():
    """Adds the project root to the Python path for absolute imports."""
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)


def load_config(
    base_config_path: str = os.path.join(PROJECT_ROOT, "configs/base.yaml"),
    experiment_config_path: Optional[str] = None,
    cli_overrides: Optional[List[str]] = None,
) -> DictConfig:
    """
    Loads and merges configurations from base, experiment, and CLI.

    Args:
        base_config_path: Path to the base YAML configuration.
        experiment_config_path: Optional path to an experiment override YAML file.
        cli_overrides: Optional list of 'dot-list' strings for CLI overrides.

    Returns:
        The final merged OmegaConf DictConfig object.
    """
    # Load base configuration
    conf = OmegaConf.load(base_config_path)

    # Merge with experiment-specific configuration if provided
    if experiment_config_path:
        exp_conf = OmegaConf.load(experiment_config_path)
        conf = OmegaConf.merge(conf, exp_conf)

    #  Merge with command-line overrides (highest precedence)
    if cli_overrides:
        cli_conf = OmegaConf.from_dotlist(cli_overrides)
        conf = OmegaConf.merge(conf, cli_conf)

    return conf
