import os
import sys
from typing import List, Optional, Dict, Any

import torch
import typer
from omegaconf import OmegaConf, DictConfig

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def add_project_root_to_path():
    """Adds the project root to the Python path for absolute imports."""
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)


def load_config_from_checkpoint(ckpt_path: str) -> Dict[str, Any]:
    """
    Loads the 'config' dictionary saved in a Lightning checkpoint's hyperparameters.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        config = checkpoint["hyper_parameters"]["config"]
        if not isinstance(config, dict):
            raise ValueError(
                "Could not find a valid config dictionary in the checkpoint's hyperparameters."
            )
        typer.echo(f"[INFO] Loading config from checkpoint: {ckpt_path}")
        return config
    except KeyError:
        raise KeyError(
            "Could not find hyperparameters or the 'config' key in the checkpoint."
        )
    except Exception as e:
        print(f"Error loading config from checkpoint: {e}")
        raise


def load_config(
    resume_from_checkpoint: Optional[str] = None,
    experiment_config_path: Optional[str] = None,
    cli_overrides: Optional[List[str]] = None,
) -> DictConfig:
    """
    Loads and merges configurations in a strict hierarchical order.
    """
    typer.echo("--- Loading and Merging Configurations ---")

    base_config_path = os.path.join(PROJECT_ROOT, "configs/base.yaml")
    if not os.path.exists(base_config_path):
        raise FileNotFoundError(f"Base config file not found at: {base_config_path}")
    conf = OmegaConf.load(base_config_path)
    typer.echo(f"\t | > Loaded base config from: {base_config_path}")

    architecture_to_enforce = None

    if resume_from_checkpoint:
        ckpt_conf = OmegaConf.create(
            load_config_from_checkpoint(resume_from_checkpoint)
        )

        # This is the architecture we MUST use to avoid errors
        if (
            "modeling_config" in ckpt_conf
            and "architecture" in ckpt_conf.modeling_config
        ):
            architecture_to_enforce = ckpt_conf.modeling_config.architecture
        else:
            raise ValueError(
                "Resumed checkpoint is invalid: missing 'modeling_config.architecture'."
            )

        conf = OmegaConf.merge(conf, ckpt_conf)
        typer.echo("\t | > Merged config from checkpoint.")

    if experiment_config_path:
        if not os.path.exists(experiment_config_path):
            raise FileNotFoundError(
                f"Experiment config file not found at: {experiment_config_path}"
            )
        exp_conf = OmegaConf.load(experiment_config_path)
        conf = OmegaConf.merge(conf, exp_conf)
        typer.echo(f"\t | > Merged experiment config from: {experiment_config_path}")

    if cli_overrides:
        cli_conf = OmegaConf.from_dotlist(cli_overrides)
        conf = OmegaConf.merge(conf, cli_conf)
        typer.echo(" Merged config from command-line arguments.")

    # SAFETY CHECK: If resuming, enforce the original architecture
    if architecture_to_enforce:
        if conf.modeling_config.architecture != architecture_to_enforce:
            typer.secho(
                "\t | > SAFETY OVERRIDE: Enforcing model architecture from the checkpoint.",
                fg=typer.colors.YELLOW,
            )
            # This is the crucial step that prevents mismatches
            conf.modeling_config.architecture = architecture_to_enforce

    typer.echo("------------------------------------------\n")
    return conf
