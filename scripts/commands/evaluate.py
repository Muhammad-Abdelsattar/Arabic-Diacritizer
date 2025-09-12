from typing import Optional

import typer
from omegaconf import OmegaConf

from scripts.utils import load_config
from arabic_diacritizer.training.training_pipeline import TrainingPipeline

app = typer.Typer()

@app.callback(invoke_without_command=True)
def evaluate(
    ckpt_path: str = typer.Argument(..., help="Path to the model checkpoint (.ckpt) file."),
    experiment: Optional[str] = typer.Option(
        None, "-e", "--experiment", help="Path to the experiment config. If not provided, base.yaml is used."
    ),
):
    """
    Evaluate a trained model checkpoint on the test set.
    """
    config = load_config(experiment_config_path=experiment)
    
    typer.echo("--- Using Configuration for Evaluation ---")
    typer.echo(OmegaConf.to_yaml(config))

    config_dict = OmegaConf.to_container(config, resolve=True)
    
    pipeline = TrainingPipeline(config_dict)
    pipeline.evaluate(ckpt_path=ckpt_path)


if __name__ == "__main__":
    typer.echo("Error: This script is not meant to be run directly.", err=True)
    typer.echo("Please use the main entry point: `python scripts/run.py export [OPTIONS]`", err=True)
    exit(1)
