from typing import List, Optional

import typer
from omegaconf import OmegaConf

from scripts.utils import load_config
from arabic_diacritizer.training.training_pipeline import TrainingPipeline

app = typer.Typer()


@app.callback(invoke_without_command=True)
def train(
    ctx: typer.Context,
    experiment: Optional[str] = typer.Option(
        None, "-e", "--experiment", help="Path to the experiment override config file."
    ),
):
    """
    Train a new model from scratch based on the configuration.
    You can override any config value from the CLI, e.g.:
    `... train -e conf.yaml data.batch_size=32`
    """
    # The 'ctx.args' will capture all extra arguments for OmegaConf
    config = load_config(experiment_config_path=experiment, cli_overrides=ctx.args)

    typer.echo("--- Final Training Configuration ---")
    typer.echo(OmegaConf.to_yaml(config))

    # Convert to a plain dictionary for the pipeline
    config_dict = OmegaConf.to_container(config, resolve=True)

    pipeline = TrainingPipeline(config_dict)
    pipeline.run()

if __name__ == "__main__":
    typer.echo("Error: This script is not meant to be run directly.", err=True)
    typer.echo("Please use the main entry point: `python scripts/run.py train [OPTIONS]`", err=True)
    exit(1)
