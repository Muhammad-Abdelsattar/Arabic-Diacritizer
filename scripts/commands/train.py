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
        None,
        "-e",
        "--experiment",
        help="Path to an experiment file to override base configs.",
    ),
    resume_from_checkpoint: Optional[str] = typer.Option(
        None,
        "-r",
        "--resume-from-checkpoint",
        help="Path to a .ckpt file to resume training. This loads the config from the checkpoint.",
    ),
):
    """
    Train a new model or resume training from a checkpoint.
    """
    reset_optimizer = False
    if resume_from_checkpoint and ctx.args:
        # Check if any CLI override targets the optimizer or scheduler
        if any(arg.startswith(("optimizer.", "scheduler.")) for arg in ctx.args):
            reset_optimizer = True
            typer.secho(
                "[INFO] Optimizer/scheduler config override detected. The optimizer and scheduler states will be reset.",
                fg=typer.colors.YELLOW,
            )

    # The load_config function handles all complex logic for determining the final configuration
    config = load_config(
        resume_from_checkpoint=resume_from_checkpoint,
        experiment_config_path=experiment,
        cli_overrides=ctx.args,
    )

    if resume_from_checkpoint:
        typer.secho(
            f"Resuming training from: {resume_from_checkpoint}", fg=typer.colors.GREEN
        )
    else:
        typer.secho("Starting a new training run...", fg=typer.colors.GREEN)

    typer.echo("--- Final Merged Configuration for this Run ---")
    typer.echo(OmegaConf.to_yaml(config))
    typer.echo("---------------------------------------------\n")

    # Convert to a plain dictionary for the pipeline
    config_dict = OmegaConf.to_container(config, resolve=True)

    pipeline = TrainingPipeline(config_dict)
    pipeline.run(ckpt_path=resume_from_checkpoint, reset_optimizer_and_scheduler=reset_optimizer)


if __name__ == "__main__":
    typer.echo("Error: This script is not meant to be run directly.", err=True)
    typer.echo(
        "Please use the main entry point: `python scripts/run.py train [OPTIONS]`",
        err=True,
    )
    exit(1)
