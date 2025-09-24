from typing import Optional
import typer
from omegaconf import OmegaConf

from scripts.utils import load_config, setup_wandb
from arabic_diacritizer.training.training_pipeline import TrainingPipeline

app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]})


@app.callback(invoke_without_command=True)
def train(
    ctx: typer.Context,
    ckpt_path: Optional[str] = typer.Option(
        None,
        "--ckpt-path",
        "-c",
        help="Path to a checkpoint to start training from (either resume or finetune).",
    ),
    finetune: bool = typer.Option(
        False,
        "-f",
        "--finetune",
        help="Start a new run from the checkpoint (epoch 0, new optimizer). If not set, resumes training state.",
    ),
    experiment: Optional[str] = typer.Option(
        None,
        "-e",
        "--experiment",
        help="Path to an experiment file to override base configs.",
    ),
):
    """
    Train a new model, resume a previous run, or finetune from existing weights.
    """
    if finetune and not ckpt_path:
        typer.secho(
            "Error: The --finetune flag requires a --ckpt-path.", fg=typer.colors.RED
        )
        raise typer.Exit(code=1)

    config = load_config(
        resume_from_checkpoint=ckpt_path,
        experiment_config_path=experiment,
        cli_overrides=ctx.args,
    )

    setup_wandb(config)

    typer.echo("---------------------------------------------\n")
    typer.echo(" Final Merged Configuration for this Run \n")
    typer.echo("---------------------------------------------\n")
    typer.echo(OmegaConf.to_yaml(config))
    typer.echo("---------------------------------------------\n")

    config_dict = OmegaConf.to_container(config, resolve=True)

    pipeline = TrainingPipeline(config_dict)

    if ckpt_path and finetune:
        # Finetune Workflow
        pipeline.finetune(ckpt_path=ckpt_path)
    else:
        # a New Run or Resume Workflow
        pipeline.run(ckpt_path=ckpt_path)
