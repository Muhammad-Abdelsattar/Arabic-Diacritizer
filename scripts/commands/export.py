import typer

from arabic_diacritizer.training import builder
from arabic_diacritizer.exporter import export_for_inference

app = typer.Typer()


@app.callback(invoke_without_command=True)
def export(
    ckpt_path: str = typer.Argument(
        ..., help="Path to the model checkpoint (.ckpt) file."
    ),
):
    """
    Export a trained model checkpoint to ONNX and other artifacts for inference.
    This command is self-contained and derives all necessary configuration
    from the checkpoint file itself.
    """
    typer.echo(f"Loading model and configuration from checkpoint: {ckpt_path}")

    # The new builder function requires only the checkpoint path
    lightning_module = builder.load_lightning_module_from_checkpoint(
        ckpt_path=ckpt_path,
    )

    # Extract the core components needed for export
    core_model = lightning_module.model
    tokenizer = lightning_module.tokenizer
    config = lightning_module.hparams.config

    # The export parameters are also loaded from the config inside the checkpoint
    try:
        export_params = lightning_module.hparams.config.get("export", {})
    except AttributeError:
        typer.secho(
            "Warning: Could not find 'export' configuration in the checkpoint. Using default export parameters.",
            fg=typer.colors.YELLOW,
        )
        export_params = {}

    output_dir = export_params.get("output_dir", "artifacts/")
    typer.echo(f"Exporting model to: {output_dir}")

    export_for_inference(model=core_model, tokenizer=tokenizer, config=config, **export_params)

    typer.secho("Export complete.", fg=typer.colors.GREEN)


if __name__ == "__main__":
    typer.echo("Error: This script is not meant to be run directly.", err=True)
    typer.echo(
        "Please use the main entry point: `python scripts/run.py export [CKPT_PATH]`",
        err=True,
    )
    exit(1)
