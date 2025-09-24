from typing import List
from pathlib import Path

import typer
import lightning as L

from arabic_diacritizer.data import DataManager
from arabic_diacritizer.training import builder

app = typer.Typer(
    help="Evaluate a model checkpoint on a given dataset.",
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.callback(invoke_without_command=True)
def evaluate(
    ckpt_path: Path = typer.Option(
        ...,
        "--ckpt-path",
        "-c",
        help="Path to the model checkpoint (.ckpt) file to evaluate.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    data_files: List[Path] = typer.Option(
        ...,
        "--data-file",
        "-d",
        help="Path to a text file for evaluation. Can be provided multiple times.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    batch_size: int = typer.Option(
        128, "--batch-size", help="Batch size for the evaluation data loader."
    ),
    num_workers: int = typer.Option(
        4, "--num-workers", help="Number of workers for the data loader."
    ),
    max_length: int = typer.Option(
        512,
        "--max-length",
        help="Maximum sequence length. Sentences will be padded/truncated to this length.",
    ),
    accelerator: str = typer.Option(
        "auto",
        help="Hardware accelerator to use ('cpu', 'gpu', 'auto').",
    ),
    devices: int = typer.Option(
        1, "--devices", help="Number of devices to use (e.g., number of GPUs)."
    ),
):
    """
    Evaluate a trained model checkpoint on one or more specified text files.

    This script is self-contained. It loads the model architecture and tokenizer
    directly from the checkpoint, ignoring any external YAML configuration files.
    """

    typer.echo(f"| > Loading model and tokenizer from: {ckpt_path}")
    try:
        lightning_module = builder.load_lightning_module_from_checkpoint(
            ckpt_path=str(ckpt_path),
        )
        lightning_module.eval()
        typer.echo("| > Model loaded successfully.")
    except Exception as e:
        typer.secho(
            f"Error: Failed to load checkpoint. Reason: {e}", fg=typer.colors.RED
        )
        raise typer.Exit(code=1)

    typer.echo("| > Preparing evaluation dataset...")
    typer.echo(f"| > Max sequence length: {max_length}")
    str_data_files = [str(f) for f in data_files]
    datamodule = DataManager(
        # The train_files argument is required, but we won't use it.
        # We pass our data to `test_files` which will be used in the 'test' stage.
        train_files=str_data_files,
        test_files=str_data_files,
        batch_size=batch_size,
        num_workers=num_workers,
        max_length=max_length,
        cache_dir=".cache/",
        cache_format="npz",  # Use npz for efficiency
    )

    # Attach the tokenizer from the loaded model to the datamodule
    datamodule.tokenizer = lightning_module.tokenizer

    # Run the setup for the 'test' stage only. This will create the test_dataset.
    datamodule.setup(stage="test")
    typer.echo(f"| > Data loaded with {len(datamodule.test_dataset)} samples.")

    # We only need a minimal trainer for evaluation. No loggers or callbacks needed.
    typer.echo("| > Configuring trainer...")
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=False,
    )

    typer.echo("\nRunning Evaluation ... ")
    # The `trainer.test` method will automatically use the test dataloader
    # from our datamodule and print the results to the console.
    trainer.test(model=lightning_module, datamodule=datamodule, verbose=True)

    typer.secho("\n✔ Evaluation complete.", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
