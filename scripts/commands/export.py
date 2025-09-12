from typing import Optional

import typer
from omegaconf import OmegaConf

from scripts.utils import load_config
from arabic_diacritizer.training import builder
from arabic_diacritizer.exporter import export_for_inference

app = typer.Typer()

@app.callback(invoke_without_command=True)
def export(
    ckpt_path: str = typer.Argument(..., help="Path to the model checkpoint (.ckpt) file."),
    experiment: Optional[str] = typer.Option(
        None, "-e", "--experiment", help="Path to the experiment config used for training."
    ),
):
    """
    Export a trained model checkpoint to ONNX and other artifacts for inference.
    """
    config = load_config(experiment_config_path=experiment)
    config_dict = OmegaConf.to_container(config, resolve=True)
    
    typer.echo(f"Loading model from checkpoint: {ckpt_path}")
    lightning_module = builder.load_lightning_module_from_checkpoint(
        ckpt_path=ckpt_path,
        config=config_dict,
    )
    
    core_model = lightning_module.model
    tokenizer = lightning_module.tokenizer
    export_params = config_dict.get("export", {})

    typer.echo(f"Exporting model to: {export_params.get('output_dir', 'N/A')}")
    export_for_inference(model=core_model, tokenizer=tokenizer, **export_params)
    typer.echo("Export complete.")
