import os
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError 

from .exceptions import ModelNotFound

DEFAULT_HUB_REPO_ID = "muhammad7777/arabic-diacritizer-models"


def resolve_model_path(
    model_identifier: str, size: str, revision: str, force_sync: bool
) -> tuple[Path, Path]:
    """
    Resolves the paths to the model artifacts, downloading from the Hub if necessary.

    This function implements the core logic for finding model files. It checks
    if the identifier is a local path first. If not, it treats it as an HF Hub
    repository ID and proceeds with downloading and caching.

    Args:
        model_identifier (str): A local path or a Hugging Face Hub repo ID.
        size (str): The model size ('small', 'medium', 'large').
        revision (str): The model revision (tag, branch, or commit hash).
        force_sync (bool): Whether to force a re-download from the Hub.

    Returns:
        A tuple containing the local paths to the onnx model and the vocab file.

    Raises:
        ModelNotFound: If the model artifacts cannot be found locally or on the Hub.
    """
    # Case 1: The identifier is a local directory path
    if os.path.isdir(model_identifier):
        model_dir = Path(model_identifier)
        onnx_path = model_dir / "model.onnx"
        vocab_path = model_dir / "vocab.json"

        if not onnx_path.exists() or not vocab_path.exists():
            raise ModelNotFound(
                f"Local model directory must contain 'model.onnx' and 'vocab.json'. "
                f"Path: {model_dir}"
            )
        return onnx_path, vocab_path

    # Case 2: The identifier is a Hugging Face Hub repository ID
    try:
        # Download the model.onnx file from the specified subfolder.
        onnx_path = hf_hub_download(
            repo_id=model_identifier,
            filename="model.onnx",
            subfolder=size,
            revision=revision,
            force_download=force_sync,
        )

        # Download the vocab.json file.
        vocab_path = hf_hub_download(
            repo_id=model_identifier,
            filename="vocab.json",
            subfolder=size,
            revision=revision,
            force_download=force_sync,
        )
        return Path(onnx_path), Path(vocab_path)

    except (EntryNotFoundError, LocalEntryNotFoundError) as e:
        raise ModelNotFound(
            f"Could not find or download model for size '{size}' at revision '{revision}' "
            f"in repository '{model_identifier}'. Please ensure the identifier is correct, "
            "the model size exists, and you are online if the model is not cached."
        ) from e
    except Exception as e:
        raise ModelNotFound(
            f"An unexpected error occurred while downloading the model: {e}"
        ) from e
