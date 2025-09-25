import os
from pathlib import Path
from typing import Tuple

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

from .exceptions import ModelNotFound

DEFAULT_HUB_REPO_ID = "muhammad7777/arabic-diacritizer-models"


def _download_from_hub(
    repo_id: str, size: str, revision: str, force_sync: bool
) -> Tuple[str, str]:
    """
    Internal helper function to download model artifacts from the Hugging Face Hub.
    Implements a robust, offline-first strategy.
    """
    try:
        # Attempt to load from cache first (offline-first)
        # If force_sync is True, this stage is skipped, and we go directly to downloading.
        if not force_sync:
            # We set local_files_only=True to prevent any network calls.
            onnx_path = hf_hub_download(
                repo_id=repo_id,
                filename="model.onnx",
                subfolder=size,
                revision=revision,
                local_files_only=True,
            )
            vocab_path = hf_hub_download(
                repo_id=repo_id,
                filename="vocab.json",
                subfolder=size,
                revision=revision,
                local_files_only=True,
            )
            return onnx_path, vocab_path
    except LocalEntryNotFoundError:
        pass

    # Download from the Hub (online fallback)
    # This code is reached if the model is not cached or if force_sync=True.
    try:
        # We now run the same calls but with local_files_only=False (the default).
        # The library will download the files and place them in the cache.
        onnx_path = hf_hub_download(
            repo_id=repo_id,
            filename="model.onnx",
            subfolder=size,
            revision=revision,
            force_download=force_sync,  # Pass the force_sync flag
        )
        vocab_path = hf_hub_download(
            repo_id=repo_id,
            filename="vocab.json",
            subfolder=size,
            revision=revision,
            force_download=force_sync,  # Pass the force_sync flag
        )
        return onnx_path, vocab_path
    except EntryNotFoundError as e:
        raise ModelNotFound(
            f"Could not find model for size '{size}' at revision '{revision}' "
            f"in repository '{repo_id}'. Please check the Hub for available models."
        ) from e
    except Exception as e:
        raise ModelNotFound(
            f"Failed to download model from the Hub. Please check your internet connection. "
            f"Original error: {e}"
        ) from e


def resolve_model_path(
    model_identifier: str, size: str, revision: str, force_sync: bool
) -> Tuple[Path, Path]:
    """
    Resolves the paths to the model artifacts, downloading from the Hub if necessary.

    This function implements the core logic for finding model files. It checks
    if the identifier is a local path first. If not, it treats it as an HF Hub
    repository ID and delegates to the download helper function.

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
    # Delegate all the complex download logic to our internal helper function.
    onnx_path_str, vocab_path_str = _download_from_hub(
        repo_id=model_identifier,
        size=size,
        revision=revision,
        force_sync=force_sync,
    )

    return Path(onnx_path_str), Path(vocab_path_str)
