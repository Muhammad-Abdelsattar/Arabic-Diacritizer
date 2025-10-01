import os
from pathlib import Path
from typing import Tuple

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

from .exceptions import ModelNotFound

DEFAULT_HUB_REPO_ID = "muhammad7777/arabic-diacritizer-models"


def _download_from_hub(
    repo_id: str, architecture: str, size: str, revision: str, force_sync: bool
) -> Tuple[str, str]:
    """
    Internal helper to download model artifacts from the Hub, using a nested subfolder.
    """
    # Construct the nested path (e.g., "bilstm/medium")
    model_subfolder = f"{architecture}/{size}"

    try:
        # Attempt to load from cache first (offline-first)
        if not force_sync:
            onnx_path = hf_hub_download(
                repo_id=repo_id,
                filename="model.onnx",
                subfolder=model_subfolder,
                revision=revision,
                local_files_only=True,
            )
            vocab_path = hf_hub_download(
                repo_id=repo_id,
                filename="vocab.json",
                subfolder=model_subfolder,
                revision=revision,
                local_files_only=True,
            )
            return onnx_path, vocab_path
    except LocalEntryNotFoundError:
        pass  # Not found in cache, proceed to download.

    # Download from the Hub (online fallback)
    try:
        onnx_path = hf_hub_download(
            repo_id=repo_id,
            filename="model.onnx",
            subfolder=model_subfolder,
            revision=revision,
            force_download=force_sync,
        )
        vocab_path = hf_hub_download(
            repo_id=repo_id,
            filename="vocab.json",
            subfolder=model_subfolder,
            revision=revision,
            force_download=force_sync,
        )
        return onnx_path, vocab_path
    except EntryNotFoundError as e:
        # Make the error message more informative
        raise ModelNotFound(
            f"Could not find model for architecture '{architecture}' and size '{size}' "
            f"at revision '{revision}' in repository '{repo_id}'. "
            f"Please check the Hub for available models."
        ) from e
    except Exception as e:
        raise ModelNotFound(
            f"Failed to download model from the Hub. Please check your internet connection. "
            f"Original error: {e}"
        ) from e


def resolve_model_path(
    model_identifier: str, architecture: str, size: str, revision: str, force_sync: bool
) -> Tuple[Path, Path]:
    """
    Resolves model artifact paths, now with architecture awareness.
    """
    # Case 1: The identifier is a local directory path (no change here)
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
    # Delegate the download logic, passing the new architecture parameter.
    onnx_path_str, vocab_path_str = _download_from_hub(
        repo_id=model_identifier,
        architecture=architecture,
        size=size,
        revision=revision,
        force_sync=force_sync,
    )

    return Path(onnx_path_str), Path(vocab_path_str)
