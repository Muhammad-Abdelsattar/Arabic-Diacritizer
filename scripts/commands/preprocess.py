from typing import List
from pathlib import Path
import typer

from arabic_diacritizer.data import DatasetPreprocessor

# Create a new typer application for this command
app = typer.Typer(
    # Add some settings to make the help text cleaner
    context_settings={"help_option_names": ["-h", "--help"]}
)


@app.callback(invoke_without_command=True)
def preprocess(
    input_file: List[Path] = typer.Option(
        ...,
        "--input",
        "-i",
        help="Path to a raw text file to process. Can be provided multiple times.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    output_file: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="The path to save the cleaned, processed text file.",
        writable=True,
        prompt="Please enter the path for the output file",  # Makes it interactive if not provided
    ),
    max_chars: int = typer.Option(
        512, "--max-chars", help="Maximum sentence length before segmenting."
    ),
    chunk_size: int = typer.Option(
        1000, "--chunk-size", help="Number of lines to process in memory at once."
    ),
    validate: bool = typer.Option(
        True,
        "--validate/--no-validate",
        help="Validate that diacritics are well-formed. Use --no-validate to disable.",
    ),
    filter_non_arabic: bool = typer.Option(
        True,
        "--filter/--no-filter",
        help="Filter out non-Arabic characters. Use --no-filter to disable.",
    ),
):
    """
    Processes raw text corpora into a single, clean file for training.
    All parameters are provided directly as command-line options.
    """
    typer.echo("|> Starting Data Preprocessing (CLI Mode) ")

    typer.echo("Initializing DatasetPreprocessor with the following settings:")
    typer.echo(f"  - Max Chars: {max_chars}")
    typer.echo(f"  - Validate Diacritics: {validate}")
    typer.echo(f"  - Filter Non-Arabic: {filter_non_arabic}")
    typer.echo(f"  - Chunk Size: {chunk_size}")

    preprocessor = DatasetPreprocessor(
        max_chars=max_chars,
        validate=validate,
        filter_non_arabic=filter_non_arabic,
        chunk_size=chunk_size,
    )

    typer.echo(f"Processing raw files: {[str(f) for f in input_file]}")
    typer.echo(f"Output will be saved to: {output_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    preprocessor.process_corpus_to_file(
        corpus_paths=[str(f) for f in input_file],
        output_file=str(output_file),
        overwrite=True,
    )

    typer.secho(
        f"✔ Preprocessing complete. Clean data saved to '{output_file}'.",
        fg=typer.colors.GREEN,
    )

