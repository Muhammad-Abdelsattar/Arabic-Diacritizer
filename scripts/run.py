import typer

# from scripts.utils import add_project_root_to_path
#
# # add_project_root_to_path()

from scripts.commands import train, evaluate, export, preprocess

# Create the main Typer application
app = typer.Typer(
    name="arabic-diacritizer-cli",
    help="A CLI for training, evaluating, and exporting Arabic diacritization models.",
    add_completion=False,
    pretty_exceptions_show_locals=False,
)

# Register the sub-commands
app.add_typer(train.app, name="train")
app.add_typer(evaluate.app, name="evaluate")
app.add_typer(export.app, name="export")
app.add_typer(preprocess.app, name="preprocess")


if __name__ == "__main__":
    app()
