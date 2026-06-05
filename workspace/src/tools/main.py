import typer
from commands import detector, extract, video2img, viewer

app = typer.Typer(help="RnD CLI Tools")
app.add_typer(video2img.app, name="video2img")
app.add_typer(detector.app, name="detect")
app.add_typer(extract.app, name="extract")
app.add_typer(viewer.app, name="view")

if __name__ == "__main__":
    app()
