"""Typer CLI — `localgate serve`, `localgate keys create`, etc."""
import typer
import uvicorn

app = typer.Typer(help="localgate: a local-first API gateway for open-source LLMs.")


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    """Start the API gateway server."""
    uvicorn.run("localgate.app:create_app", host=host, port=port, reload=reload, factory=True)


@app.command()
def version() -> None:
    """Print the installed version."""
    from localgate import __version__
    typer.echo(__version__)


if __name__ == "__main__":
    app()
