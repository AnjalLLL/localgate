"""Console color themes for `localgate code`.

Rich resolves a markup style like ``[bold green]`` by checking the console's
theme for that literal string *before* falling back to parsing it as color
names (`Console.get_style`) — so overriding the handful of literal style
strings this REPL actually uses (grep-verified against `agent/repl.py` and
`agent/render.py`) is enough to give `/theme light` a real, visible effect
without threading semantic style names through every call site.
"""

from __future__ import annotations

from rich.console import Console
from rich.theme import Theme as RichTheme

THEMES: tuple[str, ...] = ("dark", "light", "none")

#: "dark" intentionally matches Rich's own default color resolution — it's
#: listed for symmetry/documentation, not because it changes anything.
_STYLES: dict[str, dict[str, str]] = {
    "dark": {
        "bold green": "bold green",
        "bold": "bold",
        "cyan": "cyan",
        "dim": "dim",
        "red": "red",
        "yellow": "yellow",
    },
    "light": {
        "bold green": "bold dark_green",
        "bold": "bold black",
        "cyan": "blue",
        "dim": "grey42",
        "red": "dark_red",
        "yellow": "dark_orange3",
    },
}

_SYNTAX_THEME = {"dark": "ansi_dark", "light": "ansi_light", "none": "ansi_dark"}


def make_console(theme_name: str, *, no_color: bool = False) -> Console:
    """A `Console` for the given theme — `no_color` (or `theme_name == "none"`)
    disables styling outright, which is also what `NO_COLOR`/`--no-color` map to.
    """
    if no_color or theme_name == "none":
        return Console(no_color=True)
    styles = _STYLES.get(theme_name, _STYLES["dark"])
    return Console(theme=RichTheme(styles))


def syntax_theme_for(theme_name: str) -> str:
    """The Pygments/Rich `Syntax` theme name for new-file previews in diffs."""
    return _SYNTAX_THEME.get(theme_name, "ansi_dark")
