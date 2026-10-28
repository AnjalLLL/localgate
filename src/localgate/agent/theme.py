"""Console color themes for `localgate code`.

A theme is a small palette (accent, secondary, success, warning, error, muted).
Two things are derived from it:

* a Rich theme. Rich resolves a markup style like ``[bold green]`` by checking the
  console's theme for that literal string *before* parsing it as color names
  (`Console.get_style`), so overriding the literal style strings the REPL uses —
  plus Rich's own Markdown, table, rule and spinner styles — recolors everything
  without threading semantic style names through every call site;
* a prompt_toolkit style for the input line, its frame, the completion menu and
  the bottom toolbar, which Rich never draws.

A terminal program owns its background color, so a theme can't change it: "light"
means dark text meant for a light-background terminal, not a light background.
"""

from __future__ import annotations

from dataclasses import dataclass

from prompt_toolkit.styles import Style
from rich.console import Console
from rich.theme import Theme as RichTheme

THEMES: tuple[str, ...] = ("sunrise", "dark", "light", "none")
DEFAULT_THEME = "sunrise"


@dataclass(frozen=True)
class Palette:
    accent: str
    secondary: str
    success: str
    warning: str
    error: str
    muted: str
    text: str
    #: Background of the completion menu, so it reads as a menu on either background.
    menu_bg: str
    syntax: str


_PALETTES: dict[str, Palette] = {
    # The dashboard's "sunrise" palette (amber accent, blue second color), for a
    # dark-background terminal.
    "sunrise": Palette(
        accent="#ffb454",
        secondary="#78b1f0",
        success="#7fd4a0",
        warning="#ffc477",
        error="#ff7b72",
        muted="#8b8f94",
        text="#e8e6e3",
        menu_bg="#2a2420",
        syntax="ansi_dark",
    ),
    # Rich's own colors; the original look.
    "dark": Palette(
        accent="green",
        secondary="cyan",
        success="green",
        warning="yellow",
        error="red",
        muted="bright_black",
        text="default",
        menu_bg="#303030",
        syntax="ansi_dark",
    ),
    # Darker shades that stay readable on a white or light-gray terminal.
    "light": Palette(
        accent="#a8590e",
        secondary="#1d5fa8",
        success="#19794a",
        warning="#9a5b00",
        error="#b3372b",
        muted="#6b7075",
        text="#15181b",
        menu_bg="#ece7e1",
        syntax="ansi_light",
    ),
}


def palette_for(theme_name: str) -> Palette:
    return _PALETTES.get(theme_name, _PALETTES[DEFAULT_THEME])


def _rich_styles(p: Palette) -> dict[str, str]:
    return {
        # Literal style strings used by repl.py / render.py.
        "bold green": f"bold {p.accent}",
        "green": p.accent,
        "cyan": p.secondary,
        "bold": f"bold {p.text}" if p.text != "default" else "bold",
        "dim": p.muted,
        "dim strike": f"{p.muted} strike",
        "red": p.error,
        "yellow": p.warning,
        "bold yellow": f"bold {p.warning}",
        # Rich's built-in styles behind Markdown answers, tables, rules and spinners.
        "markdown.h1": f"bold {p.accent}",
        "markdown.h2": f"bold {p.accent}",
        "markdown.h3": f"bold {p.accent}",
        "markdown.h4": f"bold {p.secondary}",
        "markdown.code": f"bold {p.secondary}",
        "markdown.link": p.secondary,
        "markdown.link_url": p.muted,
        "markdown.item.bullet": f"bold {p.accent}",
        "markdown.item.number": f"bold {p.accent}",
        "markdown.block_quote": p.muted,
        "markdown.hr": p.muted,
        "table.header": f"bold {p.accent}",
        "rule.line": p.muted,
        "status.spinner": p.accent,
        "repr.number": p.secondary,
        "repr.str": p.success,
    }


def make_console(theme_name: str, *, no_color: bool = False) -> Console:
    """A `Console` for the given theme — `no_color` (or `theme_name == "none"`)
    disables styling outright, which is also what `NO_COLOR`/`--no-color` map to.
    """
    if no_color or theme_name == "none":
        return Console(no_color=True)
    return Console(theme=RichTheme(_rich_styles(palette_for(theme_name))))


def prompt_style(theme_name: str, *, no_color: bool = False) -> Style:
    """The prompt_toolkit style for the input area and toolbar."""
    if no_color or theme_name == "none":
        return Style.from_dict({"bottom-toolbar": "noreverse", "frame": "", "prompt": "bold"})
    p = palette_for(theme_name)

    def fg(color: str) -> str:
        return "" if color == "default" else color.replace("bright_black", "ansibrightblack")

    return Style.from_dict(
        {
            "prompt": f"bold {fg(p.accent)}",
            "frame": fg(p.muted),
            "bottom-toolbar": f"noreverse {fg(p.muted)}",
            "bottom-toolbar.model": f"bold {fg(p.accent)}",
            "completion-menu": f"bg:{p.menu_bg} {fg(p.text)}",
            "completion-menu.completion.current": f"bg:{fg(p.accent)} #000000 bold",
            "completion-menu.meta.completion": f"bg:{p.menu_bg} {fg(p.muted)}",
            "completion-menu.meta.completion.current": f"bg:{fg(p.accent)} #000000",
            "scrollbar.background": f"bg:{p.menu_bg}",
            "scrollbar.button": f"bg:{fg(p.muted)}",
        }
    )


def syntax_theme_for(theme_name: str) -> str:
    """The Pygments/Rich `Syntax` theme name for code blocks and new-file previews."""
    return palette_for(theme_name).syntax
