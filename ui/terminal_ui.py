"""
Terminal UI using Rich.

Provides a live dashboard showing:
  - Current running count and true count
  - Recommended bet size
  - Predicted EV
  - Bet spread visualization
  - Last N cards seen
  - Gauss-Markov model quality indicators
  - Training progress
"""

from __future__ import annotations
from typing import List, Optional, TYPE_CHECKING
import os

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    from rich.layout import Layout
    from rich.text import Text
    from rich.rule import Rule
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

if TYPE_CHECKING:
    from ml.betting_strategy import BetRecommendation
    from counting.base_counter import BaseCounter
    from simulator.deck import Card

console = Console() if RICH_AVAILABLE else None


def _tc_color(tc: float) -> str:
    """Color-code true count: green = good, red = bad."""
    if tc >= 3:
        return "bold green"
    elif tc >= 1:
        return "green"
    elif tc >= -1:
        return "yellow"
    elif tc >= -3:
        return "orange3"
    else:
        return "bold red"


def _ev_color(ev: float) -> str:
    if ev > 0:
        return "bold green"
    elif ev > -0.01:
        return "yellow"
    else:
        return "red"


def print_header() -> None:
    """Print Rain Man ASCII header."""
    if not RICH_AVAILABLE:
        print("=" * 55)
        print("  RAIN MAN — Blackjack Card Counter")
        print("  Based on 'The World's Greatest Blackjack Book'")
        print("=" * 55)
        return

    console.print()
    console.print(Panel.fit(
        "[bold cyan]RAIN MAN[/bold cyan] — Blackjack Card Counting System\n"
        "[dim]Monte Carlo + OLS + Gauss-Markov | Hi-Lo / KO / Omega II[/dim]",
        border_style="cyan",
    ))


def print_training_progress(step: str, pct: float) -> None:
    if not RICH_AVAILABLE:
        print(f"[{pct:.0f}%] {step}")
        return
    console.print(f"  [dim][{pct:>3.0f}%][/dim] {step}")


def print_bet_recommendation(rec: "BetRecommendation") -> None:
    """Print the current bet recommendation panel."""
    if not RICH_AVAILABLE:
        print(str(rec))
        return

    tc_col = _tc_color(rec.true_count)
    ev_col = _ev_color(rec.predicted_ev)

    edge_str = (
        f"[{ev_col}]{rec.predicted_ev:+.2%}[/{ev_col}]"
        if rec.player_has_edge
        else f"[{ev_col}]{rec.predicted_ev:+.2%}[/{ev_col}] (house)"
    )

    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    table.add_column("Label", style="dim", width=22)
    table.add_column("Value", width=25)

    table.add_row("Running Count",
                  f"[bold]{rec.running_count:+d}[/bold]")
    table.add_row("True Count",
                  f"[{tc_col}]{rec.true_count:+.2f}[/{tc_col}]")
    table.add_row("Predicted EV", edge_str)
    table.add_row("Decks Remaining",
                  f"{rec.decks_remaining:.1f}")
    table.add_row("Shoe Penetration",
                  f"{rec.penetration:.0%}")
    table.add_row("─" * 20, "─" * 23)
    table.add_row(
        "[bold]RECOMMENDED BET[/bold]",
        f"[bold cyan]${rec.recommended_bet:.0f}[/bold cyan]  "
        f"[dim]({rec.bet_units:.0f}u)[/dim]"
    )
    table.add_row("Kelly Bet",
                  f"${rec.kelly_bet:.0f}")
    table.add_row("Model Confidence",
                  f"{rec.confidence:.4f} R²")

    border = "green" if rec.player_has_edge else "red"
    console.print(Panel(table, title="[bold]BET RECOMMENDATION[/bold]",
                        border_style=border))

    if rec.index_plays:
        console.print("[bold yellow]INDEX PLAYS ACTIVE:[/bold yellow]")
        for play in rec.index_plays[:5]:
            console.print(f"  [yellow]↳ {play['play']}: {play['note']} "
                          f"(TC≥{play['threshold']})[/yellow]")


def print_cards_seen(cards: List["Card"], last_n: int = 10) -> None:
    """Print the last N cards seen."""
    if not RICH_AVAILABLE:
        print("Cards:", " ".join(str(c) for c in cards[-last_n:]))
        return

    recent = cards[-last_n:]
    card_texts = []
    for c in recent:
        if c.suit in ("♥", "♦"):
            card_texts.append(f"[red]{c}[/red]")
        else:
            card_texts.append(f"[white]{c}[/white]")

    console.print(f"[dim]Last {len(recent)} cards:[/dim] " + "  ".join(card_texts))


def print_count_bar(running_count: int, true_count: float) -> None:
    """Visual count bar."""
    if not RICH_AVAILABLE:
        print(f"RC={running_count:+d}  TC={true_count:+.2f}")
        return

    # Scale: -10 to +10 → 0 to 40 chars
    bar_total = 40
    center = bar_total // 2
    tc_clamped = max(-10, min(10, true_count))
    filled = int((tc_clamped + 10) / 20 * bar_total)

    bar = "[" + "─" * min(filled, center)
    if filled >= center:
        bar += "█" * (filled - center)
    bar += " " * max(0, bar_total - max(filled, center))
    bar += "]"

    tc_col = _tc_color(true_count)
    console.print(f"  Count: [{tc_col}]{bar}[/{tc_col}] "
                  f"RC=[bold]{running_count:+d}[/bold] "
                  f"TC=[{tc_col}]{true_count:+.2f}[/{tc_col}]")


def print_ev_table(ev_curve: dict) -> None:
    """Print the EV-by-true-count table."""
    if not RICH_AVAILABLE:
        for tc, ev in ev_curve.items():
            print(f"  TC {tc:>+3}: {ev:+.4f}")
        return

    table = Table(title="EV by True Count (OLS Model)", box=box.SIMPLE_HEAVY)
    table.add_column("True Count", justify="center", style="bold")
    table.add_column("Predicted EV", justify="right")
    table.add_column("Edge", justify="left")

    for tc, ev in sorted(ev_curve.items()):
        if -4 <= tc <= 8:
            bar_len = int(abs(ev) * 300)
            bar = ("+" if ev >= 0 else "─") * min(bar_len, 20)
            col = _ev_color(ev)
            table.add_row(
                f"TC {tc:>+3}",
                f"[{col}]{ev:+.4f}[/{col}]",
                f"[{col}]{bar}[/{col}]",
            )
    console.print(table)


def print_simulation_progress(hand_num: int, total: int, edge: float) -> None:
    pct = hand_num / total * 100
    if not RICH_AVAILABLE:
        if hand_num % 10000 == 0:
            print(f"  {pct:.0f}% | edge: {edge:+.3f}%")
        return
    if hand_num % 10000 == 0:
        console.print(f"  [dim]{pct:>3.0f}%[/dim] "
                      f"Simulated [bold]{hand_num:,}[/bold] hands | "
                      f"Edge: [{_ev_color(edge/100)}"
                      f"]{edge:+.3f}%[/{_ev_color(edge/100)}]")


def print_rule(title: str = "") -> None:
    if RICH_AVAILABLE:
        console.print(Rule(title, style="dim"))
    else:
        print(f"── {title} " + "─" * (50 - len(title)))


def ask_mode() -> str:
    """Prompt user to choose operation mode."""
    print_header()
    if RICH_AVAILABLE:
        console.print("\n[bold]Choose mode:[/bold]")
        console.print("  [cyan]1[/cyan] → Train model (Monte Carlo simulation)")
        console.print("  [cyan]2[/cyan] → Live play (screen reader)")
        console.print("  [cyan]3[/cyan] → Live play (manual input)")
        console.print("  [cyan]4[/cyan] → Show training stats")
        console.print("  [cyan]5[/cyan] → Simulate a few hands (demo)")
    else:
        print("\nChoose mode:")
        print("  1 → Train model")
        print("  2 → Live (screen reader)")
        print("  3 → Live (manual input)")
        print("  4 → Show training stats")
        print("  5 → Demo")
    try:
        choice = input("\nChoice [1-5]: ").strip()
        return choice
    except (EOFError, KeyboardInterrupt):
        return "quit"
