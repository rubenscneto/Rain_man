"""
Rain Man — Blackjack Card Counting ML System
Main entry point.

Usage
-----
  python main.py --mode train       # Train on Monte Carlo data (first step)
  python main.py --mode live        # Live play with screen reader
  python main.py --mode manual      # Live play with manual card input
  python main.py --mode stats       # Show training statistics
  python main.py --mode demo        # Quick demo simulation

  Options:
    --decks N          Number of decks in shoe (default: 6)
    --penetration F    Penetration 0.5–0.85 (default: 0.75)
    --system SYSTEM    Counting system: hi_lo, ko, omega2 (default: hi_lo)
    --bankroll N       Starting bankroll in units (default: 1000)
    --min-bet N        Table minimum bet (default: 10)
    --max-bet N        Table maximum bet (default: 200)
    --hands N          Hands to simulate in training (default: 100000)
    --no-screen        Disable screen reader (use manual input)
"""

from __future__ import annotations
import argparse
import sys
import os
import time
from typing import List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from config import RainManConfig, ShoeConfig, BettingConfig, MonteCarloConfig, CountingSystem
from simulator.deck import Shoe, Card
from counting import make_counter
from ml.trainer import Trainer, MODEL_PATH
from ml.betting_strategy import BettingStrategy
from screen_reader.capture import ScreenCapture
from screen_reader.card_detector import CardDetector
from ui.terminal_ui import (
    print_header, print_bet_recommendation, print_cards_seen,
    print_count_bar, print_ev_table, print_rule, ask_mode, console,
)
from ui.manual_input import ManualCardInput


def build_config(args) -> RainManConfig:
    """Build configuration from CLI arguments."""
    cfg = RainManConfig()
    cfg.shoe = ShoeConfig(
        num_decks=args.decks,
        penetration=args.penetration,
    )
    cfg.betting = BettingConfig(
        bankroll=args.bankroll,
        min_bet=args.min_bet,
        max_bet=args.max_bet,
    )
    cfg.monte_carlo = MonteCarloConfig(
        num_simulations=args.hands,
    )
    # Load BSE for the configured casino rules
    from ml.advantage_calc import CASINO_RULES, DEFAULT_RULES
    rules = CASINO_RULES.get(cfg.casino_rules_key, DEFAULT_RULES)
    cfg.counting = CountingSystem(name=args.system, bse=rules.bse)
    return cfg


# ---------------------------------------------------------------------------
# Mode: TRAIN
# ---------------------------------------------------------------------------

def mode_train(cfg: RainManConfig) -> None:
    """
    Training pipeline:
      1. Monte Carlo simulation (100k+ hands)
      2. OLS regression on simulation data
      3. Gauss-Markov assumptions tests
      4. Risk of Ruin estimation
      5. Save trained model
    """
    print_header()
    print_rule("TRAINING MODE")

    trainer = Trainer(config=cfg)
    trainer.train(verbose=True)
    trainer.save()

    if trainer.ols_results:
        print_rule("EV Curve from OLS Model")
        print_ev_table(trainer.ols_results.ev_curve)

    print_rule()
    if console:
        console.print("\n[bold green]Training complete![/bold green] "
                      "Run [cyan]python main.py --mode manual[/cyan] to start counting.")
    else:
        print("\nTraining complete! Run: python main.py --mode manual")


# ---------------------------------------------------------------------------
# Mode: LIVE (screen reader)
# ---------------------------------------------------------------------------

def mode_live(cfg: RainManConfig) -> None:
    """
    Live mode with automatic screen reading.
    Captures the screen, detects cards via OCR, updates count, recommends bets.
    """
    print_header()
    print_rule("LIVE MODE — Screen Reader")

    capture = ScreenCapture()
    detector = CardDetector()

    if not capture.is_available() or not detector.is_available():
        if console:
            console.print("[yellow]Screen reader not available. Switching to manual mode.[/yellow]")
        else:
            print("Screen reader not available. Switching to manual mode.")
        mode_manual(cfg)
        return

    shoe = Shoe(num_decks=cfg.shoe.num_decks, penetration=cfg.shoe.penetration)
    counter = make_counter(cfg.counting.name, shoe)

    # Load trained model if available
    strategy = _load_strategy(cfg)
    bankroll = cfg.betting.bankroll
    cards_seen: List[Card] = []
    hand_num = 0

    if console:
        console.print(f"[green]Starting live mode. Press Ctrl+C to stop.[/green]")
        console.print(f"[dim]Monitoring screen every {cfg.screen.capture_interval_ms}ms...[/dim]")
    else:
        print("Starting live mode. Press Ctrl+C to stop.")

    seen_cards_set = set()  # Track cards already counted (by position hash)

    try:
        while True:
            img = capture.grab()
            if img is None:
                time.sleep(0.5)
                continue

            detected = detector.detect(img)
            new_cards = []

            for det in detected:
                card = det.card
                if card and det.confidence >= cfg.screen.ocr_confidence:
                    # Simple deduplication: track by position
                    pos_key = det.bounding_box
                    if pos_key not in seen_cards_set:
                        seen_cards_set.add(pos_key)
                        counter.see_card(card)
                        cards_seen.append(card)
                        new_cards.append(card)

            if new_cards:
                hand_num += 1
                print_count_bar(counter.running_count, counter.true_count)
                print_cards_seen(cards_seen)
                rec = strategy.recommend_bet(counter, bankroll)
                print_bet_recommendation(rec)

            # Clear position cache on reshuffle (detected by sudden count reset)
            if shoe.needs_reshuffle:
                seen_cards_set.clear()
                counter.reset()
                shoe.shuffle()
                if console:
                    console.print("[bold yellow]SHOE RESHUFFLED — Count Reset[/bold yellow]")

            time.sleep(cfg.screen.capture_interval_ms / 1000.0)

    except KeyboardInterrupt:
        print_rule("Session Ended")
        _print_session_summary(bankroll, cfg.betting.bankroll, hand_num)


# ---------------------------------------------------------------------------
# Mode: MANUAL INPUT
# ---------------------------------------------------------------------------

def mode_manual(cfg: RainManConfig) -> None:
    """
    Manual card input mode.
    The user types cards as they are dealt.
    The system updates the count and recommends the next bet.
    """
    print_header()
    print_rule("MANUAL INPUT MODE")

    shoe = Shoe(num_decks=cfg.shoe.num_decks, penetration=cfg.shoe.penetration)
    counter = make_counter(cfg.counting.name, shoe)
    strategy = _load_strategy(cfg)
    inp = ManualCardInput()

    bankroll = cfg.betting.bankroll
    cards_seen: List[Card] = []
    hand_num = 0

    if console:
        console.print(f"\n[bold]Counting System:[/bold] [cyan]{counter.name}[/cyan]")
        console.print(f"[bold]Shoe:[/bold] {cfg.shoe.num_decks} decks, "
                      f"{cfg.shoe.penetration:.0%} penetration")
        console.print(f"[bold]Bankroll:[/bold] ${bankroll:.0f} | "
                      f"[bold]Bet range:[/bold] ${cfg.betting.min_bet}–${cfg.betting.max_bet}")
        console.print("\n[dim]Type cards as they are dealt (e.g. AS, 10H, KD).[/dim]")
        console.print("[dim]Commands: 'shuffle' = new shoe | 'count' = show count | 'quit' = exit[/dim]")
    else:
        print(f"\nCounting: {counter.name}")
        print(f"Shoe: {cfg.shoe.num_decks} decks, {cfg.shoe.penetration:.0%} penetration")
        print("Type cards as dealt (AS, 10H, KD, etc). 'shuffle' to reset, 'quit' to exit.")

    print_rule()

    # Show initial bet recommendation
    rec = strategy.recommend_bet(counter, bankroll)
    print_bet_recommendation(rec)

    try:
        while True:
            print_rule(f"Hand #{hand_num + 1}")
            cmd = inp.prompt_command(f"[RC={counter.running_count:+d} TC={counter.true_count:+.1f}]")

            if cmd in ("quit", "q", "exit"):
                break

            elif cmd in ("shuffle", "s", "reshuffle"):
                shoe.shuffle()
                counter.reset()
                cards_seen.clear()
                if console:
                    console.print("[bold yellow]Shoe reshuffled. Count reset to 0.[/bold yellow]")
                else:
                    print("Shoe reshuffled.")

            elif cmd in ("count", "c", "show"):
                print_count_bar(counter.running_count, counter.true_count)

            elif cmd in ("stats", "ev"):
                if strategy.ols_results:
                    print_ev_table(strategy.ols_results.ev_curve)
                else:
                    print("No model trained yet. Run: python main.py --mode train")

            elif cmd in ("advantage", "adv", "edge"):
                # Show exact Hi-Opt I/II advantage at current TC
                from ml.advantage_calc import AdvantageCalculator, CASINO_RULES
                rules = CASINO_RULES.get(cfg.casino_rules_key)
                if rules:
                    calc = AdvantageCalculator(rules)
                    calc.print_ev_table()
                    rep = calc.report(counter.true_count, bankroll,
                                      cfg.betting.min_bet, cfg.betting.max_bet)
                    if console:
                        col = "green" if rep.advantage_pct > 0 else "red"
                        console.print(f"\n[{col}]{rep}[/{col}]")
                    else:
                        print(f"\n{rep}")

            elif cmd in ("compare", "systems"):
                from ml.advantage_calc import AdvantageCalculator
                AdvantageCalculator.compare_rule_sets()

            elif cmd in ("systems", "list"):
                from counting import list_systems
                list_systems()

            elif cmd in ("help", "?"):
                from ui.manual_input import HELP_TEXT
                extra = (
                    "\nExtra commands:\n"
                    "  advantage / adv   → Show exact EV table (BSE + 0.515×TC)\n"
                    "  compare           → Compare all casino rule sets\n"
                    "  systems           → List all counting systems\n"
                    "  insurance         → Should I take insurance?\n"
                )
                print(HELP_TEXT + extra)

            elif cmd in ("insurance",):
                if hasattr(counter, "should_take_insurance"):
                    take = counter.should_take_insurance()
                else:
                    take = counter.true_count >= 3
                msg = "TAKE insurance (TC >= +3, deck is 10-rich)" if take else "DECLINE insurance"
                if console:
                    col = "green" if take else "red"
                    console.print(f"[{col}]{msg}[/{col}]")
                else:
                    print(msg)

            else:
                # Treat input as card notation
                card = inp._parse_single(cmd)
                if card:
                    counter.see_card(card)
                    cards_seen.append(card)
                    if console:
                        suit_col = "red" if card.suit in ("♥", "♦") else "white"
                        console.print(f"  Saw: [{suit_col}]{card}[/{suit_col}] | "
                                      f"Count tag: {counter.card_value(card):+d}")
                    else:
                        print(f"  Saw: {card}")

                    # After each card, show updated recommendation
                    print_count_bar(counter.running_count, counter.true_count)
                    rec = strategy.recommend_bet(counter, bankroll)
                    print_bet_recommendation(rec)

                    # Track penetration
                    # Simulate "dealing" from shoe for penetration tracking
                    if len(cards_seen) % 6 == 0:
                        # Rough penetration estimate
                        total_seen = len(cards_seen)
                        total_cards = cfg.shoe.num_decks * 52
                        pen = total_seen / total_cards
                        if pen > cfg.shoe.penetration:
                            if console:
                                console.print("[yellow]Cut card reached! Shuffle expected soon.[/yellow]")
                            else:
                                print("Cut card reached! Shuffle expected soon.")

                    # Check for Wong-out condition
                    if strategy.should_wong_out(counter):
                        if console:
                            console.print("[bold red]TC very negative — consider leaving table (Wong-out)[/bold red]")
                        else:
                            print("TC very negative — consider leaving table.")

                    hand_num += 1
                else:
                    # Try multi-card input
                    tokens = cmd.split()
                    parsed_any = False
                    for tok in tokens:
                        c = inp._parse_single(tok)
                        if c:
                            counter.see_card(c)
                            cards_seen.append(c)
                            parsed_any = True
                    if parsed_any:
                        print_count_bar(counter.running_count, counter.true_count)
                        rec = strategy.recommend_bet(counter, bankroll)
                        print_bet_recommendation(rec)
                        hand_num += 1
                    elif cmd:
                        if console:
                            console.print(f"[dim]Unknown command '{cmd}'. Type 'help' for commands.[/dim]")
                        else:
                            print(f"Unknown: '{cmd}'. Type 'help'.")

    except KeyboardInterrupt:
        pass

    print_rule("Session Ended")
    _print_session_summary(bankroll, cfg.betting.bankroll, hand_num)


# ---------------------------------------------------------------------------
# Mode: STATS
# ---------------------------------------------------------------------------

def mode_stats(cfg: RainManConfig) -> None:
    """Show statistics from the last training run."""
    import json
    from ml.trainer import STATS_PATH

    print_header()
    print_rule("TRAINING STATISTICS")

    if not os.path.exists(STATS_PATH):
        print("No training stats found. Run: python main.py --mode train")
        return

    with open(STATS_PATH) as f:
        stats = json.load(f)

    if console:
        table = __import__("rich.table", fromlist=["Table"]).Table(
            title="Rain Man Training Stats", show_header=True,
        )
        table.add_column("Metric", style="bold")
        table.add_column("Value")
        for k, v in stats.items():
            if k == "ev_curve":
                continue
            table.add_row(k.replace("_", " ").title(), str(v))
        console.print(table)
    else:
        for k, v in stats.items():
            if k != "ev_curve":
                print(f"  {k}: {v}")

    # Load model and show EV table
    try:
        trainer = Trainer.load(config=cfg)
        if trainer.ols_results:
            print_ev_table(trainer.ols_results.ev_curve)
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Mode: DEMO
# ---------------------------------------------------------------------------

def mode_demo(cfg: RainManConfig) -> None:
    """Quick demo: simulate 20 hands with Hi-Lo counting, show live recommendations."""
    print_header()
    print_rule("DEMO MODE — 20 Simulated Hands")

    cfg.monte_carlo.num_simulations = 5000  # Quick training
    cfg.monte_carlo.num_sessions = 100

    shoe = Shoe(num_decks=cfg.shoe.num_decks, penetration=cfg.shoe.penetration, seed=42)
    counter = make_counter(cfg.counting.name, shoe)
    strategy = BettingStrategy(config=cfg)

    # Try to load existing model, train quick if not found
    try:
        trainer = Trainer.load(config=cfg)
        strategy.update_model(trainer.ols_results)
        if console:
            console.print("[green]Loaded existing trained model.[/green]")
    except FileNotFoundError:
        if console:
            console.print("[yellow]No trained model found. Running quick 5k-hand simulation...[/yellow]")
        else:
            print("Running quick training...")
        trainer = Trainer(config=cfg)
        results = trainer.train(verbose=False)
        strategy.update_model(results)

    from simulator.blackjack_game import BlackjackGame
    game = BlackjackGame(shoe, cfg.rules, counter)
    bankroll = cfg.betting.bankroll
    cards_seen: List[Card] = []

    if console:
        console.print(f"\nSimulating 20 hands with {counter.name} counting...\n")
    else:
        print(f"\nSimulating 20 hands with {counter.name}...\n")

    for hand_num in range(1, 21):
        if shoe.needs_reshuffle:
            shoe.shuffle()
            counter.reset()
            cards_seen.clear()
            if console:
                console.print("[yellow]→ Shoe reshuffled[/yellow]")
            else:
                print("→ Shoe reshuffled")

        rec = strategy.recommend_bet(counter, bankroll)
        result = game.play_hand(rec.recommended_bet)
        bankroll += result.net_result

        for card in result.cards_seen:
            cards_seen.append(card)

        # Print hand result
        net_str = f"{result.net_result:+.0f}"
        if console:
            col = "green" if result.net_result > 0 else "red" if result.net_result < 0 else "yellow"
            console.print(f"  Hand {hand_num:>2}: bet=${rec.recommended_bet:.0f} | "
                          f"result=[{col}]{net_str}[/{col}] | "
                          f"bank=${bankroll:.0f} | "
                          f"RC={counter.running_count:+d} TC={counter.true_count:+.1f}")
        else:
            print(f"  Hand {hand_num:>2}: bet=${rec.recommended_bet:.0f} | "
                  f"result={net_str} | bank=${bankroll:.0f} | "
                  f"RC={counter.running_count:+d} TC={counter.true_count:+.1f}")

    print_rule()
    net = bankroll - cfg.betting.bankroll
    col = "green" if net >= 0 else "red"
    if console:
        console.print(f"\nFinal bankroll: [bold]${bankroll:.0f}[/bold] "
                      f"([{col}]{net:+.0f}[/{col}] from ${cfg.betting.bankroll:.0f})")
    else:
        print(f"\nFinal: ${bankroll:.0f} ({net:+.0f})")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_strategy(cfg: RainManConfig) -> BettingStrategy:
    """Load trained model into betting strategy, or use untrained fallback."""
    strategy = BettingStrategy(config=cfg)
    try:
        trainer = Trainer.load(config=cfg)
        strategy.update_model(trainer.ols_results)
        if console:
            console.print(f"[green]Loaded trained model (R²={trainer.ols_results.r_squared:.6f})[/green]")
        else:
            print(f"Loaded trained model (R²={trainer.ols_results.r_squared:.6f})")
    except FileNotFoundError:
        if console:
            console.print("[yellow]No trained model found — using empirical fallback.[/yellow]")
            console.print("[yellow]Run 'python main.py --mode train' for best results.[/yellow]")
        else:
            print("No model. Run: python main.py --mode train")
    return strategy


def _print_session_summary(final_bankroll: float, start_bankroll: float,
                            hands: int) -> None:
    net = final_bankroll - start_bankroll
    if console:
        col = "green" if net >= 0 else "red"
        console.print(f"\n[bold]Session Summary[/bold]")
        console.print(f"  Hands played : {hands}")
        console.print(f"  Final bankroll: ${final_bankroll:.0f}")
        console.print(f"  Net P&L: [{col}]{net:+.0f}[/{col}]")
    else:
        print(f"\nHands: {hands} | Final: ${final_bankroll:.0f} | Net: {net:+.0f}")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Rain Man — Blackjack Card Counting ML System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode", choices=["train", "live", "manual", "stats", "demo"],
                        default=None, help="Operation mode")
    parser.add_argument("--decks", type=int, default=6,
                        help="Number of decks in shoe (default: 6)")
    parser.add_argument("--penetration", type=float, default=0.75,
                        help="Shoe penetration 0.5–0.85 (default: 0.75)")
    parser.add_argument("--system",
                        choices=["hi_lo", "ko", "hi_opt1", "hi_opt2", "omega2"],
                        default="hi_opt1",
                        help="Card counting system (default: hi_opt1 — Humble & Cooper)")
    parser.add_argument("--bankroll", type=float, default=1000.0,
                        help="Starting bankroll in units (default: 1000)")
    parser.add_argument("--min-bet", type=float, default=10.0, dest="min_bet")
    parser.add_argument("--max-bet", type=float, default=200.0, dest="max_bet")
    parser.add_argument("--hands", type=int, default=100_000,
                        help="Hands to simulate in training (default: 100000)")
    parser.add_argument("--no-screen", action="store_true",
                        help="Disable screen reader, use manual input")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = build_config(args)

    mode = args.mode
    if mode is None:
        # Interactive mode selection
        choice = ask_mode()
        mode_map = {"1": "train", "2": "live", "3": "manual", "4": "stats", "5": "demo"}
        mode = mode_map.get(choice, "manual")

    if mode == "train":
        mode_train(cfg)
    elif mode == "live" and not args.no_screen:
        mode_live(cfg)
    elif mode == "manual" or args.no_screen:
        mode_manual(cfg)
    elif mode == "stats":
        mode_stats(cfg)
    elif mode == "demo":
        mode_demo(cfg)
    else:
        mode_manual(cfg)


if __name__ == "__main__":
    main()
