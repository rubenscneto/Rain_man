"""
Watch Mode — Live dealer monitoring.

This is the main live session orchestrator. It combines:
  1. Automatic screen capture + OCR (Evolution Gaming detector)
  2. Hotkey fallback for fast manual override
  3. Live count dashboard
  4. Bet recommendations

The two modes complement each other:
  - OCR runs in background, auto-detecting cards when possible
  - Hotkeys are ALWAYS available for instant manual input
  - If OCR detects a card, it's shown; user can press backspace to undo
  - User presses 'n' / Space when the round ends to confirm round boundary

Layout:
┌─────────────────────────────────────────────────┐
│  RC: +5   TC: +1.2   Advantage: +0.08%          │
│  ████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
│                                                  │
│  BET NEXT HAND:  $40  (4 units)                  │
│  Insurance: NO                                   │
│                                                  │
│  Cards this shoe: A♠ K♥ 5♦ 3♣ 7♠ Q♦ 2♥ ...     │
│  [AUTO] 10♥ detected  [MANUAL] 6♦ entered        │
└─────────────────────────────────────────────────┘
"""

from __future__ import annotations
import threading
import time
from typing import List, Optional, TYPE_CHECKING

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    from rich import box as rich_box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from simulator.deck import Card
from counting.base_counter import BaseCounter
from ml.betting_strategy import BettingStrategy
from ui.hotkey_input import HotkeyListener, PYNPUT_AVAILABLE
from screen_reader.calibrate import get_capture_region

if TYPE_CHECKING:
    from config import RainManConfig


class WatchSession:
    """
    Live watch session.

    Runs two threads:
      1. Screen capture thread (OCR, polls every 300ms)
      2. Hotkey listener thread (instant response)

    Main thread drives the Rich live display.
    """

    def __init__(self, counter: BaseCounter, strategy: BettingStrategy,
                 config: "RainManConfig", bankroll: float):
        self.counter = counter
        self.strategy = strategy
        self.config = config
        self.bankroll = bankroll

        self.cards_seen: List[Card] = []
        self.cards_this_round: List[Card] = []
        self.last_source: str = ""    # "AUTO" or "MANUAL"
        self.last_card_str: str = ""
        self.round_num: int = 0
        self._lock = threading.Lock()
        self._running = False
        self._console = Console() if RICH_AVAILABLE else None

        # Última estimativa de baralhos (do detector durante mistura)
        self._deck_estimate_str: str = ""

        # ── Conselheiro de mão ─────────────────────────────────────────
        # Pressione 'm' para ativar modo minha-mão, depois tecle suas cartas.
        # Pressione 'd' para informar a carta visível do dealer.
        # O painel mostrará a ação recomendada (Pedir / Parar / Dobrar...).
        self.player_hand: List[Card] = []      # Cartas da SUA mão
        self.dealer_upcard: Optional[Card] = None  # Carta visível do dealer
        self._hand_mode: bool = False   # True = próximas cartas vão para player_hand
        self._dealer_mode: bool = False # True = próxima carta é o dealer_upcard

        # OCR detector
        self._ocr_thread: Optional[threading.Thread] = None
        self._ocr_available = False
        self._setup_ocr()

        # Hotkey listener
        self._hotkey = HotkeyListener(
            on_card=self._on_card_hotkey,
            on_control=self._on_control,
            verbose=False,
        )

    def _setup_ocr(self) -> None:
        """Try to initialise the Evolution Gaming OCR detector."""
        try:
            from screen_reader.evolution_detector import EvolutionDetector
            region = get_capture_region()
            self._detector = EvolutionDetector(
                region=region,
                poll_ms=self.config.screen.capture_interval_ms,
                on_card=self._on_card_ocr,
                on_new_round=self._on_new_round,
                on_deck_estimate=self._on_deck_estimate,
            )
            self._ocr_available = self._detector.is_available()
        except Exception as e:
            self._ocr_available = False
            self._detector = None

    # ------------------------------------------------------------------
    # Card / control event handlers (called from background threads)
    # ------------------------------------------------------------------

    def _on_card_ocr(self, det) -> None:
        """Called by OCR detector when a card is detected."""
        card = det.card
        if card is None:
            return
        with self._lock:
            self.counter.see_card(card)
            self.cards_seen.append(card)
            self.cards_this_round.append(card)
            self.last_source = "AUTO"
            self.last_card_str = str(card)

    def _on_card_hotkey(self, card: Card) -> None:
        """Called by hotkey listener when user presses a card key."""
        with self._lock:
            # Sempre conta a carta normalmente
            self.counter.see_card(card)
            self.cards_seen.append(card)
            self.cards_this_round.append(card)
            self.last_source = "KEY"
            self.last_card_str = str(card)

            # Modo dealer: registra como carta visível do dealer
            if self._dealer_mode:
                self.dealer_upcard = card
                self._dealer_mode = False   # auto-desativa após 1 carta

            # Modo minha mão: adiciona à mão do jogador
            elif self._hand_mode:
                self.player_hand.append(card)

    def _on_new_round(self) -> None:
        """Called when OCR detects a new round starting."""
        with self._lock:
            self.cards_this_round.clear()
            self.round_num += 1

    def _on_deck_estimate(self, estimate) -> None:
        """Called when OCR estimates deck count after a shuffle."""
        with self._lock:
            conf = f"{estimate.confidence:.0%}"
            self._deck_estimate_str = (
                f"~{estimate.estimated_decks} baralhos ({conf} confiança)"
            )

    def _on_control(self, cmd: str) -> None:
        """Called by hotkey listener for control keys."""
        with self._lock:
            if cmd == "next_hand":
                self.cards_this_round.clear()
                self.round_num += 1
                # Limpa mão e modos ao iniciar nova rodada
                self.player_hand.clear()
                self.dealer_upcard = None
                self._hand_mode = False
                self._dealer_mode = False

            elif cmd == "shuffle":
                self.counter.reset()
                self.cards_seen.clear()
                self.cards_this_round.clear()
                self.player_hand.clear()
                self.dealer_upcard = None
                self._hand_mode = False
                self._dealer_mode = False
                self.round_num = 0
                self.last_source = "CTRL"
                self.last_card_str = "SHOE RESET"

            elif cmd == "hand_mode":
                # Alterna modo minha-mão (desativa dealer_mode se ativo)
                self._hand_mode = not self._hand_mode
                self._dealer_mode = False

            elif cmd == "dealer_mode":
                # Ativa modo dealer para a próxima carta (desativa minha-mão)
                self._dealer_mode = not self._dealer_mode
                self._hand_mode = False

            elif cmd == "quit":
                self._running = False

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the watch session (blocking)."""
        self._running = True

        # Start OCR thread
        if self._ocr_available:
            self._ocr_thread = threading.Thread(
                target=self._detector.start, daemon=True
            )
            self._ocr_thread.start()

        # Start hotkey listener
        if PYNPUT_AVAILABLE:
            self._hotkey.start()
        else:
            self._print_fallback_notice()

        self._print_header()

        if RICH_AVAILABLE:
            self._run_rich()
        else:
            self._run_plain()

        # Cleanup
        if self._detector:
            self._detector.stop()
        self._hotkey.stop()

    def _run_rich(self) -> None:
        """Rich live display loop."""
        with Live(self._build_panel(), refresh_per_second=4,
                  console=self._console) as live:
            while self._running:
                try:
                    live.update(self._build_panel())
                    time.sleep(0.25)
                except KeyboardInterrupt:
                    break

    def _run_plain(self) -> None:
        """Plain text display loop (no Rich)."""
        while self._running:
            try:
                self._print_plain_update()
                time.sleep(0.5)
            except KeyboardInterrupt:
                break

    # ------------------------------------------------------------------
    # Conselheiro de estratégia
    # ------------------------------------------------------------------

    def _get_strategy_advice(self) -> str:
        """
        Retorna a ação recomendada para a mão atual do jogador vs dealer.
        Usa estratégia básica + desvios de índice do Hi-Opt I.
        Chamado com o lock JÁ adquirido.
        """
        if len(self.player_hand) < 2 or self.dealer_upcard is None:
            return ""

        try:
            from simulator.blackjack_game import basic_strategy_action, Action
            from simulator.deck import Hand as BjHand

            # Monta o objeto Hand com as cartas do jogador
            hand = BjHand()
            for card in self.player_hand:
                hand.add_card(card)

            action = basic_strategy_action(hand, self.dealer_upcard, self.config.rules)

            # Traduz ação para português
            _pt = {
                Action.HIT:       ("PEDIR CARTA",   "yellow"),
                Action.STAND:     ("PARAR",          "green"),
                Action.DOUBLE:    ("DOBRAR APOSTA",  "cyan"),
                Action.SPLIT:     ("DIVIDIR",        "magenta"),
                Action.SURRENDER: ("RENDER (metade)","red"),
            }
            action_str, action_col = _pt.get(action, (str(action), "white"))

            # Resumo da mão
            total = hand.total
            soft  = " mole" if hand.is_soft else ""
            hand_cards = " ".join(
                f"[{'red' if c.suit in ('♥','♦') else 'white'}]{c}[/]"
                for c in self.player_hand
            )
            dealer_str = (
                f"[{'red' if self.dealer_upcard.suit in ('♥','♦') else 'white'}]"
                f"{self.dealer_upcard}[/]"
            )

            # Verifica desvio de índice do Hi-Opt I (se disponível)
            index_note = ""
            try:
                if hasattr(self.counter, "get_hard_standing_action"):
                    idx = self.counter.get_hard_standing_action(
                        total, self.dealer_upcard
                    )
                    if idx is not None and idx != action_str:
                        index_note = f"  [dim](TC{self.counter.true_count:+.1f} → {idx})[/dim]"
            except Exception:
                pass

            return (
                f"  Sua mão: {hand_cards} = [bold]{total}{soft}[/bold]  "
                f"vs dealer {dealer_str}\n"
                f"  ▶ [{action_col}][bold]{action_str}[/bold][/{action_col}]"
                f"{index_note}"
            )

        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Display rendering
    # ------------------------------------------------------------------

    def _build_panel(self):
        """Build Rich panel for live display."""
        with self._lock:
            rc = self.counter.running_count
            tc = self.counter.true_count
            decks = self.counter.shoe.decks_remaining

            rec = self.strategy.recommend_bet(self.counter, self.bankroll)
            ev = rec.predicted_ev

            # Colour coding
            tc_col = ("bold green" if tc >= 2 else
                      "green" if tc >= 1 else
                      "yellow" if tc >= -1 else
                      "red")
            ev_col = "bold green" if ev > 0 else "red"

            # Build count bar
            clamped = max(-10, min(10, tc))
            filled = int((clamped + 10) / 20 * 38)
            bar = "█" * filled + "░" * (38 - filled)

            # Cards this round
            round_str = " ".join(
                f"[{'red' if c.suit in ('♥','♦') else 'white'}]{c}[/]"
                for c in self.cards_this_round[-12:]
            )

            # Last card detected
            source_col = "cyan" if self.last_source == "AUTO" else "yellow"
            last_str = (
                f"[{source_col}][{self.last_source}][/{source_col}] "
                f"{self.last_card_str}"
                if self.last_card_str else "—"
            )

            # OCR status
            ocr_status = (
                "[green]OCR ON[/green]" if self._ocr_available
                else "[yellow]OCR OFF[/yellow]"
            )

            # Deck estimate
            deck_str = (
                f"  [dim]Baralhos estimados: {self._deck_estimate_str}[/dim]\n"
                if self._deck_estimate_str else ""
            )

            # Insurance
            ins_advice = ""
            if hasattr(self.counter, "should_take_insurance"):
                if self.counter.should_take_insurance():
                    ins_advice = "\n  [bold green]★ FAZER SEGURO (TC≥+3)[/bold green]"

            # ── Conselheiro de mão ────────────────────────────────────
            advice_str = self._get_strategy_advice()

            # Indicadores de modo ativo
            hand_indicator = ""
            if self._hand_mode:
                hand_indicator = "  [bold yellow]► MODO MINHA MÃO (m)[/bold yellow]  tecle suas cartas\n"
            elif self._dealer_mode:
                hand_indicator = "  [bold cyan]► MODO DEALER (d)[/bold cyan]  tecle a carta do dealer\n"

            # Bloco do conselheiro
            if advice_str:
                advice_block = (
                    f"\n  ──────────────── CONSELHEIRO ─────────────────\n"
                    f"{advice_str}\n"
                    f"  ───────────────────────────────────────────────\n"
                )
            elif hand_indicator:
                advice_block = f"\n{hand_indicator}"
            else:
                advice_block = (
                    f"\n  [dim]Conselheiro: pressione [bold]m[/bold] (minha mão) "
                    f"depois [bold]d[/bold] (dealer) para ver a ação recomendada[/dim]\n"
                )

            text = Text.from_markup(
                f"\n"
                f"  [{tc_col}]{bar}[/{tc_col}]\n"
                f"  RC: [bold]{rc:+d}[/bold]   "
                f"TC: [{tc_col}]{tc:+.2f}[/{tc_col}]   "
                f"Decks: {decks:.1f}   "
                f"Rodada #{self.round_num}   {ocr_status}\n\n"
                f"  [{ev_col}]EV: {ev:+.3%}[/{ev_col}]   "
                f"[bold cyan]APOSTAR: ${rec.recommended_bet:.0f}[/bold cyan]"
                f"  ({rec.bet_units:.0f} unidades)"
                f"{ins_advice}\n\n"
                f"  Cartas desta rodada: {round_str or '—'}\n"
                f"  Última: {last_str}\n"
                f"{deck_str}"
                f"{advice_block}"
                f"\n  [dim]Cartas: 3-9 · t=10/J/Q/K · a=Ás  │  "
                f"m=minha mão · d=dealer · n=próxima · s=embaralhar · Esc=sair[/dim]\n"
            )

            border = "green" if ev > 0 else "red"
            return Panel(text, title="[bold]RAINMAN — Live Count[/bold]",
                         border_style=border, padding=(0, 1))

    def _print_plain_update(self) -> None:
        with self._lock:
            rc = self.counter.running_count
            tc = self.counter.true_count
            rec = self.strategy.recommend_bet(self.counter, self.bankroll)
            print(f"\rRC={rc:+d}  TC={tc:+.1f}  "
                  f"EV={rec.predicted_ev:+.3%}  "
                  f"BET=${rec.recommended_bet:.0f}  "
                  f"[{self.last_source}]{self.last_card_str}   ",
                  end="", flush=True)

    def _print_header(self) -> None:
        if self._console:
            self._console.print()
            self._console.print(
                Panel.fit(
                    "[bold cyan]RAINMAN — WATCH MODE[/bold cyan]\n"
                    + ("[green]OCR active — auto-detecting cards[/green]\n"
                       if self._ocr_available
                       else "[yellow]OCR inactive — use hotkeys[/yellow]\n")
                    + "[dim]Hotkeys: 3456789=card  t=10  a=Ace  "
                      "n=next  s=shuffle  Esc=quit[/dim]",
                    border_style="cyan"
                )
            )
        else:
            print("\n=== RAINMAN WATCH MODE ===")
            print("Hotkeys: 3-9=card, t=10/J/Q/K, a=Ace, n=next, s=shuffle, Esc=quit\n")

    def _print_fallback_notice(self) -> None:
        if self._console:
            self._console.print(
                "[yellow]pynput not available — type single characters + Enter[/yellow]"
            )
        else:
            print("pynput not available — using terminal input mode.")
