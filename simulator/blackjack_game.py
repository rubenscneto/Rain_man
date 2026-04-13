"""
Full blackjack game engine with basic strategy.

Implements all standard rules: hit, stand, double, split, surrender.
Basic strategy is encoded as lookup tables (hard, soft, pair).
Based on "The World's Greatest Blackjack Book" by Humble & Cooper.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple, TYPE_CHECKING

from simulator.deck import Card, Hand, Shoe
from config import BlackjackRules

if TYPE_CHECKING:
    from counting.base_counter import BaseCounter


class Action(Enum):
    HIT = auto()
    STAND = auto()
    DOUBLE = auto()
    SPLIT = auto()
    SURRENDER = auto()


class HandResult(Enum):
    WIN = auto()
    LOSS = auto()
    PUSH = auto()
    BLACKJACK = auto()
    SURRENDER = auto()


@dataclass
class RoundResult:
    """Result of a single blackjack round."""
    player_hands: List[Hand] = field(default_factory=list)
    dealer_hand: Hand = field(default_factory=Hand)
    results: List[HandResult] = field(default_factory=list)
    bets: List[float] = field(default_factory=list)
    net_result: float = 0.0           # Net profit/loss in units
    true_count_at_bet: float = 0.0    # True count when bet was placed
    running_count_at_bet: int = 0
    cards_seen: List[Card] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Basic Strategy Tables
# Encoded as: {player_total: {dealer_upcard: action}}
# Based on standard multi-deck (4-8), H17, DAS, no RSA rules.
# ---------------------------------------------------------------------------

# Actions shorthand
H = Action.HIT
S = Action.STAND
D = Action.DOUBLE
P = Action.SPLIT
R = Action.SURRENDER   # Surrender (R = suRrender)

# Hard totals basic strategy (player_total -> dealer_upcard 2..A)
HARD_STRATEGY: dict[int, dict] = {
    # total: {upcard_value: action}
    5:  {2:H,3:H,4:H,5:H,6:H,7:H,8:H,9:H,10:H,11:H},
    6:  {2:H,3:H,4:H,5:H,6:H,7:H,8:H,9:H,10:H,11:H},
    7:  {2:H,3:H,4:H,5:H,6:H,7:H,8:H,9:H,10:H,11:H},
    8:  {2:H,3:H,4:H,5:H,6:H,7:H,8:H,9:H,10:H,11:H},
    9:  {2:H,3:D,4:D,5:D,6:D,7:H,8:H,9:H,10:H,11:H},
    10: {2:D,3:D,4:D,5:D,6:D,7:D,8:D,9:D,10:H,11:H},
    11: {2:D,3:D,4:D,5:D,6:D,7:D,8:D,9:D,10:D,11:H},
    12: {2:H,3:H,4:S,5:S,6:S,7:H,8:H,9:H,10:H,11:H},
    13: {2:S,3:S,4:S,5:S,6:S,7:H,8:H,9:H,10:H,11:H},
    14: {2:S,3:S,4:S,5:S,6:S,7:H,8:H,9:H,10:H,11:H},
    15: {2:S,3:S,4:S,5:S,6:S,7:H,8:H,9:H,10:R,11:H},
    16: {2:S,3:S,4:S,5:S,6:S,7:H,8:H,9:R,10:R,11:R},
    17: {2:S,3:S,4:S,5:S,6:S,7:S,8:S,9:S,10:S,11:S},
    18: {2:S,3:S,4:S,5:S,6:S,7:S,8:S,9:S,10:S,11:S},
    19: {2:S,3:S,4:S,5:S,6:S,7:S,8:S,9:S,10:S,11:S},
    20: {2:S,3:S,4:S,5:S,6:S,7:S,8:S,9:S,10:S,11:S},
    21: {2:S,3:S,4:S,5:S,6:S,7:S,8:S,9:S,10:S,11:S},
}

# Soft totals (A + X) strategy — key is the non-Ace card value
SOFT_STRATEGY: dict[int, dict] = {
    2:  {2:H,3:H,4:H,5:D,6:D,7:H,8:H,9:H,10:H,11:H},   # A+2 = soft 13
    3:  {2:H,3:H,4:H,5:D,6:D,7:H,8:H,9:H,10:H,11:H},   # A+3 = soft 14
    4:  {2:H,3:H,4:D,5:D,6:D,7:H,8:H,9:H,10:H,11:H},   # A+4 = soft 15
    5:  {2:H,3:H,4:D,5:D,6:D,7:H,8:H,9:H,10:H,11:H},   # A+5 = soft 16
    6:  {2:H,3:D,4:D,5:D,6:D,7:H,8:H,9:H,10:H,11:H},   # A+6 = soft 17
    7:  {2:S,3:D,4:D,5:D,6:D,7:S,8:S,9:H,10:H,11:H},   # A+7 = soft 18
    8:  {2:S,3:S,4:S,5:S,6:S,7:S,8:S,9:S,10:S,11:S},   # A+8 = soft 19
    9:  {2:S,3:S,4:S,5:S,6:S,7:S,8:S,9:S,10:S,11:S},   # A+9 = soft 20
    10: {2:S,3:S,4:S,5:S,6:S,7:S,8:S,9:S,10:S,11:S},   # A+10 = soft 21 (BJ)
}

# Pair splitting strategy — key is the card value of the pair
PAIR_STRATEGY: dict[int, dict] = {
    2:  {2:P,3:P,4:P,5:P,6:P,7:P,8:H,9:H,10:H,11:H},
    3:  {2:P,3:P,4:P,5:P,6:P,7:P,8:H,9:H,10:H,11:H},
    4:  {2:H,3:H,4:H,5:P,6:P,7:H,8:H,9:H,10:H,11:H},
    5:  {2:D,3:D,4:D,5:D,6:D,7:D,8:D,9:D,10:H,11:H},   # Never split 5s
    6:  {2:P,3:P,4:P,5:P,6:P,7:H,8:H,9:H,10:H,11:H},
    7:  {2:P,3:P,4:P,5:P,6:P,7:P,8:H,9:H,10:H,11:H},
    8:  {2:P,3:P,4:P,5:P,6:P,7:P,8:P,9:P,10:P,11:P},   # Always split 8s
    9:  {2:P,3:P,4:P,5:P,6:P,7:S,8:P,9:P,10:S,11:S},
    10: {2:S,3:S,4:S,5:S,6:S,7:S,8:S,9:S,10:S,11:S},   # Never split 10s
    11: {2:P,3:P,4:P,5:P,6:P,7:P,8:P,9:P,10:P,11:P},   # Always split Aces
}


def _upcard_key(card: Card) -> int:
    """Map dealer upcard to strategy key (Ace = 11)."""
    return card.value


def basic_strategy_action(hand: Hand, dealer_upcard: Card,
                           rules: BlackjackRules,
                           allow_split: bool = True) -> Action:
    """
    Return the basic-strategy optimal action for a given hand vs dealer upcard.
    This is the mathematically correct play, ignoring card counting deviations.
    """
    upcard = _upcard_key(dealer_upcard)
    total = hand.total

    # Pair splitting
    if hand.is_pair and allow_split and rules.max_splits > 0:
        pair_val = hand.cards[0].value
        action = PAIR_STRATEGY.get(pair_val, {}).get(upcard, H)
        if action == P:
            return Action.SPLIT

    # Soft hand
    if hand.is_soft and len(hand.cards) == 2:
        non_ace_val = min(c.value for c in hand.cards if not c.is_ace)
        if non_ace_val in SOFT_STRATEGY:
            action = SOFT_STRATEGY[non_ace_val].get(upcard, H)
            # Can only double on first two cards
            if action == D and not hand.can_double:
                action = H
            return action

    # Hard hand
    capped_total = min(total, 21)
    if capped_total < 5:
        capped_total = 5
    action = HARD_STRATEGY.get(capped_total, {}).get(upcard, S)

    # Surrender only on first two cards
    if action == R:
        if rules.surrender_allowed and len(hand.cards) == 2:
            return Action.SURRENDER
        else:
            return Action.HIT

    # Double only on first two cards
    if action == D and not hand.can_double:
        return Action.HIT

    return action


class BlackjackGame:
    """
    Blackjack game engine.

    Plays full hands, applies basic strategy by default,
    and integrates with any card counting system.
    """

    def __init__(self, shoe: Shoe, rules: BlackjackRules,
                 counter: Optional["BaseCounter"] = None):
        self.shoe = shoe
        self.rules = rules
        self.counter = counter

    def _deal_card(self, hand: Hand) -> Card:
        """Deal one card, add to hand, notify counter."""
        card = self.shoe.deal()
        hand.add_card(card)
        if self.counter:
            self.counter.see_card(card)
        return card

    def _deal_hidden(self, hand: Hand) -> Card:
        """Deal dealer hole card (not visible to counter until revealed)."""
        card = self.shoe.deal()
        hand.add_card(card)
        return card

    def _reveal_hole_card(self, card: Card) -> None:
        """Reveal the dealer's hole card to the counter."""
        if self.counter:
            self.counter.see_card(card)

    def play_hand(self, bet: float,
                  player_strategy=None) -> RoundResult:
        """
        Play one complete round of blackjack.

        Parameters
        ----------
        bet : float
            Player's initial bet.
        player_strategy : callable | None
            Function(hand, dealer_upcard, rules) -> Action.
            Defaults to basic strategy.

        Returns
        -------
        RoundResult
        """
        if self.shoe.needs_reshuffle:
            self.shoe.shuffle()
            if self.counter:
                self.counter.reset()

        result = RoundResult()
        result.true_count_at_bet = self.counter.true_count if self.counter else 0.0
        result.running_count_at_bet = self.counter.running_count if self.counter else 0

        strategy = player_strategy or (
            lambda h, u, r: basic_strategy_action(h, u, r)
        )

        # --- Initial deal ---
        player_hand = Hand()
        dealer_hand = Hand()

        # Card dealing order: P, D, P, D(hole)
        self._deal_card(player_hand)
        self._deal_card(dealer_hand)       # Dealer upcard (visible)
        self._deal_card(player_hand)
        hole_card = self._deal_hidden(dealer_hand)  # Dealer hole card

        dealer_upcard = dealer_hand.cards[0]
        result.dealer_hand = dealer_hand
        result.cards_seen = list(player_hand.cards) + [dealer_upcard]

        # --- Check for dealer blackjack (peek) ---
        dealer_has_bj = dealer_hand.is_blackjack

        # --- Player blackjack ---
        if player_hand.is_blackjack:
            self._reveal_hole_card(hole_card)
            if dealer_has_bj:
                result.results.append(HandResult.PUSH)
                result.bets.append(bet)
                result.net_result = 0.0
            else:
                result.results.append(HandResult.BLACKJACK)
                result.bets.append(bet)
                result.net_result = bet * self.rules.blackjack_payout
            result.player_hands = [player_hand]
            return result

        # --- Dealer blackjack (no player BJ) ---
        if dealer_has_bj:
            self._reveal_hole_card(hole_card)
            result.results.append(HandResult.LOSS)
            result.bets.append(bet)
            result.net_result = -bet
            result.player_hands = [player_hand]
            return result

        # --- Play player hands (supports splits) ---
        hands_to_play: List[Tuple[Hand, float]] = [(player_hand, bet)]
        completed_hands: List[Tuple[Hand, float]] = []
        split_count = 0

        while hands_to_play:
            current_hand, current_bet = hands_to_play.pop(0)

            # Need 2 cards to start a split hand
            if len(current_hand.cards) < 2:
                self._deal_card(current_hand)

            # Aces split: get only one card each
            if (current_hand.is_split_hand and
                    current_hand.cards[0].is_ace and
                    not self.rules.resplit_aces):
                completed_hands.append((current_hand, current_bet))
                continue

            while True:
                action = strategy(current_hand, dealer_upcard, self.rules)

                if action == Action.SURRENDER and len(current_hand.cards) == 2:
                    current_hand.doubled = False
                    completed_hands.append((current_hand, current_bet * 0.5))
                    result.results.append(HandResult.SURRENDER)
                    break

                elif action == Action.SPLIT and split_count < self.rules.max_splits:
                    split_count += 1
                    hand_a = Hand()
                    hand_b = Hand()
                    hand_a.is_split_hand = True
                    hand_b.is_split_hand = True
                    hand_a.add_card(current_hand.cards[0])
                    hand_b.add_card(current_hand.cards[1])
                    if self.rules.double_after_split:
                        # Re-enable doubling on split hands
                        pass
                    hands_to_play.insert(0, (hand_a, current_bet))
                    hands_to_play.insert(1, (hand_b, current_bet))
                    break

                elif action == Action.DOUBLE and current_hand.can_double:
                    self._deal_card(current_hand)
                    current_hand.doubled = True
                    completed_hands.append((current_hand, current_bet * 2))
                    break

                elif action == Action.HIT:
                    self._deal_card(current_hand)
                    if current_hand.is_bust:
                        completed_hands.append((current_hand, current_bet))
                        break

                else:  # STAND
                    completed_hands.append((current_hand, current_bet))
                    break

        # --- Reveal hole card and play dealer hand ---
        self._reveal_hole_card(hole_card)
        result.cards_seen.append(hole_card)

        # Dealer plays: hits until 17+ (or soft 17 if H17 rule)
        while True:
            dtotal = dealer_hand.total
            if dtotal < 17:
                self._deal_card(dealer_hand)
            elif dtotal == 17 and dealer_hand.is_soft and self.rules.dealer_hits_soft17:
                self._deal_card(dealer_hand)
            else:
                break

        # --- Evaluate results ---
        dealer_total = dealer_hand.total
        net = 0.0
        final_results = []
        final_bets = []

        for hand, h_bet in completed_hands:
            if len(result.results) > len(final_results):
                # Surrender was already recorded
                final_results.append(HandResult.SURRENDER)
                final_bets.append(h_bet)
                net -= h_bet
                continue

            if hand.is_bust:
                final_results.append(HandResult.LOSS)
                final_bets.append(h_bet)
                net -= h_bet
            elif dealer_hand.is_bust or hand.total > dealer_total:
                final_results.append(HandResult.WIN)
                final_bets.append(h_bet)
                net += h_bet
            elif hand.total == dealer_total:
                final_results.append(HandResult.PUSH)
                final_bets.append(h_bet)
            else:
                final_results.append(HandResult.LOSS)
                final_bets.append(h_bet)
                net -= h_bet

        result.player_hands = [h for h, _ in completed_hands]
        result.results = final_results
        result.bets = final_bets
        result.net_result = net
        return result
