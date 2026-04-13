"""
Hi-Opt I Card Counting System.

Based directly on "The World's Greatest Blackjack Book" by Humble & Cooper.

Tag values:
  3, 4, 5, 6       →  +1  (small cards that help the dealer — we want them gone)
  10, J, Q, K      →  -1  (high cards that help the player — we want them to stay)
  A, 2, 7, 8, 9   →   0  (neutral — not counted)

Key difference from Hi-Lo:
  - 2 is NOT counted (0 instead of +1)
  - Ace is NOT counted (0 instead of -1)
  - Aces tracked separately for a side count (optional, adds ~0.2% edge)

Performance (from Braun's 1M-hand simulation, p.243 of the book):
  - Betting Correlation: 0.86 (0.96 with Ace side count)
  - Playing Efficiency:  0.615
  - 1-deck flat betting:  0.8%
  - 1-deck 1:4 spread:    2.1%
  - 1-deck 1:8 spread:    2.8%
  - 4-deck 1:8 spread:    1.1%

Advantage formula (Griffin, p.514-516 of the book):
  Player Advantage (%) = BSE + 0.515 × TC

where BSE = Basic Strategy Expectation for the specific casino rules,
and TC = Hi-Opt I True Count.
Every +1 unit of true count is worth approximately 0.515% to the player.

Index plays: the full 4-deck tables (±6 range) from Chapter 8.
"Limiting what you learn to +6... offers a surprisingly good coverage
of the true counts you will see in actual play." — Humble & Cooper, p.252
"""

from __future__ import annotations
from typing import Optional
from simulator.deck import Card, Shoe
from counting.base_counter import BaseCounter


# ---------------------------------------------------------------------------
# Hi-Opt I Card Tags
# ---------------------------------------------------------------------------

HI_OPT1_TAGS: dict[str, int] = {
    "2":  0,          # NOT counted (differs from Hi-Lo)
    "3": +1,
    "4": +1,
    "5": +1,
    "6": +1,
    "7":  0,
    "8":  0,
    "9":  0,
    "10": -1,
    "J":  -1,
    "Q":  -1,
    "K":  -1,
    "A":   0,         # NOT counted — tracked by Ace side count
}

# Basic Strategy Expectations (BSE) for common casino rule sets.
# Source: p.526 of the book. Negative = house edge.
BSE_TABLE: dict[str, float] = {
    "caesars_1d":          +0.17,   # Caesars Palace, single deck
    "stardust_1d":          0.00,   # Stardust, single deck
    "lv_downtown_1d":      -0.19,   # Las Vegas downtown, single deck
    "mgm_4d":              -0.34,   # MGM Grand, 4 decks
    "typical_6d_h17":      -0.54,   # Typical 6-deck H17 (most common today)
    "typical_6d_s17":      -0.34,   # Typical 6-deck S17
    "typical_8d_h17":      -0.57,   # Atlantic City 8-deck H17
    "atlantic_city_8d":    -0.43,   # Atlantic City 8-deck with LS
}

# ---------------------------------------------------------------------------
# Hi-Opt I Index Play Tables (4-deck, Chapter 8)
#
# Format: {player_hand: {dealer_upcard: index}}
# To STAND:   TC >= index
# To DOUBLE:  TC >= index
# To SPLIT:   TC >= index  (or TC < index for "below" plays)
# To SURRENDER: TC >= index
# H  = always Hit          S = always Stand
# D  = always Double       SP = always Split
# ---------------------------------------------------------------------------

# Sentinel values for "always" actions
_S  = -999   # Always STAND
_H  = +999   # Always HIT
_D  = -999   # Always DOUBLE (when in doubling table)
_SP = -999   # Always SPLIT (when in splitting table)

# --- 4-Deck Hard Standing & Hitting ---
# Read: STAND if TC >= index; HIT if TC < index
# Player total (row) vs dealer upcard 2-A (column)
HARD_STAND: dict[int, dict[int, int]] = {
    # total: {2: idx, 3: idx, ..., 10: idx, 11: idx}
    # 11 = Ace upcard
    12: {2: 2,   3: 1,   4: 0,   5: -1,  6: -1,  7: _H,  8: _H,  9: _H,  10: _H,  11: _H},
    13: {2: -1,  3: -2,  4: -3,  5: -4,  6: -4,  7: _H,  8: _H,  9: _H,  10: _H,  11: _H},
    14: {2: -3,  3: -4,  4: -5,  5: _S,  6: -6,  7: _H,  8: _H,  9: _H,  10: _H,  11: _H},
    15: {2: -5,  3: -6,  4: _S,  5: _S,  6: _S,  7: _H,  8: _H,  9: 6,   10: 3,   11: _H},
    16: {2: _S,  3: _S,  4: _S,  5: _S,  6: _S,  7: _H,  8: 6,   9: 4,   10: 0,   11: _H},
    17: {2: _S,  3: _S,  4: _S,  5: _S,  6: _S,  7: _S,  8: _S,  9: _S,  10: _S,  11: -6},
}  # 18+ → always stand

# --- 4-Deck Soft Standing & Hitting ---
# Key = non-Ace card value (soft 18 = A+7, key=7)
SOFT_STAND: dict[int, dict[int, int]] = {
    7: {2: _S, 3: _S, 4: _S, 5: _S, 6: _S, 7: _S, 8: _S, 9: _H, 10: _H, 11: 1},
    # soft 19+ → always stand; soft 17- → always hit
}

# --- 4-Deck Hard Doubling ---
# Read: DOUBLE if TC >= index; HIT if TC < index
HARD_DOUBLE: dict[int, dict[int, int]] = {
    8:  {2: _H,  3: _H,  4: 5,   5: 3,   6: 2,   7: _H,  8: _H,  9: _H,  10: _H,  11: _H},
    9:  {2: 1,   3: 0,   4: -2,  5: -4,  6: -4,  7: 3,   8: _H,  9: _H,  10: _H,  11: _H},
    10: {2: 0,   3: 0,   4: 0,   5: 0,   6: 0,   7: -6,  8: -4,  9: -1,  10: 5,   11: 4},
    11: {2: 0,   3: 0,   4: 0,   5: 0,   6: 0,   7: 0,   8: -5,  9: -4,  10: -3,  11: 1},
}  # 7 and below: never double

# --- 4-Deck Soft Doubling ---
# Key = non-Ace card value
# _D = always double when basic strategy says so; otherwise use index
SOFT_DOUBLE: dict[int, dict[int, int]] = {
    2: {2: _H, 3: _H, 4: 2,   5: -1,  6: -2,  7: _H, 8: _H, 9: _H, 10: _H, 11: _H},  # A,2
    3: {2: _H, 3: _H, 4: 1,   5: -2,  6: -4,  7: _H, 8: _H, 9: _H, 10: _H, 11: _H},  # A,3
    4: {2: _H, 3: 5,  4: -1,  5: -5,  6: _D,  7: _H, 8: _H, 9: _H, 10: _H, 11: _H},  # A,4
    5: {2: _H, 3: 3,  4: -1,  5: -5,  6: _D,  7: _H, 8: _H, 9: _H, 10: _H, 11: _H},  # A,5
    6: {2: 1,  3: -2, 4: -5,  5: _D,  6: _D,  7: _H, 8: _H, 9: _H, 10: _H, 11: _H},  # A,6
    7: {2: 1,  3: -1, 4: -5,  5: _D,  6: _D,  7: _S, 8: _S, 9: _H, 10: _H, 11: None}, # A,7 (*=soft stand table)
    8: {2: 6,  3: 4,  4: 3,   5: 1,   6: 1,   7: _S, 8: _S, 9: _S, 10: _S, 11: _S},  # A,8
    9: {2: _S, 3: 5,  4: 6,   5: 5,   6: 5,   7: _S, 8: _S, 9: _S, 10: _S, 11: _S},  # A,9
}

# --- 4-Deck Pair Splitting (no Double After Split) ---
# _SP = always split; _S = always stand; _H = always hit
# Special: 8,8 vs 10 → split if TC < +6 ("B" = below)
PAIR_SPLIT_NO_DAS: dict[int, dict[int, int]] = {
    2:  {2: 6,   3: 1,   4: -3,  5: -6,  6: _SP, 7: _SP, 8: _H,  9: _H,  10: _H,  11: _H},
    3:  {2: _H,  3: 4,   4: 0,   5: -2,  6: _SP, 7: _SP, 8: _H,  9: _H,  10: _H,  11: _H},
    # 4,4: play like hard 8 per doubling table
    6:  {2: 2,   3: 0,   4: -1,  5: -4,  6: -5,  7: _H,  8: _H,  9: _H,  10: _H,  11: _H},
    7:  {2: _SP, 3: _SP, 4: _SP, 5: _SP, 6: _SP, 7: _SP, 8: _H,  9: _H,  10: _H,  11: _H},
    8:  {2: _SP, 3: _SP, 4: _SP, 5: _SP, 6: _SP, 7: _SP, 8: _SP, 9: _SP, 10: "B6", 11: _SP},
    9:  {2: -1,  3: -2,  4: -3,  5: -5,  6: -3,  7: 6,   8: _SP, 9: _SP, 10: _S,  11: 6},
    10: {2: _S,  3: _S,  4: 6,   5: 4,   6: 4,   7: _S,  8: _S,  9: _S,  10: _S,  11: _S},
    11: {2: _SP, 3: _SP, 4: _SP, 5: _SP, 6: _SP, 7: _SP, 8: _SP, 9: _SP, 10: _SP, 11: -4},
}
# "B6" means: split if TC < +6 (reverse condition — below)

# --- 4-Deck Pair Splitting (with Double After Split) ---
PAIR_SPLIT_DAS: dict[int, dict[int, int]] = {
    2:  {2: -4,  3: -5,  4: _SP, 5: _SP, 6: _SP, 7: _SP, 8: _H,  9: _H,  10: _H,  11: _H},
    3:  {2: -2,  3: -6,  4: _SP, 5: _SP, 6: _SP, 7: _SP, 8: _H,  9: _H,  10: _H,  11: _H},
    4:  {2: _H,  3: _H,  4: 3,   5: 1,   6: 0,   7: _H,  8: _H,  9: _H,  10: _H,  11: _H},
    6:  {2: -1,  3: -3,  4: -4,  5: -6,  6: _SP, 7: _H,  8: _H,  9: _H,  10: _H,  11: _H},
    7:  {2: _SP, 3: _SP, 4: _SP, 5: _SP, 6: _SP, 7: _SP, 8: 1,   9: _H,  10: _H,  11: _H},
    8:  {2: _SP, 3: _SP, 4: _SP, 5: _SP, 6: _SP, 7: _SP, 8: _SP, 9: _SP, 10: _SP, 11: _SP},
    9:  {2: -2,  3: -3,  4: -5,  5: -6,  6: -5,  7: 3,   8: _SP, 9: _SP, 10: _S,  11: 4},
    10: {2: _S,  3: _S,  4: 6,   5: 4,   6: 4,   7: _S,  8: _S,  9: _S,  10: _S,  11: _S},
    11: {2: _SP, 3: _SP, 4: _SP, 5: _SP, 6: _SP, 7: _SP, 8: _SP, 9: _SP, 10: _SP, 11: -4},
}

# --- 4-Deck Surrender ---
# Player hand value → dealer upcard (7 through 11=Ace)
# Read: SURRENDER if TC >= index
SURRENDER_TABLE: dict[int, dict[int, int]] = {
    # (player_total, specific_hand): {dealer: index}
    # 16 hands (10,6 and 9,7)
    16: {7: _H,  8: 4,  9: 1,  10: -2,  11: 0},
    # 15 hands (10,5 and 9,6)
    15: {7: _H,  8: 5,  9: 2,  10: -1,  11: 1},
    14: {7: _H,  8: 6,  9: 2,  10: 3,   11: 5},  # 10,4 / 8,6
    # 8,8 → split (special case in surrender table)
    # 7,7 → see pair-split table for vs 8; otherwise split
}
# Surrender hand details (Humble & Cooper p.261):
# 10,6 / 9,7 vs dealer 8: index 4; vs 9: 1; vs 10: -2; vs A: 0
# 10,5 / 9,6 / 8,7 vs 8: 5/6/6; vs 9: 2/2/2; vs 10: -1/0/1; vs A: 1/2/2
# 8,8 vs 10: index 1 (surrender if TC >= 1)
# 7,7: always split (per pair table); vs 9: surrender at 4; vs 10: 2; vs A: 4


class HiOpt1Counter(BaseCounter):
    """
    Hi-Opt I counting system as described in Humble & Cooper.

    Features:
    - True count computed as RC / decks_remaining
    - Optional Ace side count for bet sizing adjustment
    - Full index play recommendations (±6 range, 4-deck tables)
    - Exact advantage formula: Advantage = BSE + 0.515 × TC
    """

    def __init__(self, shoe: Shoe, bse: float = -0.54,
                 use_ace_side_count: bool = True):
        """
        Parameters
        ----------
        shoe : Shoe
        bse : float
            Basic Strategy Expectation for the casino rules in use.
            Default: -0.54% (typical 6-deck H17 game).
        use_ace_side_count : bool
            If True, track Aces separately and adjust TC for bet sizing.
            Adds ~0.2% to overall edge. Default: True.
        """
        super().__init__(shoe)
        self.bse = bse
        self.use_ace_side_count = use_ace_side_count
        self._ace_count: int = 0       # Aces seen so far
        self._cards_seen: int = 0

    @property
    def name(self) -> str:
        return "Hi-Opt I"

    @property
    def is_balanced(self) -> bool:
        return True  # Sum of all 52 tags = 0 (Aces and 2s are neutral)

    def reset(self) -> None:
        super().reset()
        self._ace_count = 0
        self._cards_seen = 0

    def card_value(self, card: Card) -> int:
        return HI_OPT1_TAGS.get(card.rank, 0)

    def see_card(self, card: Card) -> None:
        self._running_count += self.card_value(card)
        self._cards_seen += 1
        if card.is_ace:
            self._ace_count += 1

    @property
    def ace_side_count(self) -> int:
        return self._ace_count

    @property
    def ace_adjusted_true_count(self) -> float:
        """
        True count adjusted for Ace richness/poorness.
        Source: Humble & Cooper p.245 — Ace side count adds ~0.2% edge.

        Excess Aces remaining relative to expected → add to TC.
        Deficit of Aces → subtract from TC.
        """
        tc = self.true_count
        if not self.use_ace_side_count:
            return tc
        decks_remaining = self.shoe.decks_remaining
        expected_aces = decks_remaining * 4   # 4 Aces per deck
        aces_remaining = (self.shoe.num_decks * 4) - self._ace_count
        # Deviation from expected, normalised per deck
        ace_deviation = (aces_remaining - expected_aces) / decks_remaining
        return tc + ace_deviation

    @property
    def player_advantage(self) -> float:
        """
        Exact player advantage using Griffin's formula (p.514-516):
            Advantage (%) = BSE + 0.515 × TC

        This is the value you plug into the Kelly Criterion for bet sizing.
        """
        return self.bse + 0.515 * self.ace_adjusted_true_count

    def should_take_insurance(self) -> bool:
        """
        Insurance is profitable when more than 1/3 of remaining cards
        are 10-value. With Hi-Opt I, this corresponds to TC >= +3.
        (Since Hi-Opt I doesn't count Aces, actual threshold may vary.)
        """
        return self.true_count >= 3

    # ------------------------------------------------------------------
    # Index Play Recommendations
    # These translate the full 4-deck strategy tables into actionable
    # suggestions given the current true count.
    # ------------------------------------------------------------------

    def get_hard_standing_action(self, player_total: int,
                                  dealer_upcard_value: int) -> str:
        """
        Should we stand or hit with a hard total?
        Returns 'STAND', 'HIT', or 'BASIC' (follow basic strategy).
        """
        tc = self.true_count
        if player_total >= 18:
            return "STAND"
        row = HARD_STAND.get(player_total)
        if row is None:
            return "BASIC"
        idx = row.get(dealer_upcard_value, _H)
        if idx == _S:
            return "STAND"
        if idx == _H:
            return "HIT"
        return "STAND" if tc >= idx else "HIT"

    def get_hard_double_action(self, player_total: int,
                                dealer_upcard_value: int) -> str:
        """
        Should we double a hard total?
        Returns 'DOUBLE' or 'HIT'.
        """
        tc = self.true_count
        row = HARD_DOUBLE.get(player_total)
        if row is None:
            return "HIT"
        idx = row.get(dealer_upcard_value, _H)
        if idx == _H:
            return "HIT"
        if idx == _D or idx == _S:
            return "DOUBLE"
        return "DOUBLE" if tc >= idx else "HIT"

    def get_surrender_action(self, player_total: int,
                              dealer_upcard_value: int) -> str:
        """
        Should we surrender? Returns 'SURRENDER' or 'CONTINUE'.
        """
        if dealer_upcard_value < 7:
            return "CONTINUE"
        tc = self.true_count
        row = SURRENDER_TABLE.get(player_total)
        if row is None:
            return "CONTINUE"
        idx = row.get(dealer_upcard_value, _H)
        if idx == _H:
            return "CONTINUE"
        return "SURRENDER" if tc >= idx else "CONTINUE"

    def get_pair_split_action(self, pair_value: int,
                               dealer_upcard_value: int,
                               das_allowed: bool = True) -> str:
        """
        Should we split a pair?
        Returns 'SPLIT', 'STAND', 'HIT', or 'BASIC'.
        """
        tc = self.true_count
        table = PAIR_SPLIT_DAS if das_allowed else PAIR_SPLIT_NO_DAS
        row = table.get(pair_value)
        if row is None:
            return "BASIC"
        idx = row.get(dealer_upcard_value, _H)
        if idx == _SP:
            return "SPLIT"
        if idx == _S:
            return "STAND"
        if idx == _H:
            return "HIT"
        if idx == "B6":
            # Special: 8,8 vs 10 — split only if TC < +6
            return "SPLIT" if tc < 6 else "HIT"
        if isinstance(idx, (int, float)):
            return "SPLIT" if tc >= idx else "HIT"
        return "BASIC"

    def get_all_index_plays(self, player_total: int, is_soft: bool,
                             is_pair: bool, pair_value: int,
                             dealer_upcard_value: int,
                             das_allowed: bool = True) -> list[dict]:
        """
        Return all relevant index play recommendations for the current hand.
        Used by the UI to show actionable deviations from basic strategy.
        """
        plays = []
        tc = round(self.true_count, 1)

        # 1. Surrender check (highest priority per Humble & Cooper p.262)
        surr = self.get_surrender_action(player_total, dealer_upcard_value)
        if surr == "SURRENDER":
            plays.append({"action": "SURRENDER", "reason": f"TC={tc:+.1f} → surrender threshold met"})
            return plays  # Surrender overrides everything

        # 2. Pair split
        if is_pair:
            split = self.get_pair_split_action(pair_value, dealer_upcard_value, das_allowed)
            plays.append({"action": split, "reason": f"Pair split index (TC={tc:+.1f})"})
            return plays

        # 3. Hard doubling
        if not is_soft and player_total in HARD_DOUBLE:
            double_action = self.get_hard_double_action(player_total, dealer_upcard_value)
            plays.append({"action": double_action, "reason": f"Hard double index (TC={tc:+.1f})"})

        # 4. Hard standing
        if not is_soft and player_total in HARD_STAND:
            stand_action = self.get_hard_standing_action(player_total, dealer_upcard_value)
            plays.append({"action": stand_action, "reason": f"Hard stand index (TC={tc:+.1f})"})

        return plays

    def __repr__(self) -> str:
        return (f"HiOpt1(RC={self._running_count}, "
                f"TC={self.true_count:+.2f}, "
                f"adv={self.player_advantage:+.3f}%, "
                f"aces_seen={self._ace_count})")
