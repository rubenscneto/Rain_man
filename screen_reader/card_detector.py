"""
Card Detector — OCR + Computer Vision.

Detects playing cards from screen captures using:
  1. OpenCV for image preprocessing (thresholding, contour detection)
  2. Pytesseract (Tesseract OCR) for rank/suit text extraction

Pipeline:
  screenshot → grayscale → threshold → find contours →
  crop card regions → OCR each card → parse rank/suit

Limitations:
  - Requires a clear digital representation of cards on screen
  - Works best with standard casino software card graphics
  - Physical cards (live stream) require better image quality
  - Falls back gracefully to manual input if confidence is low

The "caixinha" (shoe) affects detection only insofar as the cards
are in the frame — the detector reads whatever is visible.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import pytesseract
    TESS_AVAILABLE = True
except ImportError:
    TESS_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from simulator.deck import Card, RANKS, SUITS


@dataclass
class DetectedCard:
    """A card detected from the screen."""
    rank: str
    suit: str
    confidence: float           # OCR confidence (0.0 – 1.0)
    bounding_box: Optional[Tuple[int, int, int, int]] = None  # x,y,w,h

    @property
    def card(self) -> Optional[Card]:
        if self.rank in RANKS:
            # Map suit symbol if needed
            suit_map = {"S": "♠", "H": "♥", "D": "♦", "C": "♣",
                        "♠": "♠", "♥": "♥", "♦": "♦", "♣": "♣",
                        "s": "♠", "h": "♥", "d": "♦", "c": "♣"}
            suit = suit_map.get(self.suit, self.suit)
            if suit in SUITS:
                return Card(self.rank, suit)
        return None

    def __str__(self) -> str:
        return f"{self.rank}{self.suit} (conf={self.confidence:.0%})"


# Rank aliases that OCR commonly confuses
RANK_ALIASES: dict[str, str] = {
    "1": "A", "I": "A",          # Ace misread
    "0": "10", "O": "10",         # Zero/O → 10
    "l": "J", "L": "J",          # J misread
    "Q ": "Q", " Q": "Q",
    "K ": "K", " K": "K",
    "A ": "A", " A": "A",
}

# Suit aliases
SUIT_ALIASES: dict[str, str] = {
    "S": "♠", "SPADE": "♠", "SPADES": "♠",
    "H": "♥", "HEART": "♥", "HEARTS": "♥",
    "D": "♦", "DIAMOND": "♦", "DIAMONDS": "♦",
    "C": "♣", "CLUB": "♣", "CLUBS": "♣",
}


def parse_card_text(text: str) -> Optional[Tuple[str, str]]:
    """
    Parse OCR text into (rank, suit) tuple.
    Handles common OCR errors and multiple formats:
    "AS", "A♠", "10H", "K of Hearts", etc.
    """
    text = text.strip().upper()
    if not text:
        return None

    # Normalize suit words
    for alias, sym in SUIT_ALIASES.items():
        text = text.replace(alias, sym)

    # Try: rank + suit symbol (e.g., "A♠", "10♥")
    pattern = r"([A-Z]|\d{1,2})\s*([♠♥♦♣])"
    m = re.search(pattern, text)
    if m:
        rank_raw = m.group(1)
        suit = m.group(2)
        rank = RANK_ALIASES.get(rank_raw, rank_raw)
        if rank in RANKS:
            return rank, suit

    # Try: just rank letters (e.g., "A", "K", "10")
    rank_only = re.search(r"\b(10|[AKQJ2-9])\b", text)
    if rank_only:
        rank = RANK_ALIASES.get(rank_only.group(1), rank_only.group(1))
        if rank in RANKS:
            # Unknown suit — use placeholder
            return rank, "♠"

    return None


class CardDetector:
    """
    Detects cards in a screenshot using OpenCV + Tesseract OCR.

    Usage
    -----
    detector = CardDetector()
    cards = detector.detect(img_array)  # → List[DetectedCard]
    """

    def __init__(self, min_confidence: float = 0.5):
        self.min_confidence = min_confidence
        self._available = CV2_AVAILABLE and TESS_AVAILABLE and PIL_AVAILABLE

        if not self._available:
            missing = []
            if not CV2_AVAILABLE:
                missing.append("opencv-python")
            if not TESS_AVAILABLE:
                missing.append("pytesseract")
            print(f"[CardDetector] WARNING: Missing {missing}. "
                  "Card detection disabled. Use manual input.")

    def detect(self, img: np.ndarray) -> List[DetectedCard]:
        """
        Detect all cards in the image.

        Parameters
        ----------
        img : np.ndarray
            RGB image array from ScreenCapture.grab()

        Returns
        -------
        List[DetectedCard]
            Detected cards sorted left-to-right.
        """
        if not self._available or img is None:
            return []

        try:
            return self._detect_via_contours(img)
        except Exception as e:
            print(f"[CardDetector] Detection failed: {e}")
            return []

    def _detect_via_contours(self, img: np.ndarray) -> List[DetectedCard]:
        """Detect card contours and OCR each one."""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Adaptive threshold to handle varying lighting
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=11, C=2
        )

        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detected = []
        h, w = img.shape[:2]

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter by area: cards are large-ish rectangles
            if area < 1000 or area > w * h * 0.3:
                continue

            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = cw / (ch + 1e-6)
            # Cards are roughly portrait-oriented (aspect ~0.6–0.8)
            if not (0.4 < aspect < 1.5):
                continue

            # Crop card region
            card_img = img[y:y+ch, x:x+cw]
            if card_img.size == 0:
                continue

            # Focus on top-left corner where rank/suit are printed
            corner_h = max(1, ch // 4)
            corner_w = max(1, cw // 4)
            corner = card_img[:corner_h, :corner_w]

            # OCR the corner
            result = self._ocr_region(corner)
            if result:
                rank, suit = result
                detected.append(DetectedCard(
                    rank=rank,
                    suit=suit,
                    confidence=0.7,   # Base confidence from contour detection
                    bounding_box=(x, y, cw, ch),
                ))

        # Sort left-to-right by x position
        detected.sort(key=lambda d: d.bounding_box[0] if d.bounding_box else 0)
        return detected

    def _ocr_region(self, img_region: np.ndarray) -> Optional[Tuple[str, str]]:
        """Run Tesseract OCR on a small card corner region."""
        if not TESS_AVAILABLE:
            return None
        try:
            pil_img = Image.fromarray(img_region)
            # Scale up for better OCR accuracy
            pil_img = pil_img.resize(
                (pil_img.width * 4, pil_img.height * 4),
                Image.LANCZOS
            )
            # Tesseract config: single character/word, whitelist
            config = (
                "--psm 6 "
                "-c tessedit_char_whitelist="
                "0123456789AKQJ♠♥♦♣SHDCshdc"
            )
            text = pytesseract.image_to_string(pil_img, config=config)
            return parse_card_text(text)
        except Exception:
            return None

    def detect_from_text_overlay(self, text_regions: List[str]) -> List[DetectedCard]:
        """
        Parse cards from text strings (e.g. from a game's text overlay).
        Format examples: "AS", "10H", "KD", "A♠", "10♥"
        """
        cards = []
        for text in text_regions:
            result = parse_card_text(text)
            if result:
                rank, suit = result
                cards.append(DetectedCard(rank=rank, suit=suit, confidence=1.0))
        return cards

    def is_available(self) -> bool:
        return self._available
