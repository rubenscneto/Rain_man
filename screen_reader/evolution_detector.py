"""
Evolution Gaming Live Dealer — Specialized Card Detector.

Analysis of the live stream screenshot:
  - Platform   : Evolution Gaming (Blackjack A, R$25-5.000)
  - Stream type: Live HD video in Chrome browser
  - Table felt : Dark green background
  - Cards      : White/cream rectangles, rank+suit visible in corners
  - Overlays   : Digital hand totals (18, 16, 1/11 etc.) per player

WHY standard OCR is hard here:
  1. Video stream = JPEG compression artifacts on card text
  2. Cards can be partially overlapping when dealt face-up
  3. Motion blur during the 0.5–1s dealing animation
  4. Camera angle creates slight perspective distortion

STRATEGY (multi-layer, most reliable first):
  1. Frame differencing  → detect moment new cards appear (trigger)
  2. Color segmentation  → isolate white card regions on green felt (OpenCV)
  3. Corner crop + OCR   → read rank/suit from card top-left corner
  4. Score overlay parse → read the digital "18", "1/11" totals as backup
  5. Hotkey fallback     → single-key input if OCR confidence < threshold

For counting purposes, we need individual cards (not totals):
  "Player has 18" could be K+8 (Hi-Opt: -1) or A+7 (Hi-Opt: 0) — different!
"""

from __future__ import annotations
import time
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image, ImageEnhance, ImageFilter
    TESS_AVAILABLE = True
except ImportError:
    TESS_AVAILABLE = False

from simulator.deck import Card, RANKS
from screen_reader.capture import ScreenCapture, CaptureRegion
from screen_reader.card_detector import DetectedCard, parse_card_text
from screen_reader.deck_estimator import DeckEstimator, DeckEstimate


# ---------------------------------------------------------------------------
# Color ranges for Evolution Gaming table (HSV)
# ---------------------------------------------------------------------------

# Green felt (table background)  — used to mask OUT non-card regions
FELT_HSV_LOW  = np.array([35, 30, 30])    # dark/mid green
FELT_HSV_HIGH = np.array([90, 255, 180])

# Card face (white/cream)
CARD_HSV_LOW  = np.array([0,  0,  160])   # low saturation, high brightness
CARD_HSV_HIGH = np.array([180, 60, 255])

# Red suit symbols (♥ ♦)
RED_HSV_LOW  = np.array([0,  100, 100])
RED_HSV_HIGH = np.array([15, 255, 255])

# Minimum card area in pixels (at 1080p, a card is ~70×100px)
MIN_CARD_AREA  = 2_500    # 50×50
MAX_CARD_AREA  = 30_000   # ~200×150


@dataclass
class GameFrame:
    """Parsed state of one captured frame."""
    timestamp: float
    cards_detected: List[DetectedCard] = field(default_factory=list)
    scores_detected: List[int] = field(default_factory=list)    # e.g. [18, 16, 12]
    new_round_detected: bool = False    # "PRÓXIMO JOGO EM BREVE" visible
    shuffle_detected: bool = False      # "MISTURA EM ANDAMENTO" visible
    frame_hash: int = 0                 # used for change detection

    @property
    def new_cards_vs(self, prev: "GameFrame") -> List[DetectedCard]:
        """Cards that are in this frame but not in prev."""
        prev_strs = {str(c) for c in prev.cards_detected}
        return [c for c in self.cards_detected if str(c) not in prev_strs]


class EvolutionDetector:
    """
    Specialized card detector for Evolution Gaming live dealer blackjack.

    Workflow:
        detector = EvolutionDetector(region=CaptureRegion(...))
        for frame in detector.stream():
            for card in frame.new_cards:
                counter.see_card(card.card)
    """

    # OCR minimum confidence to accept a card detection
    CONFIDENCE_THRESHOLD = 0.5

    # Approximate card corners to crop for OCR (fraction of card size)
    CORNER_CROP_X = 0.28
    CORNER_CROP_Y = 0.30

    def __init__(self, region: Optional[CaptureRegion] = None,
                 poll_ms: int = 300,
                 on_card: Optional[Callable[[DetectedCard], None]] = None,
                 on_new_round: Optional[Callable[[], None]] = None,
                 on_deck_estimate: Optional[Callable[["DeckEstimate"], None]] = None):
        """
        Parameters
        ----------
        region : CaptureRegion
            Screen area to capture (set by calibration tool).
        poll_ms : int
            Polling interval in milliseconds (default: 300ms = ~3fps).
        on_card : callable
            Called with DetectedCard whenever a new card is detected.
        on_new_round : callable
            Called when "PRÓXIMO JOGO EM BREVE" / new round is detected.
        on_deck_estimate : callable
            Called with DeckEstimate at the end of each shuffle phase.
        """
        self.capture = ScreenCapture(region)
        self.poll_ms = poll_ms
        self.on_card = on_card
        self.on_new_round = on_new_round
        self.on_deck_estimate = on_deck_estimate
        self._prev_frame: Optional[np.ndarray] = None
        self._prev_cards: List[str] = []   # str(card) of already-counted cards
        self._running = False
        self._available = CV2_AVAILABLE and TESS_AVAILABLE

        # Deck estimator — accumulates measurements during shuffle phase
        self._deck_estimator = DeckEstimator()
        self._in_shuffle = False           # True while "MISTURA EM ANDAMENTO" active
        self.last_deck_estimate: Optional[DeckEstimate] = None

        if not self._available:
            print("[EvolutionDetector] OpenCV or Tesseract not available.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start continuous polling loop (blocking)."""
        if not self._available:
            print("[EvolutionDetector] Cannot start — missing dependencies.")
            return
        self._running = True
        print(f"[EvolutionDetector] Watching screen at {1000//self.poll_ms}fps...")
        try:
            while self._running:
                t0 = time.time()
                self._process_frame()
                elapsed = (time.time() - t0) * 1000
                sleep_ms = max(0, self.poll_ms - elapsed)
                time.sleep(sleep_ms / 1000.0)
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False

    def stop(self) -> None:
        self._running = False

    def capture_once(self) -> Optional[GameFrame]:
        """Capture and analyse a single frame (for testing)."""
        img = self.capture.grab()
        if img is None:
            return None
        return self._analyse_frame(img)

    def is_available(self) -> bool:
        return self._available and self.capture.is_available()

    # ------------------------------------------------------------------
    # Internal frame processing
    # ------------------------------------------------------------------

    def _process_frame(self) -> None:
        img = self.capture.grab()
        if img is None:
            return

        frame = self._analyse_frame(img)
        if frame is None:
            return

        # ── Shuffle phase handling ──────────────────────────────────────
        if frame.shuffle_detected:
            if not self._in_shuffle:
                # Mistura acabou de começar → resetar acumulador
                self._in_shuffle = True
                self._deck_estimator.reset()
                print("[EvolutionDetector] Mistura detectada — estimando baralhos...")
            # Acumular estimativa de baralhos neste frame
            self._deck_estimator.add_frame(img)
        else:
            if self._in_shuffle:
                # Mistura terminou → consolidar estimativa
                self._in_shuffle = False
                estimate = self._deck_estimator.consolidate()
                if estimate is not None:
                    self.last_deck_estimate = estimate
                    print(f"[EvolutionDetector] {estimate}")
                    if self.on_deck_estimate:
                        self.on_deck_estimate(estimate)

        # ── New round ──────────────────────────────────────────────────
        if frame.new_round_detected and self.on_new_round:
            self._prev_cards.clear()
            self.on_new_round()

        # ── Report new cards ───────────────────────────────────────────
        for det in frame.cards_detected:
            key = f"{det.rank}{det.suit}"
            if key not in self._prev_cards and det.confidence >= self.CONFIDENCE_THRESHOLD:
                self._prev_cards.append(key)
                if self.on_card:
                    self.on_card(det)

        self._prev_frame = img

    def _analyse_frame(self, img: np.ndarray) -> Optional[GameFrame]:
        """Full analysis pipeline on one frame."""
        frame = GameFrame(timestamp=time.time())

        # 1. Check if frame changed significantly (skip static frames)
        frame_hash = self._frame_hash(img)
        if not self._has_changed(img):
            return frame

        frame.frame_hash = frame_hash

        # 2. Detect round/shuffle text overlays
        frame.new_round_detected, frame.shuffle_detected = self._detect_overlays(img)

        # 3. Find card regions using color segmentation
        card_regions = self._find_card_regions(img)

        # 4. OCR each card
        for (x, y, w, h) in card_regions:
            card_img = img[y:y+h, x:x+w]
            det = self._ocr_card(card_img, bbox=(x, y, w, h))
            if det is not None:
                frame.cards_detected.append(det)

        # 5. Parse digital score overlays as supplementary data
        frame.scores_detected = self._detect_score_overlays(img)

        return frame

    def _has_changed(self, img: np.ndarray) -> bool:
        """True if frame changed enough to warrant full analysis."""
        if self._prev_frame is None or img.shape != self._prev_frame.shape:
            self._prev_frame = img
            return True
        # Compare a downsampled version for speed
        small = cv2.resize(img, (160, 90))
        prev_small = cv2.resize(self._prev_frame, (160, 90))
        diff = np.mean(np.abs(small.astype(float) - prev_small.astype(float)))
        return diff > 2.0   # Threshold: >2 mean pixel difference = change

    @staticmethod
    def _frame_hash(img: np.ndarray) -> int:
        small = cv2.resize(img, (32, 18))
        return hash(small.tobytes())

    def _find_card_regions(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Find white card rectangles on the green felt using color segmentation.

        HSV approach:
          1. Convert to HSV
          2. Threshold for white/cream pixels (low saturation, high value)
          3. Morphological operations to fill card body
          4. Find contours → filter by size and aspect ratio
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # Mask: white/cream pixels (card faces)
        card_mask = cv2.inRange(hsv, CARD_HSV_LOW, CARD_HSV_HIGH)

        # Morphological close to fill gaps between card rank/suit and body
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        card_mask = cv2.morphologyEx(card_mask, cv2.MORPH_CLOSE, kernel)
        card_mask = cv2.morphologyEx(card_mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(card_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        h_img, w_img = img.shape[:2]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (MIN_CARD_AREA <= area <= MAX_CARD_AREA):
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / (h + 1e-6)
            # Cards are roughly portrait 0.5–0.85 aspect or landscape 1.0–2.0 (face-up fan)
            if not (0.3 < aspect < 2.2):
                continue
            # Filter out edge fragments
            if x < 2 or y < 2 or x + w > w_img - 2 or y + h > h_img - 2:
                continue
            regions.append((x, y, w, h))

        # Sort left-to-right, top-to-bottom
        regions.sort(key=lambda r: (r[1] // 60, r[0]))
        return regions

    def _ocr_card(self, card_img: np.ndarray,
                   bbox: Tuple) -> Optional[DetectedCard]:
        """OCR the rank and suit from the top-left corner of a card."""
        if not TESS_AVAILABLE:
            return None
        h, w = card_img.shape[:2]
        if h < 20 or w < 20:
            return None

        # Crop top-left corner (where rank+suit are printed)
        cy = max(1, int(h * self.CORNER_CROP_Y))
        cx = max(1, int(w * self.CORNER_CROP_X))
        corner = card_img[:cy, :cx]

        # Preprocess: grayscale → upscale → threshold
        pil_corner = Image.fromarray(corner)
        # 4× upscale for better OCR
        scale = max(1, int(80 / min(cy, cx)))
        new_size = (pil_corner.width * scale, pil_corner.height * scale)
        pil_corner = pil_corner.resize(new_size, Image.LANCZOS)
        # Enhance contrast
        pil_corner = ImageEnhance.Contrast(pil_corner).enhance(2.5)
        # Convert to grayscale and threshold
        gray = pil_corner.convert("L")
        # Invert if needed (dark text on white background OR white on dark)
        arr = np.array(gray)
        if arr.mean() < 128:
            gray = Image.fromarray(255 - arr)

        config = (
            "--psm 6 --oem 3 "
            "-c tessedit_char_whitelist=A23456789TJQK♠♥♦♣SHDCshdc0 "
        )
        try:
            text = pytesseract.image_to_string(gray, config=config).strip()
        except Exception:
            return None

        result = parse_card_text(text)
        if result is None:
            return None

        rank, suit = result
        if rank not in RANKS:
            return None

        return DetectedCard(
            rank=rank, suit=suit,
            confidence=0.65,
            bounding_box=bbox,
        )

    def _detect_overlays(self, img: np.ndarray) -> tuple:
        """
        Detecta overlays de texto no centro da tela:
          - "PRÓXIMO JOGO EM BREVE" → nova rodada começando
          - "MISTURA EM ANDAMENTO"  → mistura em progresso (todos baralhos visíveis)

        Retorna (new_round: bool, shuffle: bool).
        """
        if not TESS_AVAILABLE:
            return False, False
        h, w = img.shape[:2]
        # Overlay aparece na faixa central da tela
        center_band = img[h//3:h//2, w//4:3*w//4]
        gray = cv2.cvtColor(center_band, cv2.COLOR_RGB2GRAY)
        bright_ratio = np.sum(gray > 210) / gray.size

        new_round = False
        shuffle   = False

        if bright_ratio > 0.04:
            try:
                pil  = Image.fromarray(center_band)
                text = pytesseract.image_to_string(pil, config="--psm 6").upper()

                new_round_kw = ["PROXIMO", "PRÓXIMO", "BREVE", "NEXT GAME",
                                "PLACE YOUR BETS", "FAÇA SUAS"]
                shuffle_kw   = ["MISTURA", "SHUFFLING", "EMBARALH"]

                new_round = any(k in text for k in new_round_kw)
                shuffle   = any(k in text for k in shuffle_kw)
            except Exception:
                pass

        return new_round, shuffle

    # mantém compatibilidade com chamadas legadas
    def _detect_new_round(self, img: np.ndarray) -> bool:
        new_round, _ = self._detect_overlays(img)
        return new_round

    def _detect_score_overlays(self, img: np.ndarray) -> List[int]:
        """
        Parse the digital score overlays (e.g. "18", "1/11") shown
        over each player position. Useful as supplementary information.
        """
        if not TESS_AVAILABLE:
            return []
        # Score overlays are in the lower third of the screen, in white bubbles
        h, w = img.shape[:2]
        lower = img[int(h * 0.6):, :]
        try:
            pil = Image.fromarray(lower)
            text = pytesseract.image_to_string(pil, config="--psm 6")
            scores = []
            for token in text.split():
                clean = token.strip(".,")
                if clean.isdigit():
                    val = int(clean)
                    if 2 <= val <= 21:
                        scores.append(val)
                elif "/" in clean:
                    # "1/11" style Ace display
                    parts = clean.split("/")
                    if len(parts) == 2 and all(p.isdigit() for p in parts):
                        scores.append(int(parts[1]))  # Use the higher value
            return scores
        except Exception:
            return []

    def debug_frame(self, save_path: str = "/tmp/debug_frame.png") -> None:
        """Save an annotated debug frame showing detected card regions."""
        img = self.capture.grab()
        if img is None:
            print("No frame captured.")
            return
        frame = self._analyse_frame(img)
        debug = img.copy()
        if frame:
            for det in frame.cards_detected:
                if det.bounding_box:
                    x, y, w, h = det.bounding_box
                    cv2.rectangle(debug, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(debug, f"{det.rank}{det.suit}",
                                (x, y-5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2)
        rgb_debug = cv2.cvtColor(debug, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, rgb_debug)
        print(f"[Debug] Frame saved to {save_path}")
        print(f"[Debug] Cards detected: {[str(d) for d in frame.cards_detected]}")
        print(f"[Debug] Scores detected: {frame.scores_detected}")
        print(f"[Debug] New round: {frame.new_round_detected}")
