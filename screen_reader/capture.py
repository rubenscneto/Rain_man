"""
Screen Capture Module.

Captures screenshots of the blackjack game running on screen.
Uses `mss` for fast multi-monitor screen capture.

Can operate in two modes:
  1. FULL_SCREEN  — capture the entire monitor
  2. REGION       — capture a specific bounding box (faster, more accurate)

If screen reading fails or is not available, the system falls back to
manual card input via the terminal UI.
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

try:
    import mss
    import mss.tools
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


@dataclass
class CaptureRegion:
    """Screen region to capture."""
    top: int = 0
    left: int = 0
    width: int = 1920
    height: int = 1080
    monitor: int = 1  # Monitor index (1 = primary)


class ScreenCapture:
    """
    Captures screenshots for card detection.

    Usage
    -----
    capture = ScreenCapture()
    img_array = capture.grab()          # Returns numpy RGB array
    capture.save("screenshot.png")      # Save to disk for debugging
    """

    def __init__(self, region: Optional[CaptureRegion] = None):
        self.region = region or CaptureRegion()
        self._sct = None
        self._available = MSS_AVAILABLE and PIL_AVAILABLE

        if not self._available:
            print("[ScreenCapture] WARNING: mss/PIL not available. "
                  "Screen reading disabled. Use manual input mode.")

    def _get_sct(self):
        if self._sct is None and MSS_AVAILABLE:
            self._sct = mss.mss()
        return self._sct

    def grab(self) -> Optional[np.ndarray]:
        """
        Capture screen region and return as numpy RGB array.
        Returns None if screen capture is unavailable.
        """
        if not self._available:
            return None
        try:
            sct = self._get_sct()
            region = {
                "top": self.region.top,
                "left": self.region.left,
                "width": self.region.width,
                "height": self.region.height,
                "mon": self.region.monitor,
            }
            sct_img = sct.grab(region)
            # Convert to numpy RGB array (mss returns BGRA)
            img = np.array(sct_img)
            # BGRA → RGB
            img_rgb = img[:, :, :3][:, :, ::-1]
            return img_rgb
        except Exception as e:
            print(f"[ScreenCapture] Capture failed: {e}")
            return None

    def grab_pil(self) -> Optional["Image.Image"]:
        """Grab screen and return as PIL Image."""
        arr = self.grab()
        if arr is None or not PIL_AVAILABLE:
            return None
        return Image.fromarray(arr)

    def save(self, path: str) -> bool:
        """Save current screen capture to file (for debugging)."""
        img = self.grab_pil()
        if img is None:
            return False
        img.save(path)
        return True

    def list_monitors(self) -> list:
        """List available monitors."""
        if not MSS_AVAILABLE:
            return []
        sct = self._get_sct()
        return list(sct.monitors)

    def set_region_interactive(self) -> CaptureRegion:
        """
        Helper to define the capture region by printing monitor info.
        The user specifies the blackjack table area manually.
        """
        monitors = self.list_monitors()
        print("\n[ScreenCapture] Available monitors:")
        for i, m in enumerate(monitors):
            print(f"  Monitor {i}: {m}")
        print("\nCurrent region:", self.region)
        print("To change region, edit config.py → ScreenReaderConfig.capture_region")
        return self.region

    def is_available(self) -> bool:
        return self._available

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self._sct:
            self._sct.close()
