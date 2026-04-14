"""
Screen Region Calibration Tool.

Before watch mode can work, you need to tell Rainman exactly WHERE the
game appears on your screen. This tool helps you do that interactively.

Usage:
    python main.py --mode calibrate

The tool:
  1. Takes a screenshot of your full screen
  2. Saves it to /tmp/rainman_calibration.png  (or rainman_calibration.png)
  3. Shows the monitor dimensions
  4. Guides you to set x, y, width, height for the game window
  5. Saves the region config to screen_region.json

For Chrome with Evolution Gaming, typical setup:
  - Open Chrome full-screen on your primary monitor
  - Navigate to the blackjack game
  - Run calibration WHILE the game is visible
  - The table area (green felt) is usually in the CENTER of the stream
"""

from __future__ import annotations
import json
import os

REGION_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "screen_region.json")


def load_region() -> dict:
    """Load saved capture region from disk."""
    if os.path.exists(REGION_CONFIG_PATH):
        with open(REGION_CONFIG_PATH) as f:
            return json.load(f)
    # Defaults: full 1920×1080 primary monitor
    return {"top": 0, "left": 0, "width": 1920, "height": 1080, "monitor": 1}


def save_region(region: dict) -> None:
    """Save capture region to disk."""
    with open(REGION_CONFIG_PATH, "w") as f:
        json.dump(region, f, indent=2)
    print(f"[Calibrate] Region saved to {REGION_CONFIG_PATH}")


def run_calibration() -> dict:
    """Interactive calibration wizard."""
    try:
        from screen_reader.capture import ScreenCapture, CaptureRegion
        import numpy as np
    except ImportError:
        print("Screen capture dependencies not available.")
        return load_region()

    print("\n" + "=" * 60)
    print("  RAINMAN SCREEN CALIBRATION")
    print("=" * 60)
    print("\nStep 1: Taking a screenshot of your full screen...")

    cap = ScreenCapture()
    monitors = cap.list_monitors()
    if monitors:
        print("\nDetected monitors:")
        for i, m in enumerate(monitors):
            print(f"  Monitor {i}: {m}")

    # Take full screenshot
    full = cap.grab()
    if full is not None:
        # Try to save it
        try:
            from PIL import Image
            img = Image.fromarray(full)
            save_path = "/tmp/rainman_calibration.png"
            img.save(save_path)
            print(f"\n[OK] Screenshot saved: {save_path}")
            print(f"     Size: {full.shape[1]}×{full.shape[0]} pixels")
            print(f"\n  Open {save_path} to see your screen and note")
            print("  the coordinates of the game window/table area.")
        except Exception as e:
            print(f"Could not save screenshot: {e}")
    else:
        print("[WARN] Could not capture screen — using defaults.")

    print("\nStep 2: Set the capture region.")
    print("  This is the area Rainman will monitor for cards.")
    print("  For Chrome full-screen Evolution Gaming, typical values:")
    print("  - Start with the FULL screen: top=0, left=0, width=1920, height=1080")
    print("  - Then narrow it to just the TABLE AREA for better accuracy:")
    print("    e.g. top=80, left=0, width=940, height=550")
    print()

    current = load_region()
    print(f"  Current region: {current}")
    print()

    try:
        change = input("  Change region? (y/n) [n]: ").strip().lower()
        if change != "y":
            print("[Calibrate] Keeping current region.")
            return current

        print("\n  Enter new values (press Enter to keep current):")
        fields = ["top", "left", "width", "height", "monitor"]
        region = {}
        for field in fields:
            cur = current.get(field, 0)
            val_str = input(f"    {field} [{cur}]: ").strip()
            region[field] = int(val_str) if val_str else cur

        print(f"\n  New region: {region}")
        confirm = input("  Save this region? (y/n) [y]: ").strip().lower()
        if confirm != "n":
            save_region(region)
            return region
        return current

    except (EOFError, KeyboardInterrupt):
        return current


def get_capture_region():
    """Get the configured capture region as a CaptureRegion object."""
    from screen_reader.capture import CaptureRegion
    data = load_region()
    return CaptureRegion(
        top=data.get("top", 0),
        left=data.get("left", 0),
        width=data.get("width", 1920),
        height=data.get("height", 1080),
        monitor=data.get("monitor", 1),
    )
