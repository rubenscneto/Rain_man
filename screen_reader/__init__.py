"""Screen reader package."""
from screen_reader.capture import ScreenCapture, CaptureRegion
from screen_reader.card_detector import CardDetector, DetectedCard, parse_card_text

__all__ = ["ScreenCapture", "CaptureRegion", "CardDetector", "DetectedCard", "parse_card_text"]
