"""Screen reader package."""
from screen_reader.capture import ScreenCapture, CaptureRegion
from screen_reader.card_detector import CardDetector, DetectedCard, parse_card_text
from screen_reader.evolution_detector import EvolutionDetector, GameFrame
from screen_reader.calibrate import run_calibration, get_capture_region, load_region, save_region

__all__ = [
    "ScreenCapture", "CaptureRegion",
    "CardDetector", "DetectedCard", "parse_card_text",
    "EvolutionDetector", "GameFrame",
    "run_calibration", "get_capture_region", "load_region", "save_region",
]
