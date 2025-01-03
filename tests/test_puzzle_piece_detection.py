from puzzle_piece_detection.puzzle_piece_detection import load_model
from ultralytics import YOLO


def test_load_model():
    model = load_model()
    assert isinstance(model, YOLO)
