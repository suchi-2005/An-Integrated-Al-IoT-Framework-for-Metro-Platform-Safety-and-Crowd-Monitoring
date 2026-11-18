import warnings
import logging
from contextlib import redirect_stderr, redirect_stdout
from os import devnull

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

from ultralytics import YOLO
from ultralytics.utils import LOGGER

from config import DETECTION_CONFIDENCE, MODEL_IMAGE_SIZE, MODEL_PATH


PERSON_CLASS_ID = 0


class CrowdDetector:
    def __init__(self, model_path=MODEL_PATH):
        LOGGER.setLevel(logging.ERROR)
        with open(devnull, "w", encoding="utf-8") as sink:
            with redirect_stdout(sink), redirect_stderr(sink):
                self.model = YOLO(model_path)
                self.model.to("cpu")
        try:
            with open(devnull, "w", encoding="utf-8") as sink:
                with redirect_stdout(sink), redirect_stderr(sink):
                    self.model.fuse()
        except Exception:
            pass

    def detect_people(self, frame):
        results = self.model(
            frame,
            imgsz=MODEL_IMAGE_SIZE,
            conf=DETECTION_CONFIDENCE,
            classes=[PERSON_CLASS_ID],
            verbose=False,
        )

        boxes = []
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) != PERSON_CLASS_ID:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                boxes.append((x1, y1, x2, y2, confidence))

        return len(boxes), boxes


_detector = None


def detect_people(frame):
    global _detector
    if _detector is None:
        _detector = CrowdDetector()
    return _detector.detect_people(frame)
