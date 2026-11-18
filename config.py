import os


def _load_dotenv(path=".env"):
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            name, value = line.split("=", 1)
            name = name.strip()
            value = value.strip().strip('"').strip("'")
            if name and name not in os.environ:
                os.environ[name] = value


def _get_int(name, default):
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {value!r}") from None


def _get_float(name, default):
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {value!r}") from None


_load_dotenv()

BOT_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN") or "").strip()
CHAT_ID = (os.getenv("TELEGRAM_CHAT_ID") or os.getenv("CHAT_ID") or "").strip()

MODEL_PATH = os.getenv("MODEL_PATH", "yolov8s.pt")
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "0")
CROWD_LIMIT = _get_int("CROWD_LIMIT", 4)
FRAME_SKIP = _get_int("FRAME_SKIP", 2)
ALERT_COOLDOWN = _get_int("ALERT_COOLDOWN", 20)
CROWD_CLEAR_SECONDS = _get_int("CROWD_CLEAR_SECONDS", 5)
TELEGRAM_RETRY_COOLDOWN = _get_int("TELEGRAM_RETRY_COOLDOWN", 60)
DETECTION_CONFIDENCE = _get_float("DETECTION_CONFIDENCE", 0.38)
MODEL_IMAGE_SIZE = _get_int("MODEL_IMAGE_SIZE", 640)
DISPLAY_WIDTH = _get_int("DISPLAY_WIDTH", 640)
DISPLAY_HEIGHT = _get_int("DISPLAY_HEIGHT", 480)
CAMERA_FPS = _get_int("CAMERA_FPS", 30)
