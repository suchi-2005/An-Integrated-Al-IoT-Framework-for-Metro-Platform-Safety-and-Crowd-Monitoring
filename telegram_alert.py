import queue
import threading
import time

import requests

from config import BOT_TOKEN, CHAT_ID, TELEGRAM_RETRY_COOLDOWN


INVALID_VALUES = {
    "",
    "replace_with_your_bot_token",
    "replace_with_your_chat_id",
    "your_bot_token",
    "your_chat_id",
}


def telegram_is_configured():
    return BOT_TOKEN not in INVALID_VALUES and CHAT_ID not in INVALID_VALUES


def _safe_error_message(exc):
    if isinstance(exc, requests.Timeout):
        return "Telegram timeout. Check internet connection or Telegram access."
    if isinstance(exc, requests.ConnectionError):
        return "Telegram connection failed. Check internet connection or firewall."
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return f"Telegram HTTP error: {exc.response.status_code}"
    return f"Telegram error: {type(exc).__name__}"


def _send_message_now(message):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID,
        "text": message,
    }

    response = requests.post(url, data=data, timeout=(2, 4))
    response.raise_for_status()


class TelegramNotifier:
    def __init__(self):
        self._queue = queue.Queue(maxsize=4)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started = False
        self._paused_until = 0

    def start(self):
        if not telegram_is_configured() or self._started:
            return
        self._started = True
        self._thread.start()

    def send(self, message):
        if not telegram_is_configured():
            return False

        if time.time() < self._paused_until:
            return False

        if self._queue.full():
            return False

        try:
            self._queue.put_nowait(message)
            return True
        except queue.Full:
            return False

    def stop(self):
        self._stop_event.set()
        if self._started:
            self._thread.join(timeout=1)

    def _run(self):
        while not self._stop_event.is_set():
            try:
                message = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                _send_message_now(message)
                print("Telegram alert sent")
            except requests.RequestException as exc:
                print(_safe_error_message(exc))
                self._paused_until = time.time() + TELEGRAM_RETRY_COOLDOWN


def send_telegram_alert(message):
    if not telegram_is_configured():
        print("Telegram alert skipped: credentials missing in .env")
        return False

    try:
        _send_message_now(message)
        print("Telegram alert sent")
        return True
    except requests.RequestException as exc:
        print(_safe_error_message(exc))
        return False
