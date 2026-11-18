import argparse
import threading
import time

import cv2
from flask import Flask, Response, jsonify, render_template

from config import (
    ALERT_COOLDOWN,
    CAMERA_FPS,
    CAMERA_SOURCE,
    CROWD_CLEAR_SECONDS,
    CROWD_LIMIT,
    DISPLAY_HEIGHT,
    DISPLAY_WIDTH,
    FRAME_SKIP,
)
from detection_worker import DetectionWorker
from render import draw_detection, draw_message
from telegram_alert import TelegramNotifier, telegram_is_configured


app = Flask(__name__)
camera_stream = None


class CameraStream:
    def __init__(self, source, crowd_limit, frame_skip, cooldown, clear_seconds, alerts_enabled):
        self.source = source
        self.crowd_limit = crowd_limit
        self.frame_skip = frame_skip
        self.cooldown = cooldown
        self.clear_seconds = clear_seconds
        self.alerts_enabled = alerts_enabled

        self._lock = threading.Lock()
        self._latest_jpeg = None
        self._status = {
            "people_count": 0,
            "crowd_detected": False,
            "status_text": "STARTING",
            "alerts_enabled": alerts_enabled,
            "model_ready": False,
            "error": None,
        }
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

        self.detector_worker = DetectionWorker()
        self.telegram_notifier = TelegramNotifier()

    def start(self):
        self.detector_worker.start()
        self.telegram_notifier.start()
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=2)
        self.telegram_notifier.stop()
        self.detector_worker.stop()

    def latest_jpeg(self):
        with self._lock:
            return self._latest_jpeg

    def latest_status(self):
        with self._lock:
            return dict(self._status)

    def _set_status(self, **updates):
        with self._lock:
            self._status.update(updates)

    def _run(self):
        cap = cv2.VideoCapture(
            self.source, cv2.CAP_DSHOW if isinstance(self.source, int) else 0
        )
        if not cap.isOpened():
            self._set_status(error=f"Could not open video source: {self.source}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        frame_count = 0
        last_alert_time = 0
        crowd_active = False
        below_limit_since = None

        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    self._set_status(error="Video source ended or disconnected.")
                    break

                frame_count += 1
                frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

                if self.detector_worker.error:
                    self._set_status(error=str(self.detector_worker.error))
                    break

                if frame_count % self.frame_skip == 0:
                    self.detector_worker.submit(frame)

                people_count, boxes = self.detector_worker.latest_result()
                crowd_detected, status_text = draw_detection(
                    frame, people_count, boxes, self.crowd_limit
                )

                if not self.detector_worker.ready:
                    draw_message(frame, "Loading AI model...", DISPLAY_HEIGHT - 20)
                elif self.alerts_enabled:
                    draw_message(frame, "Telegram alerts ON", DISPLAY_HEIGHT - 20, (0, 255, 0))
                else:
                    draw_message(frame, "Telegram alerts OFF", DISPLAY_HEIGHT - 20, (0, 255, 255))

                current_time = time.time()

                if crowd_detected:
                    below_limit_since = None
                    can_send_alert = (
                        not crowd_active
                        and self.alerts_enabled
                        and current_time - last_alert_time >= self.cooldown
                    )
                    if can_send_alert:
                        queued = self.telegram_notifier.send(
                            f"WARNING: Crowd detected. People count = {people_count}"
                        )
                        if queued:
                            crowd_active = True
                            last_alert_time = current_time
                elif crowd_active:
                    if below_limit_since is None:
                        below_limit_since = current_time
                    elif current_time - below_limit_since >= self.clear_seconds:
                        crowd_active = False
                        below_limit_since = None
                        if self.alerts_enabled:
                            self.telegram_notifier.send("Crowd cleared. Back to normal.")

                ok, buffer = cv2.imencode(".jpg", frame)
                if ok:
                    with self._lock:
                        self._latest_jpeg = buffer.tobytes()
                        self._status.update(
                            people_count=people_count,
                            crowd_detected=crowd_detected,
                            status_text=status_text,
                            model_ready=self.detector_worker.ready,
                        )
        finally:
            cap.release()

    @property
    def frame_ready(self):
        return self.latest_jpeg() is not None


def parse_source(value):
    if value.isdigit():
        return int(value)
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Crowd detection web dashboard (Flask + YOLOv8)"
    )
    parser.add_argument(
        "--source",
        default=CAMERA_SOURCE,
        help=f"Camera index or video file path. Default: {CAMERA_SOURCE}",
    )
    parser.add_argument(
        "--crowd-limit",
        type=int,
        default=CROWD_LIMIT,
        help=f"Send alert when people count is above this value. Default: {CROWD_LIMIT}",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=FRAME_SKIP,
        help=f"Run detection every N frames. Default: {FRAME_SKIP}",
    )
    parser.add_argument(
        "--cooldown",
        type=int,
        default=ALERT_COOLDOWN,
        help=f"Minimum seconds between two separate crowd alerts. Default: {ALERT_COOLDOWN}",
    )
    parser.add_argument(
        "--clear-seconds",
        type=int,
        default=CROWD_CLEAR_SECONDS,
        help=(
            "Seconds the people count must stay at/below the crowd limit "
            f"before the alert re-arms. Default: {CROWD_CLEAR_SECONDS}"
        ),
    )
    parser.add_argument(
        "--no-alerts",
        action="store_true",
        help="Run detection without sending Telegram messages",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Web server host. Default: 127.0.0.1")
    parser.add_argument("--port", type=int, default=5000, help="Web server port. Default: 5000")
    args = parser.parse_args()
    if args.crowd_limit < 0:
        parser.error("--crowd-limit must be 0 or greater")
    if args.frame_skip <= 0:
        parser.error("--frame-skip must be greater than 0")
    if args.cooldown < 0:
        parser.error("--cooldown must be 0 or greater")
    if args.clear_seconds < 0:
        parser.error("--clear-seconds must be 0 or greater")
    return args


@app.route("/")
def index():
    return render_template("index.html")


def _mjpeg_generator():
    boundary = b"--frame"
    while True:
        jpeg = camera_stream.latest_jpeg()
        if jpeg is not None:
            yield (
                boundary
                + b"\r\nContent-Type: image/jpeg\r\n\r\n"
                + jpeg
                + b"\r\n"
            )
        time.sleep(0.03)


@app.route("/video_feed")
def video_feed():
    return Response(
        _mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/status")
def status():
    return jsonify(camera_stream.latest_status())


def main():
    global camera_stream
    args = parse_args()

    alerts_enabled = not args.no_alerts and telegram_is_configured()
    if args.no_alerts:
        print("Telegram alerts disabled")
    elif not alerts_enabled:
        print("Telegram credentials missing. Detection will run without Telegram alerts.")

    camera_stream = CameraStream(
        source=parse_source(args.source),
        crowd_limit=args.crowd_limit,
        frame_skip=args.frame_skip,
        cooldown=args.cooldown,
        clear_seconds=args.clear_seconds,
        alerts_enabled=alerts_enabled,
    )
    camera_stream.start()

    print(f"Crowd monitoring dashboard: http://{args.host}:{args.port}")

    try:
        app.run(host=args.host, port=args.port, threaded=True, debug=False, use_reloader=False)
    finally:
        camera_stream.stop()


if __name__ == "__main__":
    main()
