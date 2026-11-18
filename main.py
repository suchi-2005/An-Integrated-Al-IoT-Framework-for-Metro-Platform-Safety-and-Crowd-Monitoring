import argparse
import time

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


WINDOW_NAME = "Crowd Monitoring"
cv2 = None


def parse_source(value):
    if value.isdigit():
        return int(value)
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Crowd detection system with Telegram alerts"
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
    parser.add_argument(
        "--test-telegram",
        action="store_true",
        help="Send one Telegram test message and exit",
    )
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


def configure_camera(cap):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def main():
    args = parse_args()

    from telegram_alert import send_telegram_alert, telegram_is_configured

    if args.test_telegram:
        if not telegram_is_configured():
            raise RuntimeError(
                "Telegram credentials missing. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID."
            )
        send_telegram_alert("Crowd detection Telegram test message")
        return

    global cv2
    import cv2 as cv2_module

    cv2 = cv2_module
    source = parse_source(args.source)
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW if isinstance(source, int) else 0)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    configure_camera(cap)

    from telegram_alert import TelegramNotifier

    alerts_enabled = not args.no_alerts and telegram_is_configured()
    if args.no_alerts:
        print("Telegram alerts disabled")
    elif not alerts_enabled:
        print("Telegram credentials missing. Detection will run without Telegram alerts.")

    frame_count = 0
    last_alert_time = 0
    crowd_active = False
    below_limit_since = None
    people_count = 0
    boxes = []
    detector_worker = DetectionWorker()
    detector_worker.start()
    telegram_notifier = TelegramNotifier()
    telegram_notifier.start()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

            if detector_worker.error:
                raise RuntimeError(f"Detection failed: {detector_worker.error}")

            if frame_count % args.frame_skip == 0:
                detector_worker.submit(frame)

            people_count, boxes = detector_worker.latest_result()
            crowd_detected, _ = draw_detection(frame, people_count, boxes, args.crowd_limit)

            if not detector_worker.ready:
                draw_message(frame, "Loading AI model...", DISPLAY_HEIGHT - 20)
            elif alerts_enabled:
                draw_message(frame, "Telegram alerts ON", DISPLAY_HEIGHT - 20, (0, 255, 0))
            elif not alerts_enabled:
                draw_message(frame, "Telegram alerts OFF", DISPLAY_HEIGHT - 20, (0, 255, 255))

            current_time = time.time()

            if crowd_detected:
                below_limit_since = None
                can_send_alert = (
                    not crowd_active
                    and alerts_enabled
                    and current_time - last_alert_time >= args.cooldown
                )
                if can_send_alert:
                    queued = telegram_notifier.send(
                        f"WARNING: Crowd detected. People count = {people_count}"
                    )
                    if queued:
                        crowd_active = True
                        last_alert_time = current_time
            elif crowd_active:
                if below_limit_since is None:
                    below_limit_since = current_time
                elif current_time - below_limit_since >= args.clear_seconds:
                    crowd_active = False
                    below_limit_since = None
                    if alerts_enabled:
                        telegram_notifier.send("Crowd cleared. Back to normal.")

            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        telegram_notifier.stop()
        detector_worker.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
