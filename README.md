# An-Integrated-Al-IoT-Framework-for-Metro-Platform-Safety-and-Crowd-Monitoring
An AI–IoT-based system for real-time crowd monitoring, yellow-line crossing detection and automated alerts to enhance metro platform safety.


## Features

- Real-time person detection using YOLOv8
- Crowd counting from webcam or video file
- Visual bounding boxes with confidence score
- Telegram warning message when crowd limit is crossed
- Alert cooldown to avoid repeated spam messages
- Configurable crowd limit, frame skip, model path, and display size

## Project Structure

```text
crowdmanagement/
  app.py               # Web dashboard entry point (Flask, opens in Chrome)
  main.py               # Desktop entry point (OpenCV window loop)
  render.py             # Shared bounding-box/status drawing logic
  detector.py           # YOLOv8 person detection logic
  detection_worker.py   # Background detection thread for smooth display
  telegram_alert.py      # Telegram message sender
  config.py             # Environment-based configuration
  templates/index.html  # Web dashboard page
  requirements.txt      # Python packages
  yolov8s.pt             # YOLOv8 small model weights (accuracy-tuned default)
  .env                   # Local credentials and settings
  .env.example           # Safe example configuration
```

## Requirements

- Python 3.10 or newer
- Webcam or test video
- Telegram bot token and chat ID for alerts

Install packages:

```powershell
pip install -r requirements.txt
```

## Telegram Setup

1. Open Telegram and search for `BotFather`.
2. Create a bot with `/newbot`.
3. Copy the bot token.
4. Send a message to your bot from your Telegram account.
5. Get your chat ID from:

```text
https://api.telegram.org/botYOUR_BOT_TOKEN/getUpdates
```

Open the `.env` file in the project folder and enter your credentials:

```text
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

You can also change settings like `CROWD_LIMIT`, `DETECTION_CONFIDENCE`, and `ALERT_COOLDOWN` in the same `.env` file.

Test Telegram without opening the camera:

```powershell
python main.py --test-telegram
```

Important: if you already shared or committed a real bot token, revoke it in BotFather and create a new one.

## Run the Project (Web Dashboard)

The default way to run this project is now a web dashboard: the same YOLOv8
detection and Telegram alert logic runs in the background, and the live
annotated feed plus crowd status are viewed in a browser tab instead of a
desktop window.

Start the web server with the default webcam:

```powershell
python app.py
```

Then open Chrome and go to:

```text
http://127.0.0.1:5000
```

The page shows the live detection feed (bounding boxes, people count, and
alert status baked into the video, same as before) plus a status panel that
updates every second.

Other options work the same as before:

```powershell
python app.py --no-alerts
python app.py --crowd-limit 6
python app.py --source "sample_video.mp4"
python app.py --host 0.0.0.0 --port 8000
```

Stop the server with `Ctrl+C` in the terminal.

## Run the Project (Desktop Window, optional)

The original OpenCV desktop window mode is still available if you prefer a
native window instead of a browser tab:

```powershell
python main.py
python main.py --no-alerts
python main.py --crowd-limit 6
python main.py --source "sample_video.mp4"
```

Press `q` to close the video window.

## How It Works

1. OpenCV captures frames from the webcam or video file.
2. Every Nth frame is passed to YOLOv8 for object detection.
3. Only detections with class ID `0`, which means `person`, are counted.
4. If the count is above the crowd limit, the status changes to `CROWD ALERT`.
5. A Telegram message is sent after the cooldown time has passed.

## Suggested Major Project Enhancements

- Save alert images with timestamp
- Add a dashboard for crowd history
- Store alerts in SQLite or MySQL
- Support multiple CCTV camera streams
- Add email or SMS alert options
- Deploy the system on a Raspberry Pi or edge device
