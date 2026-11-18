import queue
import threading


class DetectionWorker:
    def __init__(self):
        self._input_queue = queue.Queue(maxsize=1)
        self._latest_result = (0, [])
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._ready = False
        self._error = None
        self._started = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        if self._started:
            return
        self._started = True
        self._thread.start()

    @property
    def ready(self):
        return self._ready

    @property
    def error(self):
        return self._error

    def submit(self, frame):
        if self._error or self._stop_event.is_set():
            return

        if self._input_queue.full():
            try:
                self._input_queue.get_nowait()
            except queue.Empty:
                pass

        try:
            self._input_queue.put_nowait(frame.copy())
        except queue.Full:
            pass

    def latest_result(self):
        with self._lock:
            people_count, boxes = self._latest_result
            return people_count, list(boxes)

    def stop(self):
        self._stop_event.set()
        if self._started:
            self._thread.join(timeout=2)

    def _run(self):
        try:
            from detector import CrowdDetector

            detector = CrowdDetector()
            self._ready = True

            while not self._stop_event.is_set():
                try:
                    frame = self._input_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                people_count, boxes = detector.detect_people(frame)
                with self._lock:
                    self._latest_result = (people_count, boxes)
        except Exception as exc:
            self._error = exc
