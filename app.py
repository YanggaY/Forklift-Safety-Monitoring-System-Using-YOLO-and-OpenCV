import cv2 as cv
import numpy as np
import json
import os
import time
import threading
import sys
from datetime import datetime
from pathlib import Path
from collections import deque
from ultralytics import YOLO

# ============================================================
# CCTV LIVE WORKER SAFETY MONITORING
# - UI style kept close to the previous version
# - RTSP live mode: always shows latest frame, drops old frames
# - YOLO detection runs separately so CCTV time does not slow down
# - Auto reconnect for unstable RTSP streams
# ============================================================

# RTSP stream address or recorded video file path.
# You can also run: python cctv_live_worker_safety_file_ok.py your_video.mp4
VIDEO_PATH = "ff.mp4"

DANGER_ZONE_PATH = "danger_zones.json"
IGNORE_ZONE_PATH = "ignore_zones.json"

# YOLO model settings
MODEL_NAME = "yolov8m.pt"
TARGET_CLASSES = ["person", "car", "truck", "bus"]
YOLO_CONF = 0.10
YOLO_IOU = 0.45
YOLO_IMG_SIZE = 960

# CCTV mode settings
DETECTION_INTERVAL_SECONDS = 0.20   # 0.20 = about 5 detections per second max
RECONNECT_DELAY_SECONDS = 2.0
FRAME_STALE_SECONDS = 3.0           # show disconnected/stale warning after this
USE_FFMPEG_OPTIONS = True

# Recording mode
# "analysis" = record video with boxes/zones/time overlay
# "clean"    = record original frame with time overlay only
RECORD_MODE = "analysis"
RECORD_SEGMENT_SECONDS = 300
RECORD_DIR = Path("Recordings")
CAPTURE_DIR = Path("Captures")

# Drawing thickness settings
BOX_THICKNESS = 2
ZONE_THICKNESS = 2
POINT_RADIUS = 3

# Warning settings: frame-count based
# Danger-zone warning keeps its existing short confirmation behavior.
WARNING_CONFIRM_SECONDS = 1.0
# Forklift/vehicle-box person warning: confirm only after about 3 seconds worth
# of YOLO detection frames. With DETECTION_INTERVAL_SECONDS=0.20, this is 15 detections.
DRIVER_CONFIRM_SECONDS = 3.0
DRIVER_CONFIRM_FRAMES = max(1, int(round(DRIVER_CONFIRM_SECONDS / max(0.001, DETECTION_INTERVAL_SECONDS))))

DEFAULT_DANGER_ZONES = [[
    (324, 355), (1179, 376), (1276, 504),
    (1276, 716), (10, 716), (11, 468)
]]

DEFAULT_IGNORE_ZONES = [[
    (0, 560), (620, 560), (620, 750), (0, 750)
]]

# Display window settings
DISPLAY_WINDOW_FRACTION = 0.5
MIN_ZOOM = 1.0
MAX_ZOOM = 8.0
ZOOM_STEP = 1.25

# Global UI / runtime state
running = True
paused_view = False        # freezes displayed view only; RTSP reader still keeps latest frame
record_video = False

zoom_scale = 1.0
zoom_offset_x = 0
zoom_offset_y = 0
is_panning = False
pan_start = (0, 0)
pan_start_offset = (0, 0)
PAN_BUTTONS = (cv.EVENT_LBUTTONDOWN, cv.EVENT_RBUTTONDOWN, cv.EVENT_MBUTTONDOWN)
PAN_RELEASE_BUTTONS = (cv.EVENT_LBUTTONUP, cv.EVENT_RBUTTONUP, cv.EVENT_MBUTTONUP)

display_width = None
display_height = None
RENDER_SCALE = 1.0

# Zone storage
danger_zones = []
ignore_zones = []

# Latest frame/detection storage
last_raw_frame = None
last_display = None
frozen_raw_frame = None
frozen_display = None
last_detections = []
last_detection_time = 0.0
last_frame_time = 0.0
last_frame_seq = 0
last_detection_seq = -1

# Warning state timers / counters
forklift_inside_since = None
danger_zone_person_since = None
forklift_inside_person_count = 0
forklift_inside_person_warning = False
danger_zone_person_warning = False

# Status shown in video/controller
last_status_text = "Status: Safe"
last_status_color = (0, 255, 0)
last_person_count = 0
last_forklift_count = 0
last_danger_progress = 0.0
last_forklift_progress = 0.0

# Recording variables
video_writer = None
output_path = None
record_start_time = None
last_saved_file = ""
last_capture_file = ""

# Performance counters
ui_fps = 0.0
detection_fps = 0.0
stream_fps = 0.0
last_ui_tick = time.time()
ui_frame_counter = 0
last_det_counter_tick = time.time()
det_counter = 0

state_lock = threading.Lock()


def is_live_source(src):
    """Return True for RTSP/HTTP live CCTV streams, False for local recorded video files."""
    src_text = str(src).lower().strip()
    return src_text.startswith(("rtsp://", "rtmp://", "http://", "https://"))


def source_label(src):
    return "LIVE" if is_live_source(src) else "FILE"

# ------------------------------------------------------------
# RTSP latest-frame reader thread
# ------------------------------------------------------------
class LatestFrameReader:
    """Keeps only the newest RTSP frame to avoid CCTV delay.

    OpenCV VideoCapture normally buffers frames. If YOLO is slow, a normal read loop
    can show old frames. This class reads continuously in a background thread and
    overwrites the previous frame, so the UI always receives the latest available frame.
    """

    def __init__(self, src):
        self.src = src
        self.is_live = is_live_source(src)
        self.cap = None
        self.lock = threading.Lock()
        self.frame = None
        self.timestamp = 0.0
        self.seq = 0
        self.connected = False
        self.eof = False
        self.last_error = "Not connected"
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.fps_window = deque(maxlen=60)
        self.measured_fps = 0.0
        self.file_fps = 30.0

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self._release_cap()

    def _release_cap(self):
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        self.cap = None
        self.connected = False

    def _open_capture(self):
        if self.is_live and USE_FFMPEG_OPTIONS:
            # TCP is usually more stable than UDP for RTSP. stimeout prevents long hangs.
            os.environ.setdefault(
                "OPENCV_FFMPEG_CAPTURE_OPTIONS",
                "rtsp_transport;tcp|stimeout;5000000|max_delay;500000"
            )

        # RTSP/live streams usually work better with FFMPEG.
        # Local files should be opened normally first because some Windows paths/codecs
        # can fail when forced through a specific backend.
        if self.is_live:
            cap = cv.VideoCapture(self.src, cv.CAP_FFMPEG)
            if not cap.isOpened():
                cap.release()
                cap = cv.VideoCapture(self.src)
        else:
            cap = cv.VideoCapture(self.src)
            if not cap.isOpened():
                cap.release()
                cap = cv.VideoCapture(self.src, cv.CAP_FFMPEG)

        if self.is_live:
            try:
                cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

        if cap.isOpened():
            fps = cap.get(cv.CAP_PROP_FPS)
            if fps is not None and fps > 1:
                self.file_fps = float(fps)
            self.cap = cap
            self.connected = True
            self.eof = False
            self.last_error = ""
            return True

        self.last_error = "Cannot open source"
        self._release_cap()
        return False

    def _loop(self):
        while not self.stop_event.is_set():
            if self.cap is None or not self.connected:
                if not self._open_capture():
                    time.sleep(RECONNECT_DELAY_SECONDS)
                    continue

            ret, frame = self.cap.read()
            now = time.time()

            if not ret or frame is None:
                if self.is_live:
                    self.last_error = "Read failed / reconnecting"
                    self._release_cap()
                    time.sleep(RECONNECT_DELAY_SECONDS)
                    continue
                else:
                    # Recorded video reached the end. Do not reconnect forever.
                    self.connected = False
                    self.eof = True
                    self.last_error = "End of video"
                    time.sleep(0.05)
                    continue

            with self.lock:
                self.frame = frame
                self.timestamp = now
                self.seq += 1
                self.connected = True
                self.eof = False
                self.fps_window.append(now)
                if len(self.fps_window) >= 2:
                    elapsed = self.fps_window[-1] - self.fps_window[0]
                    if elapsed > 0:
                        self.measured_fps = (len(self.fps_window) - 1) / elapsed

            # For recorded video files, read at the original video speed.
            # For RTSP, do not sleep; keep only the newest CCTV frame.
            if not self.is_live:
                time.sleep(max(0.001, 1.0 / max(1.0, self.file_fps)))

    def get_latest(self):
        with self.lock:
            if self.frame is None:
                return None, 0.0, 0, self.connected, self.last_error, self.measured_fps
            return self.frame.copy(), self.timestamp, self.seq, self.connected, self.last_error, self.measured_fps


# ------------------------------------------------------------
# General helpers
# ------------------------------------------------------------
def get_screen_size():
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        root.destroy()
        return int(screen_w), int(screen_h)
    except Exception:
        return 1920, 1080


def calculate_display_size(img_width, img_height, fraction=DISPLAY_WINDOW_FRACTION):
    screen_w, screen_h = get_screen_size()
    max_w = max(320, int(screen_w * fraction))
    max_h = max(240, int(screen_h * fraction))
    fit_scale = min(max_w / max(1, img_width), max_h / max(1, img_height))
    fit_scale = min(1.0, fit_scale)
    return max(1, int(img_width * fit_scale)), max(1, int(img_height * fit_scale))


def scaled_value(value):
    return max(1, int(round(value * RENDER_SCALE)))


def draw_text(img, text, pos, scale=0.7, color=(255, 255, 255), thickness=2, auto_scale=True):
    font = cv.FONT_HERSHEY_SIMPLEX
    if auto_scale:
        scale = scale * RENDER_SCALE
        thickness = scaled_value(thickness)
    else:
        thickness = max(1, int(thickness))
    text_thickness = max(1, int(thickness))
    outline_thickness = max(text_thickness + 3, int(round(text_thickness * 2.4)))
    cv.putText(img, text, pos, font, scale, (0, 0, 0), outline_thickness, cv.LINE_AA)
    cv.putText(img, text, pos, font, scale, color, text_thickness, cv.LINE_AA)


def draw_ui_text(img, text, pos, scale=0.7, color=(255, 255, 255), thickness=2):
    draw_text(img, text, pos, scale, color, thickness, auto_scale=False)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_zones(path, default_zones):
    if not os.path.exists(path):
        save_json(path, default_zones)
        return default_zones.copy()
    try:
        data = json.load(open(path, encoding="utf-8"))
        zones = [[tuple(p) for p in zone] for zone in data if len(zone) >= 3]
        return zones if zones else default_zones.copy()
    except Exception:
        save_json(path, default_zones)
        return default_zones.copy()


def point_in_polygon(point, polygon):
    return cv.pointPolygonTest(np.array(polygon, np.int32), point, False) >= 0


def is_in_any_zone(point, zones):
    return any(point_in_polygon(point, zone) for zone in zones)


def draw_polygon(img, points, color, label=None):
    if not points:
        return
    pts = np.array(points, np.int32)
    for p in points:
        cv.circle(img, p, scaled_value(POINT_RADIUS), color, -1)
    cv.polylines(img, [pts], True, color, scaled_value(ZONE_THICKNESS))
    if label:
        x, y = points[0]
        draw_text(img, label, (x, y - 10), 0.65, color, 2)


# ------------------------------------------------------------
# Zoom / pan helpers
# ------------------------------------------------------------
def clamp_zoom_offset(img_width, img_height, scale, offset_x, offset_y):
    view_w = max(1, int(img_width / scale))
    view_h = max(1, int(img_height / scale))
    max_x = max(0, img_width - view_w)
    max_y = max(0, img_height - view_h)
    offset_x = max(0, min(int(offset_x), max_x))
    offset_y = max(0, min(int(offset_y), max_y))
    return offset_x, offset_y


def get_zoomed_image(img, scale=None, offset_x=None, offset_y=None, output_size=None):
    if img is None:
        return None
    if scale is None:
        scale = zoom_scale
    if offset_x is None:
        offset_x = zoom_offset_x
    if offset_y is None:
        offset_y = zoom_offset_y
    h, w = img.shape[:2]
    if scale <= 1.0:
        shown = img.copy()
    else:
        view_w = max(1, int(w / scale))
        view_h = max(1, int(h / scale))
        offset_x, offset_y = clamp_zoom_offset(w, h, scale, offset_x, offset_y)
        shown = img[offset_y:offset_y + view_h, offset_x:offset_x + view_w]
    if output_size is not None:
        out_w, out_h = output_size
        if shown.shape[1] != out_w or shown.shape[0] != out_h:
            shown = cv.resize(shown, (out_w, out_h), interpolation=cv.INTER_AREA)
    return shown


def screen_to_image_point(x, y, img_width, img_height, scale=None, offset_x=None, offset_y=None, display_size=None):
    if scale is None:
        scale = zoom_scale
    if offset_x is None:
        offset_x = zoom_offset_x
    if offset_y is None:
        offset_y = zoom_offset_y
    if display_size is None:
        display_w, display_h = img_width, img_height
    else:
        display_w, display_h = display_size
    display_w = max(1, display_w)
    display_h = max(1, display_h)
    if scale <= 1.0:
        img_x = int(x * img_width / display_w)
        img_y = int(y * img_height / display_h)
    else:
        view_w = img_width / scale
        view_h = img_height / scale
        img_x = int(offset_x + x * view_w / display_w)
        img_y = int(offset_y + y * view_h / display_h)
    img_x = max(0, min(img_width - 1, img_x))
    img_y = max(0, min(img_height - 1, img_y))
    return img_x, img_y


def change_zoom(mouse_x, mouse_y, wheel_up, img_width, img_height, display_size=None):
    global zoom_scale, zoom_offset_x, zoom_offset_y
    old_scale = zoom_scale
    new_scale = old_scale * ZOOM_STEP if wheel_up else old_scale / ZOOM_STEP
    new_scale = max(MIN_ZOOM, min(MAX_ZOOM, new_scale))
    if abs(new_scale - old_scale) < 1e-6:
        return
    if display_size is None:
        display_w, display_h = img_width, img_height
    else:
        display_w, display_h = display_size
    focus_x, focus_y = screen_to_image_point(
        mouse_x, mouse_y, img_width, img_height,
        old_scale, zoom_offset_x, zoom_offset_y, display_size
    )
    new_view_w = img_width / new_scale
    new_view_h = img_height / new_scale
    new_offset_x = focus_x - mouse_x * new_view_w / max(1, display_w)
    new_offset_y = focus_y - mouse_y * new_view_h / max(1, display_h)
    zoom_scale = new_scale
    zoom_offset_x, zoom_offset_y = clamp_zoom_offset(img_width, img_height, zoom_scale, new_offset_x, new_offset_y)


def zoom_center(wheel_up, img_width, img_height, display_size=None):
    if display_size is None:
        mouse_x = img_width // 2
        mouse_y = img_height // 2
    else:
        mouse_x = display_size[0] // 2
        mouse_y = display_size[1] // 2
    change_zoom(mouse_x, mouse_y, wheel_up, img_width, img_height, display_size)


def reset_zoom():
    global zoom_scale, zoom_offset_x, zoom_offset_y, is_panning
    zoom_scale = 1.0
    zoom_offset_x = 0
    zoom_offset_y = 0
    is_panning = False


def get_wheel_delta(flags):
    try:
        return cv.getMouseWheelDelta(flags)
    except Exception:
        return flags


def apply_pan(mouse_x, mouse_y, img_width, img_height, shown_w, shown_h):
    global zoom_offset_x, zoom_offset_y
    if zoom_scale <= 1.0:
        return
    view_w = img_width / zoom_scale
    view_h = img_height / zoom_scale
    dx = int((pan_start[0] - mouse_x) * view_w / max(1, shown_w))
    dy = int((pan_start[1] - mouse_y) * view_h / max(1, shown_h))
    zoom_offset_x, zoom_offset_y = clamp_zoom_offset(
        img_width, img_height, zoom_scale,
        pan_start_offset[0] + dx,
        pan_start_offset[1] + dy
    )


def worker_mouse_callback(event, x, y, flags, param):
    global is_panning, pan_start, pan_start_offset
    img_width, img_height, shown_w, shown_h = param
    display_size = (shown_w, shown_h)
    if event == cv.EVENT_MOUSEWHEEL:
        delta = get_wheel_delta(flags)
        if delta != 0:
            change_zoom(x, y, delta > 0, img_width, img_height, display_size)
        return
    if event in PAN_BUTTONS:
        if zoom_scale > 1.0:
            is_panning = True
            pan_start = (x, y)
            pan_start_offset = (zoom_offset_x, zoom_offset_y)
        return
    if event in PAN_RELEASE_BUTTONS:
        is_panning = False
        return
    if event == cv.EVENT_MOUSEMOVE and is_panning:
        apply_pan(x, y, img_width, img_height, shown_w, shown_h)
        return
    if event == cv.EVENT_MOUSEMOVE and zoom_scale > 1.0:
        dragging = bool(flags & (cv.EVENT_FLAG_LBUTTON | cv.EVENT_FLAG_RBUTTON | cv.EVENT_FLAG_MBUTTON))
        if dragging and not is_panning:
            is_panning = True
            pan_start = (x, y)
            pan_start_offset = (zoom_offset_x, zoom_offset_y)
        if dragging:
            apply_pan(x, y, img_width, img_height, shown_w, shown_h)
        else:
            is_panning = False


# ------------------------------------------------------------
# Detection / rendering
# ------------------------------------------------------------
def enhance_frame(frame):
    lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    l = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv.cvtColor(cv.merge((l, a, b)), cv.COLOR_LAB2BGR)


def box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) // 2, (y1 + y2) // 2


def box_foot_point(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, y2)


def point_in_box(point, box):
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def box_overlap_ratio(inner_box, outer_box):
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box
    inter_x1 = max(ix1, ox1)
    inter_y1 = max(iy1, oy1)
    inter_x2 = min(ix2, ox2)
    inter_y2 = min(iy2, oy2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    inner_area = max(1, (ix2 - ix1) * (iy2 - iy1))
    return inter_area / inner_area


def is_probable_driver(person_det, vehicle_boxes):
    person_box = person_det["box"]
    person_center = person_det["center"]
    for vehicle_box in vehicle_boxes:
        vx1, vy1, vx2, vy2 = vehicle_box
        px1, py1, px2, py2 = person_box
        vehicle_area = max(1, (vx2 - vx1) * (vy2 - vy1))
        person_area = max(1, (px2 - px1) * (py2 - py1))
        center_inside = point_in_box(person_center, vehicle_box)
        mostly_inside = box_overlap_ratio(person_box, vehicle_box) >= 0.60
        smaller_than_vehicle = person_area < vehicle_area * 0.75
        if center_inside and mostly_inside and smaller_than_vehicle:
            return True
    return False


def person_touches_danger_zone(det):
    if det.get("class") != "person":
        return False
    box = det["box"]
    x1, y1, x2, y2 = box
    check_points = [
        det["center"],
        box_foot_point(box),
        ((x1 + x2) // 2, int(y1 + (y2 - y1) * 0.75)),
    ]
    return any(is_in_any_zone(pt, danger_zones) for pt in check_points)


def run_detection(model, frame):
    results = model(
        enhance_frame(frame),
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        imgsz=YOLO_IMG_SIZE,
        verbose=False
    )
    detections = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            conf = float(box.conf[0])
            if class_name not in TARGET_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_xyxy = (x1, y1, x2, y2)
            center = box_center(box_xyxy)
            if is_in_any_zone(center, ignore_zones):
                continue
            detections.append({
                "class": class_name,
                "conf": conf,
                "box": box_xyxy,
                "center": center,
                "inside_forklift": False,
            })

    vehicle_boxes = [det["box"] for det in detections if det["class"] in ("car", "truck", "bus")]
    for det in detections:
        if det["class"] == "person" and is_probable_driver(det, vehicle_boxes):
            det["inside_forklift"] = True
    return detections


def update_warning_timers(has_person_in_danger_zone, has_inside_forklift_person):
    global danger_zone_person_since, forklift_inside_since, forklift_inside_person_count
    global danger_zone_person_warning, forklift_inside_person_warning
    now = time.time()

    # Danger-zone warning keeps the existing short time confirmation.
    if has_person_in_danger_zone:
        if danger_zone_person_since is None:
            danger_zone_person_since = now
        danger_zone_person_warning = (now - danger_zone_person_since) >= WARNING_CONFIRM_SECONDS
    else:
        danger_zone_person_since = None
        danger_zone_person_warning = False

    # Person-inside-forklift/vehicle warning is confirmed by YOLO detection-frame count.
    # It must be detected continuously for DRIVER_CONFIRM_FRAMES detections.
    if has_inside_forklift_person:
        if forklift_inside_since is None:
            forklift_inside_since = now
        forklift_inside_person_count += 1
        forklift_inside_person_warning = forklift_inside_person_count >= DRIVER_CONFIRM_FRAMES
    else:
        forklift_inside_since = None
        forklift_inside_person_count = 0
        forklift_inside_person_warning = False


def draw_detection(img, det, in_danger):
    x1, y1, x2, y2 = det["box"]
    class_name = det["class"]
    center = det["center"]
    if class_name == "person":
        color = (0, 0, 255) if in_danger else (255, 0, 255)
        label = "PERSON IN DANGER ZONE" if in_danger else "person"
        if det.get("inside_forklift") and not in_danger:
            label = "driver/person"
    else:
        color = (0, 255, 255)
        label = "forklift"
    cv.rectangle(img, (x1, y1), (x2, y2), color, scaled_value(BOX_THICKNESS))
    cv.circle(img, center, scaled_value(POINT_RADIUS), color, -1)
    draw_text(img, label, (x1, y1 - 10), 0.6, color, 2)


def render_frame(frame, detections, update_warning_state=True):
    global last_status_text, last_status_color
    global last_person_count, last_forklift_count, last_danger_progress, last_forklift_progress
    display = frame.copy()
    person_count = 0
    forklift_count = 0
    has_person_in_danger_zone = False
    has_inside_forklift_person = False

    for i, zone in enumerate(ignore_zones):
        draw_polygon(display, zone, (120, 120, 120), f"Ignore Zone {i + 1}")

    for i, zone in enumerate(danger_zones):
        draw_polygon(display, zone, (0, 255, 0), f"Danger Zone {i + 1}")

    # Determine warning inputs first.
    # IMPORTANT:
    # - YOLO detection may run at 5 FPS.
    # - The UI may draw at 30+ FPS using the newest raw frame.
    # - Therefore warning counters must update only when a NEW detection result arrives.
    #   Otherwise the forklift/person 3-second confirmation would accidentally count UI frames.
    for det in detections:
        if det["class"] == "person":
            if det.get("inside_forklift"):
                has_inside_forklift_person = True
            elif person_touches_danger_zone(det):
                has_person_in_danger_zone = True

    if update_warning_state:
        update_warning_timers(has_person_in_danger_zone, has_inside_forklift_person)

    for det in detections:
        class_name = det["class"]
        if class_name == "person":
            person_count += 1
            if det.get("inside_forklift"):
                in_danger = forklift_inside_person_warning
            else:
                in_danger = person_touches_danger_zone(det) and danger_zone_person_warning
        else:
            forklift_count += 1
            in_danger = False
        draw_detection(display, det, in_danger)

    warning = danger_zone_person_warning or forklift_inside_person_warning
    if warning:
        for i, zone in enumerate(danger_zones):
            draw_polygon(display, zone, (0, 0, 255), f"Danger Zone {i + 1}")

    if danger_zone_person_warning:
        status = "WARNING: Person in danger zone"
    elif forklift_inside_person_warning:
        status = "WARNING: Person inside vehicle box"
    else:
        status = "Status: Safe"
    status_color = (0, 0, 255) if warning else (0, 255, 0)

    now = time.time()
    danger_progress = 0.0 if danger_zone_person_since is None else min(WARNING_CONFIRM_SECONDS, now - danger_zone_person_since)
    forklift_progress = min(DRIVER_CONFIRM_FRAMES, forklift_inside_person_count)

    last_status_text = status
    last_status_color = status_color
    last_person_count = person_count
    last_forklift_count = forklift_count
    last_danger_progress = danger_progress
    last_forklift_progress = forklift_progress
    return display


def draw_video_status_overlay(img, show_record_dot=True, live_state="LIVE", stale_seconds=0.0):
    if img is None:
        return img
    status_short = last_status_text.replace("Status: ", "STATUS: ").upper()
    draw_ui_text(img, status_short, (25, 45), 0.72, last_status_color, 2)
    draw_ui_text(img, f"Persons: {last_person_count}", (25, 78), 0.62, (255, 255, 255), 2)
    draw_ui_text(img, f"Forklifts: {last_forklift_count}", (25, 111), 0.62, (255, 255, 255), 2)

    if stale_seconds > FRAME_STALE_SECONDS:
        draw_ui_text(img, "STREAM STALE / RECONNECTING", (25, 145), 0.62, (0, 0, 255), 2)

    now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text_size, _ = cv.getTextSize(now_text, cv.FONT_HERSHEY_SIMPLEX, 0.62, 2)
    x = max(20, img.shape[1] - text_size[0] - 25)

    if show_record_dot and record_video:
        # slower, larger blink than original
        blink_on = int(time.time() * 1.25) % 2 == 0
        if blink_on:
            dot_x = max(16, x - 34)
            cv.circle(img, (dot_x, 39), 14, (0, 0, 0), -1)
            cv.circle(img, (dot_x, 39), 11, (0, 0, 255), -1)

    live_color = (0, 255, 0) if live_state == "LIVE" else (0, 255, 255)
    draw_ui_text(img, live_state, (max(20, x - 95), 45), 0.62, live_color, 2)
    draw_ui_text(img, now_text, (x, 45), 0.62, (255, 255, 255), 2)
    return img


# ------------------------------------------------------------
# Recording / capture
# ------------------------------------------------------------
def make_output_path():
    day_folder = RECORD_DIR / datetime.now().strftime("%Y-%m-%d")
    day_folder.mkdir(parents=True, exist_ok=True)
    return str(day_folder / (datetime.now().strftime("%H%M_%S") + ".mkv"))


def open_record_writer(width, height, fps):
    path = make_output_path()
    fps = max(1.0, min(float(fps or 15.0), 60.0))
    writer = cv.VideoWriter(path, cv.VideoWriter_fourcc(*"XVID"), fps, (width, height))
    if not writer.isOpened():
        print("Warning: MKV writer did not open. Check OpenCV/FFmpeg codec support.")
    return writer, path


def rotate_recording_segment(width, height, fps):
    global video_writer, output_path, last_saved_file, record_start_time
    if video_writer is not None:
        video_writer.release()
        if output_path:
            last_saved_file = os.path.basename(output_path)
        print("Saved:", output_path)
    video_writer, output_path = open_record_writer(width, height, fps)
    record_start_time = time.time()
    print("Recording started:", output_path)


def check_recording_rotation(width, height, fps):
    if not record_video or video_writer is None or record_start_time is None:
        return
    if time.time() - record_start_time >= RECORD_SEGMENT_SECONDS:
        rotate_recording_segment(width, height, fps)


def format_record_segment_time(seconds):
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds} sec"
    if seconds < 3600:
        return f"{seconds // 60} min"
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    return f"{hours} hr" if minutes == 0 else f"{hours} hr {minutes} min"


def change_record_segment(step):
    global RECORD_SEGMENT_SECONDS
    current = int(RECORD_SEGMENT_SECONDS)
    if current < 60:
        delta = 10
    elif current < 600:
        delta = 60
    else:
        delta = 300
    current += delta * step
    current = max(10, min(current, 4 * 3600))
    RECORD_SEGMENT_SECONDS = current
    print("Recording segment length:", format_record_segment_time(RECORD_SEGMENT_SECONDS))


def start_recording(width, height, fps):
    rotate_recording_segment(width, height, fps)


def stop_recording():
    global video_writer, record_start_time, last_saved_file
    if video_writer is not None:
        video_writer.release()
        if output_path:
            last_saved_file = os.path.basename(output_path)
        video_writer = None
        record_start_time = None
        print("Saved:", output_path)


def toggle_recording(width, height, fps):
    global record_video
    record_video = not record_video
    if record_video:
        start_recording(width, height, fps)
    else:
        stop_recording()


def make_capture_path():
    day_folder = CAPTURE_DIR / datetime.now().strftime("%Y-%m-%d")
    day_folder.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    base_name = now.strftime("%H%M_%S_%f")[:-3]
    path = day_folder / f"{base_name}.jpg"
    index = 2
    while path.exists():
        path = day_folder / f"{base_name}_{index}.jpg"
        index += 1
    return path


def capture_current_frame():
    global last_capture_file
    img = frozen_display if paused_view and frozen_display is not None else last_display
    if img is None:
        return
    capture_img = get_zoomed_image(img, output_size=(display_width, display_height))
    capture_img = draw_video_status_overlay(capture_img, show_record_dot=False, live_state="PAUSED" if paused_view else "LIVE")
    path = make_capture_path()
    ok = cv.imwrite(str(path), capture_img)
    if ok:
        last_capture_file = path.name
        print("Captured:", path)
    else:
        print("Capture failed:", path)


# ------------------------------------------------------------
# Zone selection / management
# ------------------------------------------------------------
def save_all_zones():
    save_json(DANGER_ZONE_PATH, danger_zones)
    save_json(IGNORE_ZONE_PATH, ignore_zones)


def select_polygon(frame, title):
    points = []
    minimum_points_flash_until = 0.0
    local_zoom_scale = 1.0
    local_offset_x = 0
    local_offset_y = 0
    local_panning = False
    local_pan_start = (0, 0)
    local_pan_start_offset = (0, 0)
    h, w = frame.shape[:2]
    local_display_size = calculate_display_size(w, h)

    def local_clamp(scale, offset_x, offset_y):
        return clamp_zoom_offset(w, h, scale, offset_x, offset_y)

    def local_screen_to_image(x, y):
        return screen_to_image_point(x, y, w, h, local_zoom_scale, local_offset_x, local_offset_y, local_display_size)

    def mouse_callback(event, x, y, flags, param):
        nonlocal local_zoom_scale, local_offset_x, local_offset_y
        nonlocal local_panning, local_pan_start, local_pan_start_offset
        if event == cv.EVENT_MOUSEWHEEL:
            delta = get_wheel_delta(flags)
            old_scale = local_zoom_scale
            new_scale = old_scale * ZOOM_STEP if delta > 0 else old_scale / ZOOM_STEP
            new_scale = max(MIN_ZOOM, min(MAX_ZOOM, new_scale))
            focus_x, focus_y = screen_to_image_point(x, y, w, h, old_scale, local_offset_x, local_offset_y, local_display_size)
            display_w, display_h = local_display_size
            local_zoom_scale = new_scale
            new_view_w = w / local_zoom_scale
            new_view_h = h / local_zoom_scale
            local_offset_x = focus_x - x * new_view_w / max(1, display_w)
            local_offset_y = focus_y - y * new_view_h / max(1, display_h)
            local_offset_x, local_offset_y = local_clamp(local_zoom_scale, local_offset_x, local_offset_y)
            return
        if event in (cv.EVENT_RBUTTONDOWN, cv.EVENT_MBUTTONDOWN):
            local_panning = True
            local_pan_start = (x, y)
            local_pan_start_offset = (local_offset_x, local_offset_y)
            return
        if event in (cv.EVENT_RBUTTONUP, cv.EVENT_MBUTTONUP):
            local_panning = False
            return
        if event == cv.EVENT_MOUSEMOVE and local_panning and local_zoom_scale > 1.0:
            display_w, display_h = local_display_size
            view_w = w / local_zoom_scale
            view_h = h / local_zoom_scale
            dx = int((local_pan_start[0] - x) * view_w / max(1, display_w))
            dy = int((local_pan_start[1] - y) * view_h / max(1, display_h))
            local_offset_x, local_offset_y = local_clamp(
                local_zoom_scale, local_pan_start_offset[0] + dx, local_pan_start_offset[1] + dy
            )
            return
        if event == cv.EVENT_LBUTTONDOWN:
            img_x, img_y = local_screen_to_image(x, y)
            points.append((img_x, img_y))
            print(f"Point {len(points)}:", (img_x, img_y))

    cv.namedWindow(title, cv.WINDOW_NORMAL)
    cv.resizeWindow(title, local_display_size[0], local_display_size[1])
    cv.setMouseCallback(title, mouse_callback)

    while True:
        display = frame.copy()
        draw_polygon(display, points, (0, 0, 255))
        display = get_zoomed_image(display, local_zoom_scale, local_offset_x, local_offset_y, local_display_size)
        now = time.time()
        if now < minimum_points_flash_until:
            blink_on = int(now * 8) % 2 == 0
            guide_color = (0, 0, 255) if blink_on else (0, 255, 255)
            guide_scale = 0.70 if blink_on else 0.62
            guide_thickness = 3
        else:
            guide_color = (255, 255, 255)
            guide_scale = 0.55
            guide_thickness = 2
        draw_ui_text(display, "Click points - minimum 3", (20, 35), guide_scale, guide_color, guide_thickness)
        draw_ui_text(display, "[ENTER] Save selected zone", (20, 65), 0.55, (255, 255, 255), 2)
        draw_ui_text(display, "[R/r] Reset selected points", (20, 95), 0.55, (255, 255, 255), 2)
        draw_ui_text(display, "[ESC] Cancel", (20, 125), 0.55, (255, 255, 255), 2)
        draw_ui_text(display, f"Current Points: {len(points)}", (20, 155), 0.55, (255, 255, 255), 2)
        cv.imshow(title, display)
        key = cv.waitKeyEx(1)
        if key == 13:
            if len(points) >= 3:
                cv.destroyWindow(title)
                return points
            minimum_points_flash_until = time.time() + 1.0
            print("Select at least 3 points.")
        elif key in (ord("r"), ord("R")):
            points.clear()
        elif key == 27:
            cv.destroyWindow(title)
            return None


def add_zone(zone_type):
    global paused_view, frozen_raw_frame, frozen_display
    frame = frozen_raw_frame if paused_view and frozen_raw_frame is not None else last_raw_frame
    if frame is None:
        return
    paused_view = True
    frozen_raw_frame = frame.copy()
    frozen_display = last_display.copy() if last_display is not None else frame.copy()
    if zone_type == "danger":
        zone = select_polygon(frame, "Select Danger Zone")
        if zone:
            danger_zones.append(zone)
    else:
        zone = select_polygon(frame, "Select Ignore Zone")
        if zone:
            ignore_zones.append(zone)
    save_all_zones()
    refresh_display_from_frame(frame)


def in_rect(x, y, rect):
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def draw_button(img, text, rect, color, active=False, scale=0.55):
    x1, y1, x2, y2 = rect
    button_color = (80, 140, 255) if active else color
    cv.rectangle(img, (x1, y1), (x2, y2), button_color, -1)
    cv.rectangle(img, (x1, y1), (x2, y2), (220, 220, 220), 2)
    action = BUTTON_ACTION_LOOKUP.get(text, "")
    draw_panel_icon(img, active=active, action=action, rect=rect)
    size, _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, scale, 2)
    tx = x1 + 62 if action else x1 + (x2 - x1 - size[0]) // 2
    ty = y1 + (y2 - y1 + size[1]) // 2
    draw_ui_text(img, text, (tx, ty), scale)


def zone_manager():
    global danger_zones, ignore_zones
    selected_type = "danger"
    selected_indices = {"danger": 0, "ignore": 0}
    exit_requested = False
    delete_buttons = []
    list_areas = {"danger": (30, 195, 520, 350), "ignore": (30, 405, 520, 560)}
    back_button = (30, 575, 250, 620)
    clear_danger_button = (275, 575, 390, 620)
    clear_ignore_button = (405, 575, 520, 620)

    def current_zones():
        return danger_zones if selected_type == "danger" else ignore_zones

    def mouse_callback(event, x, y, flags, param):
        nonlocal selected_type, exit_requested
        if event != cv.EVENT_LBUTTONDOWN:
            return
        if in_rect(x, y, back_button):
            save_all_zones()
            exit_requested = True
            return
        if in_rect(x, y, clear_danger_button):
            danger_zones.clear()
            selected_indices["danger"] = 0
            save_all_zones()
            return
        if in_rect(x, y, clear_ignore_button):
            ignore_zones.clear()
            selected_indices["ignore"] = 0
            save_all_zones()
            return
        for zone_type, rect in list_areas.items():
            if in_rect(x, y, rect):
                selected_type = zone_type
                zones = current_zones()
                if zones:
                    relative_y = y - rect[1] - 28
                    clicked_index = max(0, relative_y // 34)
                    selected_indices[zone_type] = max(0, min(clicked_index, len(zones) - 1))
                return
        for zone_type, idx, rect in delete_buttons:
            zones = danger_zones if zone_type == "danger" else ignore_zones
            if in_rect(x, y, rect) and 0 <= idx < len(zones):
                zones.pop(idx)
                selected_indices[zone_type] = max(0, min(selected_indices[zone_type], len(zones) - 1))
                save_all_zones()
                return

    cv.namedWindow("Zone Manager")
    cv.setMouseCallback("Zone Manager", mouse_callback)

    while True:
        if exit_requested:
            cv.destroyWindow("Zone Manager")
            frame = frozen_raw_frame if paused_view and frozen_raw_frame is not None else last_raw_frame
            if frame is not None:
                refresh_display_from_frame(frame)
            return
        panel = np.full((650, 550, 3), 35, np.uint8)
        delete_buttons.clear()
        draw_ui_text(panel, "ZONE MANAGER", (30, 40), 0.9)
        draw_ui_text(panel, "Both Danger Zone and Ignore Zone lists are shown together", (30, 75), 0.46, (210, 210, 210), 1)
        draw_ui_text(panel, "Click a list or use [TAB] to choose which list keyboard edits", (30, 100), 0.46, (210, 210, 210), 1)
        draw_ui_text(panel, "[UP/DOWN] Select | [D] Delete selected | [ESC] Back", (30, 125), 0.46, (210, 210, 210), 1)
        draw_ui_text(panel, "[C] Clear selected list", (30, 150), 0.46, (210, 210, 210), 1)

        def draw_zone_list(zone_type, title, zones, color, start_y):
            selected = selected_type == zone_type
            selected_index = selected_indices[zone_type]
            header_color = (0, 255, 255) if selected else color
            draw_ui_text(panel, title, (35, start_y), 0.7, header_color, 2)
            cv.rectangle(panel, (30, start_y + 20), (520, start_y + 175), color, 1)
            if not zones:
                draw_ui_text(panel, "No zones", (50, start_y + 70), 0.6, (180, 180, 180), 2)
                return
            selected_indices[zone_type] = max(0, min(selected_index, len(zones) - 1))
            first_item_y = start_y + 48
            for i, zone in enumerate(zones):
                y = first_item_y + i * 34
                if y > start_y + 165:
                    remaining = len(zones) - i
                    draw_ui_text(panel, f"... {remaining} more zones", (50, y), 0.5, (180, 180, 180), 1)
                    break
                marker = ">" if selected and i == selected_indices[zone_type] else " "
                text_color = (0, 255, 255) if selected and i == selected_indices[zone_type] else (220, 220, 220)
                draw_ui_text(panel, f"{marker} {title} {i + 1} ({len(zone)} points)", (50, y), 0.52, text_color, 2)
                delete_rect = (390, y - 24, 505, y + 4)
                delete_buttons.append((zone_type, i, delete_rect))
                draw_button(panel, "DELETE", delete_rect, (120, 50, 50), scale=0.45)

        draw_zone_list("danger", "Danger Zone", danger_zones, (0, 255, 0), 175)
        draw_zone_list("ignore", "Ignore Zone", ignore_zones, (120, 120, 120), 385)
        draw_button(panel, "BACK TO PLAYER", back_button, (60, 120, 60))
        draw_button(panel, "CLEAR DZ", clear_danger_button, (80, 80, 180), scale=0.45)
        draw_button(panel, "CLEAR IZ", clear_ignore_button, (80, 80, 180), scale=0.45)
        cv.imshow("Zone Manager", panel)
        key = cv.waitKeyEx(1)
        zones = current_zones()
        if key == 27:
            save_all_zones()
            cv.destroyWindow("Zone Manager")
            return
        elif key == 9:
            selected_type = "ignore" if selected_type == "danger" else "danger"
        elif key == 2490368:
            selected_indices[selected_type] = max(0, selected_indices[selected_type] - 1)
        elif key == 2621440:
            selected_indices[selected_type] = min(max(0, len(zones) - 1), selected_indices[selected_type] + 1)
        elif key in (ord("d"), ord("D")) and zones:
            zones.pop(selected_indices[selected_type])
            selected_indices[selected_type] = max(0, min(selected_indices[selected_type], len(zones) - 1))
            save_all_zones()
        elif key in (ord("c"), ord("C")):
            zones.clear()
            selected_indices[selected_type] = 0
            save_all_zones()


# ------------------------------------------------------------
# Controller panel UI
# ------------------------------------------------------------
def draw_panel_icon(img, action, rect, active=False):
    x1, y1, x2, y2 = rect
    cx = x1 + 30
    cy = (y1 + y2) // 2
    icon_color = (255, 255, 255)
    accent = (0, 255, 255) if active else (230, 230, 230)
    red = (0, 0, 255)
    green = (0, 220, 0)

    if action == "pause_view":
        if paused_view:
            pts = np.array([(cx - 8, cy - 13), (cx - 8, cy + 13), (cx + 14, cy)], np.int32)
            cv.fillPoly(img, [pts], green)
        else:
            cv.rectangle(img, (cx - 12, cy - 14), (cx - 4, cy + 14), icon_color, -1)
            cv.rectangle(img, (cx + 4, cy - 14), (cx + 12, cy + 14), icon_color, -1)
    elif action == "add_danger":
        pts = np.array([(cx, cy - 17), (cx - 18, cy + 15), (cx + 18, cy + 15)], np.int32)
        cv.polylines(img, [pts], True, (0, 255, 255), 2)
        draw_ui_text(img, "!", (cx - 4, cy + 9), 0.70, (0, 255, 255), 2)
    elif action == "add_ignore":
        cv.rectangle(img, (cx - 16, cy - 13), (cx + 16, cy + 13), (150, 150, 150), 2)
        cv.line(img, (cx - 18, cy + 16), (cx + 18, cy - 16), (150, 150, 150), 2)
    elif action == "manager":
        cv.circle(img, (cx, cy), 14, accent, 2)
        cv.circle(img, (cx, cy), 5, accent, 2)
        for dx, dy in [(0, -22), (0, 22), (-22, 0), (22, 0)]:
            cv.line(img, (cx, cy), (cx + dx, cy + dy), accent, 2)
    elif action == "reconnect":
        cv.ellipse(img, (cx, cy), (16, 16), 0, 40, 320, icon_color, 3)
        cv.arrowedLine(img, (cx + 10, cy - 13), (cx + 18, cy - 13), icon_color, 2, tipLength=0.5)
    elif action == "record":
        cv.circle(img, (cx, cy), 13, red if record_video else (170, 170, 170), -1)
        cv.circle(img, (cx, cy), 14, icon_color, 2)
    elif action == "capture":
        cv.rectangle(img, (cx - 17, cy - 10), (cx + 17, cy + 13), icon_color, 2)
        cv.rectangle(img, (cx - 8, cy - 16), (cx + 8, cy - 10), icon_color, -1)
        cv.circle(img, (cx, cy + 2), 8, icon_color, 2)
        cv.circle(img, (cx, cy + 2), 3, icon_color, -1)
    elif action == "zoom_in":
        cv.circle(img, (cx - 3, cy - 3), 13, icon_color, 2)
        cv.line(img, (cx + 7, cy + 7), (cx + 20, cy + 20), icon_color, 3)
        cv.line(img, (cx - 10, cy - 3), (cx + 4, cy - 3), icon_color, 2)
        cv.line(img, (cx - 3, cy - 10), (cx - 3, cy + 4), icon_color, 2)
    elif action == "zoom_out":
        cv.circle(img, (cx - 3, cy - 3), 13, icon_color, 2)
        cv.line(img, (cx + 7, cy + 7), (cx + 20, cy + 20), icon_color, 3)
        cv.line(img, (cx - 10, cy - 3), (cx + 4, cy - 3), icon_color, 2)
    elif action == "zoom_reset":
        cv.rectangle(img, (cx - 16, cy - 13), (cx + 16, cy + 13), icon_color, 2)
        cv.line(img, (cx - 9, cy), (cx + 9, cy), icon_color, 2)
        cv.line(img, (cx, cy - 8), (cx, cy + 8), icon_color, 2)
    elif action == "record_time_down":
        cv.polylines(img, [np.array([(cx + 12, cy - 14), (cx - 12, cy), (cx + 12, cy + 14)], np.int32)], False, icon_color, 3)
    elif action == "record_time_up":
        cv.polylines(img, [np.array([(cx - 12, cy - 14), (cx + 12, cy), (cx - 12, cy + 14)], np.int32)], False, icon_color, 3)
    elif action == "quit":
        cv.line(img, (cx - 13, cy - 13), (cx + 13, cy + 13), red, 3)
        cv.line(img, (cx + 13, cy - 13), (cx - 13, cy + 13), red, 3)


def draw_mouse_icon(img, x, y):
    cv.ellipse(img, (x + 13, y + 16), (13, 18), 0, 0, 360, (210, 210, 210), 2)
    cv.line(img, (x + 13, y - 2), (x + 13, y + 8), (210, 210, 210), 2)
    cv.line(img, (x + 13, y + 4), (x + 13, y + 13), (210, 210, 210), 2)
    cv.circle(img, (x + 13, y + 4), 2, (210, 210, 210), -1)


def draw_wheel_icon(img, x, y):
    cv.arrowedLine(img, (x + 12, y + 24), (x + 12, y + 3), (210, 210, 210), 2, tipLength=0.35)
    cv.arrowedLine(img, (x + 12, y + 3), (x + 12, y + 24), (210, 210, 210), 2, tipLength=0.35)


def draw_drag_icon(img, x, y):
    c = (x + 14, y + 14)
    cv.arrowedLine(img, c, (x + 14, y - 2), (210, 210, 210), 2, tipLength=0.35)
    cv.arrowedLine(img, c, (x + 14, y + 30), (210, 210, 210), 2, tipLength=0.35)
    cv.arrowedLine(img, c, (x - 2, y + 14), (210, 210, 210), 2, tipLength=0.35)
    cv.arrowedLine(img, c, (x + 30, y + 14), (210, 210, 210), 2, tipLength=0.35)


BUTTONS = [
    ("[SPACE] PAUSE VIEW", (20, 20, 320, 70), "pause_view"),
    ("[A] ADD DANGER ZONE", (20, 85, 320, 135), "add_danger"),
    ("[I] ADD IGNORE ZONE", (20, 150, 320, 200), "add_ignore"),
    ("[M] ZONE MANAGER", (20, 215, 320, 265), "manager"),
    ("[R] RECONNECT", (20, 280, 320, 330), "reconnect"),
    ("[S] RECORD ON / OFF", (20, 345, 320, 395), "record"),
    ("[C] CAPTURE IMAGE", (20, 410, 320, 460), "capture"),
    ("[ESC] QUIT", (20, 475, 320, 525), "quit"),
    ("[+] ZOOM IN", (355, 20, 635, 70), "zoom_in"),
    ("[-] ZOOM OUT", (355, 85, 635, 135), "zoom_out"),
    ("[0] RESET ZOOM", (355, 150, 635, 200), "zoom_reset"),
    ("[[] RECORD TIME -", (355, 215, 635, 265), "record_time_down"),
    ("]] RECORD TIME +", (355, 280, 635, 330), "record_time_up"),
]
BUTTON_ACTION_LOOKUP = {text: action for text, _, action in BUTTONS}


def draw_control_panel(reader_connected=True, reader_error="", stale_seconds=0.0):
    panel = np.full((725, 640, 3), 30, np.uint8)
    for text, rect, action in BUTTONS:
        active = (action == "pause_view" and paused_view) or (action == "record" and record_video)
        draw_button(panel, text, rect, (60, 100, 180), active)

    draw_wheel_icon(panel, 23, 545)
    draw_ui_text(panel, "Wheel: Zoom in / out", (65, 568), 0.50, (210, 210, 210), 1)
    draw_drag_icon(panel, 23, 585)
    draw_ui_text(panel, "Left/Right/Middle Drag: Pan", (65, 608), 0.50, (210, 210, 210), 1)
    draw_mouse_icon(panel, 370, 350)
    draw_ui_text(panel, "Wheel works on video window", (410, 373), 0.45, (210, 210, 210), 1)
    draw_ui_text(panel, f"Current Zoom: {zoom_scale:.2f}x", (360, 412), 0.55, (0, 255, 255), 1)
    draw_ui_text(panel, "Pan = drag to move after zoom", (360, 447), 0.50, (210, 210, 210), 1)

    live_state = "PAUSED VIEW" if paused_view else ((source_label(VIDEO_PATH)) if reader_connected and stale_seconds <= FRAME_STALE_SECONDS else ("RECONNECTING" if is_live_source(VIDEO_PATH) else "ENDED"))
    record_state = "ON" if record_video else "OFF"
    segment_label = format_record_segment_time(RECORD_SEGMENT_SECONDS)

    cv.rectangle(panel, (350, 475), (630, 705), (45, 45, 45), -1)
    cv.rectangle(panel, (350, 475), (630, 705), (120, 120, 120), 1)
    draw_ui_text(panel, "CCTV STATUS", (365, 505), 0.65, (255, 255, 255), 2)
    draw_ui_text(panel, f"State: {live_state}", (365, 535), 0.48, (0, 255, 0) if live_state == "LIVE" else (0, 255, 255), 1)
    draw_ui_text(panel, f"Stream FPS: {stream_fps:.1f}", (365, 560), 0.48, (220, 220, 220), 1)
    draw_ui_text(panel, f"UI FPS: {ui_fps:.1f}", (365, 585), 0.48, (220, 220, 220), 1)
    draw_ui_text(panel, f"Detect FPS: {detection_fps:.1f}", (365, 610), 0.48, (220, 220, 220), 1)
    draw_ui_text(panel, f"Record: {record_state}", (365, 635), 0.48, (0, 0, 255) if record_video else (220, 220, 220), 1)
    draw_ui_text(panel, f"Segment: {segment_label}", (365, 660), 0.45, (0, 255, 255), 1)
    current_file = os.path.basename(output_path) if output_path else "-"
    saved_file = last_saved_file if last_saved_file else "-"
    capture_file = last_capture_file if last_capture_file else "-"
    draw_ui_text(panel, f"Current: {current_file}", (365, 683), 0.40, (220, 220, 220), 1)
    draw_ui_text(panel, f"Capture: {capture_file}", (365, 704), 0.40, (220, 220, 220), 1)

    if reader_error and not reader_connected:
        draw_ui_text(panel, reader_error[:38], (25, 690), 0.43, (0, 0, 255), 1)

    return panel


# ------------------------------------------------------------
# Actions / keyboard
# ------------------------------------------------------------
def refresh_display_from_frame(frame):
    global last_detections, last_display, frozen_display
    if frame is None:
        return
    det = run_detection(model, frame)
    disp = render_frame(frame, det)
    with state_lock:
        last_detections = det
        last_display = disp
        if paused_view:
            frozen_display = disp.copy()


def handle_action(action, width, height, fps, reader=None):
    global paused_view, running, frozen_raw_frame, frozen_display, last_display, last_raw_frame
    if action == "pause_view":
        paused_view = not paused_view
        if paused_view:
            frozen_raw_frame = last_raw_frame.copy() if last_raw_frame is not None else None
            frozen_display = last_display.copy() if last_display is not None else None
        else:
            frozen_raw_frame = None
            frozen_display = None
    elif action == "add_danger":
        add_zone("danger")
    elif action == "add_ignore":
        add_zone("ignore")
    elif action == "manager":
        paused_view = True
        frozen_raw_frame = last_raw_frame.copy() if last_raw_frame is not None else None
        frozen_display = last_display.copy() if last_display is not None else None
        zone_manager()
    elif action == "reconnect":
        if reader is not None:
            reader._release_cap()
            # For a video file this restarts from the beginning; for RTSP it reconnects.
            reader.eof = False
    elif action == "record":
        toggle_recording(width, height, fps)
    elif action == "capture":
        capture_current_frame()
    elif action == "zoom_in":
        zoom_center(True, width, height, (display_width, display_height))
    elif action == "zoom_out":
        zoom_center(False, width, height, (display_width, display_height))
    elif action == "zoom_reset":
        reset_zoom()
    elif action == "record_time_down":
        change_record_segment(-1)
    elif action == "record_time_up":
        change_record_segment(1)
    elif action == "quit":
        running = False


def panel_click(event, x, y, flags, param):
    width, height, fps, reader = param
    if event == cv.EVENT_MOUSEWHEEL:
        delta = get_wheel_delta(flags)
        if delta != 0:
            zoom_center(delta > 0, width, height, (display_width, display_height))
        return
    if event != cv.EVENT_LBUTTONDOWN:
        return
    for _, rect, action in BUTTONS:
        if in_rect(x, y, rect):
            handle_action(action, width, height, fps, reader)
            break


def handle_keyboard(key, width, height, fps, reader):
    if key == -1:
        return
    if key == 27:
        handle_action("quit", width, height, fps, reader)
    elif key == 32:
        handle_action("pause_view", width, height, fps, reader)
    elif key in (ord("a"), ord("A")):
        handle_action("add_danger", width, height, fps, reader)
    elif key in (ord("i"), ord("I")):
        handle_action("add_ignore", width, height, fps, reader)
    elif key in (ord("m"), ord("M")):
        handle_action("manager", width, height, fps, reader)
    elif key in (ord("r"), ord("R")):
        handle_action("reconnect", width, height, fps, reader)
    elif key in (ord("s"), ord("S")):
        handle_action("record", width, height, fps, reader)
    elif key in (ord("c"), ord("C")):
        handle_action("capture", width, height, fps, reader)
    elif key in (ord("+"), ord("=")):
        handle_action("zoom_in", width, height, fps, reader)
    elif key in (ord("-"), ord("_")):
        handle_action("zoom_out", width, height, fps, reader)
    elif key in (ord("["), ord("{")):
        handle_action("record_time_down", width, height, fps, reader)
    elif key in (ord("]"), ord("}")):
        handle_action("record_time_up", width, height, fps, reader)
    elif key == ord("0"):
        handle_action("zoom_reset", width, height, fps, reader)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    danger_zones = load_zones(DANGER_ZONE_PATH, DEFAULT_DANGER_ZONES)
    ignore_zones = load_zones(IGNORE_ZONE_PATH, DEFAULT_IGNORE_ZONES)

    print("Loading YOLO model:", MODEL_NAME)
    model = YOLO(MODEL_NAME)

    reader = LatestFrameReader(VIDEO_PATH)
    reader.start()

    print("Opening source:", VIDEO_PATH)
    first_frame = None
    first_ts = 0.0
    first_seq = 0
    start_wait = time.time()
    while time.time() - start_wait < 20.0:
        frame, ts, seq, connected, err, measured = reader.get_latest()
        if frame is not None:
            first_frame = frame
            first_ts = ts
            first_seq = seq
            break
        time.sleep(0.05)

    if first_frame is None:
        reader.stop()
        raise SystemExit("Cannot read first frame. Check VIDEO_PATH or pass a video file path as an argument.")

    height, width = first_frame.shape[:2]
    display_width, display_height = calculate_display_size(width, height)
    RENDER_SCALE = max(1.0, min(width / max(1, display_width), height / max(1, display_height)))

    last_raw_frame = first_frame.copy()
    last_frame_time = first_ts
    last_frame_seq = first_seq
    last_detections = run_detection(model, first_frame)
    last_display = render_frame(first_frame, last_detections)
    last_detection_seq = first_seq
    last_detection_time = time.time()

    fps_for_recording = max(5.0, min((reader.file_fps if not reader.is_live else reader.measured_fps) or 15.0, 30.0))

    # Place the controller and player side by side.
    # Control Panel: left / Video Player: immediately to the right.
    PANEL_X = 20
    PANEL_Y = 40
    PANEL_WIDTH = 640
    WINDOW_GAP = 10

    cv.namedWindow("Control Panel", cv.WINDOW_NORMAL)
    cv.resizeWindow("Control Panel", PANEL_WIDTH, 725)
    cv.moveWindow("Control Panel", PANEL_X, PANEL_Y)
    cv.setMouseCallback("Control Panel", panel_click, (width, height, fps_for_recording, reader))

    cv.namedWindow("Worker Safety Monitoring", cv.WINDOW_NORMAL)
    cv.resizeWindow("Worker Safety Monitoring", display_width, display_height)
    cv.moveWindow("Worker Safety Monitoring", PANEL_X + PANEL_WIDTH + WINDOW_GAP, PANEL_Y)
    cv.setMouseCallback(
        "Worker Safety Monitoring",
        worker_mouse_callback,
        (width, height, display_width, display_height)
    )

    while running:
        loop_start = time.time()
        frame, ts, seq, connected, err, measured_fps = reader.get_latest()
        stream_fps = measured_fps
        now = time.time()
        stale_seconds = now - ts if ts > 0 else 999.0

        if frame is not None:
            last_raw_frame = frame.copy()
            last_frame_time = ts
            last_frame_seq = seq

            # Detection is throttled, but UI display is NOT throttled.
            # YOLO results are refreshed at DETECTION_INTERVAL_SECONDS, while the video view
            # is rendered every loop from the newest raw frame using the latest cached boxes.
            if (not paused_view and seq != last_detection_seq and
                    now - last_detection_time >= DETECTION_INTERVAL_SECONDS):
                try:
                    dets = run_detection(model, frame)
                    disp = render_frame(frame, dets, update_warning_state=True)
                    with state_lock:
                        last_detections = dets
                        last_display = disp
                        last_detection_seq = seq
                        last_detection_time = now
                    det_counter += 1
                except Exception as e:
                    print("Detection error:", e)

            elif not paused_view:
                # Keep the VIEW FPS high: draw the newest raw frame every loop with the
                # most recent YOLO detections. Do not update warning counters here.
                try:
                    with state_lock:
                        cached_detections = list(last_detections)
                    last_display = render_frame(frame, cached_detections, update_warning_state=False)
                except Exception as e:
                    print("Display render error:", e)
                    last_display = frame.copy()

        # Detection FPS counter
        if now - last_det_counter_tick >= 1.0:
            detection_fps = det_counter / max(0.001, now - last_det_counter_tick)
            det_counter = 0
            last_det_counter_tick = now

        # Choose display frame
        base_display = frozen_display if paused_view and frozen_display is not None else last_display
        if base_display is None and last_raw_frame is not None:
            base_display = last_raw_frame.copy()

        if base_display is not None:
            shown_display = get_zoomed_image(base_display, output_size=(display_width, display_height))
            live_state = "PAUSED" if paused_view else (("LIVE" if reader.is_live else "FILE") if connected and stale_seconds <= FRAME_STALE_SECONDS else ("LOST" if reader.is_live else "ENDED"))
            shown_display = draw_video_status_overlay(shown_display, show_record_dot=True, live_state=live_state, stale_seconds=stale_seconds)
            cv.imshow("Worker Safety Monitoring", shown_display)

        # Recording
        if record_video and video_writer is not None:
            check_recording_rotation(width, height, fps_for_recording)
            if video_writer is not None:
                if RECORD_MODE == "clean" and last_raw_frame is not None:
                    record_frame = last_raw_frame.copy()
                    record_frame = draw_video_status_overlay(record_frame, show_record_dot=False, live_state="LIVE", stale_seconds=stale_seconds)
                elif last_display is not None:
                    record_frame = last_display.copy()
                    record_frame = draw_video_status_overlay(record_frame, show_record_dot=False, live_state="LIVE", stale_seconds=stale_seconds)
                else:
                    record_frame = None
                if record_frame is not None:
                    if record_frame.shape[1] != width or record_frame.shape[0] != height:
                        record_frame = cv.resize(record_frame, (width, height), interpolation=cv.INTER_AREA)
                    video_writer.write(record_frame)

        cv.imshow("Control Panel", draw_control_panel(connected, err, stale_seconds))

        # UI FPS counter
        ui_frame_counter += 1
        if now - last_ui_tick >= 1.0:
            ui_fps = ui_frame_counter / max(0.001, now - last_ui_tick)
            ui_frame_counter = 0
            last_ui_tick = now

        key = cv.waitKeyEx(1)
        handle_keyboard(key, width, height, fps_for_recording, reader)

        # Light sleep to avoid maxing CPU when UI is fast
        elapsed = time.time() - loop_start
        if elapsed < 0.005:
            time.sleep(0.005 - elapsed)

    reader.stop()
    stop_recording()
    cv.destroyAllWindows()
