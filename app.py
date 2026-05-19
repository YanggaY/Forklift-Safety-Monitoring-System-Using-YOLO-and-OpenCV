import cv2 as cv
import numpy as np
from ultralytics import YOLO

VIDEO_PATH = "warehouse.mp4"
MODEL_NAME = "yolov8n.pt"
TARGET_CLASSES = ["person", "car", "truck", "bus"]

YOLO_CONF = 0.25
YOLO_IOU = 0.45
YOLO_IMG_SIZE = 640

DANGER_ZONE = np.array([
    (324, 355), (1179, 376), (1276, 504),
    (1276, 716), (10, 716), (11, 468)
], dtype=np.int32)

paused = False
running = True

def draw_text(img, text, pos, scale=0.7, color=(255, 255, 255), thickness=2):
    x, y = pos
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(img, text, (x + 1, y + 1), font, scale, (50, 50, 50), thickness, cv.LINE_AA)
    cv.putText(img, text, pos, font, scale, color, thickness, cv.LINE_AA)

def box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) // 2, (y1 + y2) // 2

def point_in_zone(point, zone):
    return cv.pointPolygonTest(zone, point, False) >= 0

def run_detection(model, frame):
    results = model(
        frame,
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

            if class_name not in TARGET_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_xyxy = (x1, y1, x2, y2)
            center = box_center(box_xyxy)

            detections.append({
                "class": class_name,
                "box": box_xyxy,
                "center": center
            })

    return detections

def render_frame(frame, detections):
    display = frame.copy()
    person_in_danger = False

    cv.polylines(display, [DANGER_ZONE], True, (0, 255, 0), 2)
    draw_text(display, "Danger Zone", tuple(DANGER_ZONE[0]), 0.65, (0, 255, 0), 2)

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        center = det["center"]
        class_name = det["class"]

        if class_name == "person" and point_in_zone(center, DANGER_ZONE):
            person_in_danger = True
            color = (0, 0, 255)
            label = "PERSON IN DANGER"
        elif class_name == "person":
            color = (255, 0, 255)
            label = "person"
        else:
            color = (0, 255, 255)
            label = "forklift"

        cv.rectangle(display, (x1, y1), (x2, y2), color, 2)
        cv.circle(display, center, 3, color, -1)
        draw_text(display, label, (x1, y1 - 8), 0.55, color, 2)

    status = "WARNING: Person in danger zone" if person_in_danger else "Status: Safe"
    status_color = (0, 0, 255) if person_in_danger else (0, 255, 0)

    draw_text(display, status, (30, 40), 1.0, status_color, 3)
    draw_text(display, "[SPACE] Pause | [ESC] Quit", (30, 80), 0.65, (255, 255, 0), 2)

    return display

model = YOLO(MODEL_NAME)
cap = cv.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise SystemExit("Cannot open video file. Please check VIDEO_PATH.")

ret, frame = cap.read()

if not ret:
    raise SystemExit("Cannot read video file.")

display = render_frame(frame, run_detection(model, frame))

while running:
    if not paused:
        ret, frame = cap.read()

        if not ret:
            break

        detections = run_detection(model, frame)
        display = render_frame(frame, detections)

    cv.imshow("Forklift Safety Monitoring Prototype", display)

    key = cv.waitKey(1) & 0xFF

    if key == 27:
        running = False
    elif key == 32:
        paused = not paused

cap.release()
cv.destroyAllWindows()