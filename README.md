# CCTV Worker Safety Monitoring System

Real-time CCTV-based worker safety monitoring system with Dynamic Zone Tracking, Forklift Occupancy Detection, and Low-Latency RTSP Processing.

산업 현장 CCTV 영상을 활용하여 작업자와 차량을 실시간으로 감지하고 위험 상황을 분석하는 안전 모니터링 시스템입니다.

YOLOv8 객체 검출, Dynamic Zone Tracking, Danger Zone Monitoring, Forklift Occupancy Detection 기능을 통합하여 작업자 안전을 향상시키도록 설계되었습니다.

실제 지게차 운전수 경력과 CCTV 근무 경력을 토대로 만들었습니다.

RTSP 실시간 스트림과 녹화 영상 모두 지원하며 장시간 무인 모니터링 환경을 고려하여 개발되었습니다.

sample 영상은 실제 제가 일했던 호주의 Cotton warehouse 현장 영상입니다.

## Requirements

- Python 3.10+
- OpenCV
- NumPy
- Ultralytics YOLOv8

## Installation

```bash
pip install ultralytics opencv-python numpy
```

또는

```bash
pip install -r requirements.txt
```

## Running the Application

### RTSP Stream

```python
VIDEO_SOURCE = "rtsp://username:password@ip-address:port/stream"
```

```bash
python app.py
```

### Video File

```python
VIDEO_SOURCE = "video.mp4"
```

```bash
python app.py
```

### Command Line Execution

```bash
python app.py video.mp4
python app.py videos/forklift_test.mp4
python app.py "D:\Videos\forklift_test.mp4"
python app.py rtsp://username:password@ip-address:554/stream
```

## Project Structure

```text
.
├── app.py
├── yolov8m.pt
├── danger_zones.json
├── ignore_zones.json
├── Recordings/
│   └── YYYY-MM-DD/
├── Captures/
│   └── YYYY-MM-DD/
└── README.md
```

## Core Safety Features

### Worker Detection

작업자(Person)를 실시간으로 감지합니다.

Ignore Zone 내부 객체는 자동으로 분석 대상에서 제외됩니다.

Detects workers (Person) in real time.

Objects inside Ignore Zones are automatically excluded from evaluation.

### Vehicle Detection

Car, Truck, Bus 객체를 실시간으로 감지합니다.

안전 분석 단계에서는 모든 차량을 Forklift 객체로 통합 처리합니다.

Detects Car, Truck, and Bus objects in real time.

All vehicle classes are treated as Forklift objects during safety analysis.

| YOLO Class | Display |
|------------|----------|
| Person | Person |
| Car | Forklift |
| Truck | Forklift |
| Bus | Forklift |

### Danger Zone Monitoring

사용자가 직접 지정한 다각형 위험구역을 감시합니다.

약 1초 이상 연속 검출된 경우에만 실제 경고가 발생합니다.

Monitors user-defined polygon danger zones.

Warnings are triggered only after approximately one second of continuous detection.

### Forklift Occupancy Detection

Bounding Box 중심점, 객체 겹침률, 객체 크기를 분석합니다.

겹침률 60% 이상이며 차량 Bounding Box 내부에서 약 3초 이상 연속 검출된 경우에만 경고를 발생시킵니다.

Analyzes center-point position, overlap ratio, and relative object size.

Warnings are triggered only after approximately three seconds of continuous detection inside a vehicle.

## False Alarm Reduction

- Ignore Zone Filtering
- Danger Zone Confirmation Timer
- Forklift Confirmation Counter
- Bounding Box Overlap Analysis
- Relative Object Size Comparison
- Continuous Detection Verification

## Dynamic Zone Tracking

- Good Features To Track
- Lucas-Kanade Optical Flow
- Median Motion Estimation

## Low-Latency RTSP Monitoring

Latest Frame Reader 구조를 사용하여 항상 최신 프레임만 처리합니다.

Only the newest frame is retained while older frames are discarded.

## Automatic Reconnection

RTSP 연결이 끊어진 경우 자동 재연결을 수행합니다.

Automatically reconnects when an RTSP stream is interrupted.

## Recording System

### Analysis Mode

- Detection Boxes
- Danger Zones
- Ignore Zones
- Status Information
- Timestamp

### Clean Mode

원본 영상 기반 녹화를 수행합니다.

### Segmented Recording

자동 파일 분할 녹화를 지원합니다.

## Image Capture

현재 화면을 이미지 파일로 저장할 수 있습니다.

Allows saving the current frame as an image.

## Zoom & Pan

마우스 휠 확대 및 드래그 이동을 지원합니다.

Supports zooming and panning.

## Pause View

화면 표시만 일시 정지하며 검출과 녹화는 계속 수행됩니다.

## Zone Manager

Danger Zone 및 Ignore Zone을 추가, 삭제 및 관리할 수 있습니다.

## mkv recordings

mkv형식을 사용해 혹시 프로그램이 강제종료 되거나, 정전 상황에서도 녹화영상을 복구 하기 용이하게 파일 형식을 설정했습니다.

## Keyboard Controls

| Key | Function |
|------|----------|
| Space | Pause View |
| A | Add Danger Zone |
| I | Add Ignore Zone |
| M | Zone Manager |
| S | Start / Stop Recording |
| C | Capture Image |
| R | Reconnect Stream |
| + | Zoom In |
| - | Zoom Out |
| 0 | Reset Zoom |
| [ | Decrease Recording Duration |
| ] | Increase Recording Duration |
| ESC | Exit |

## Mouse Controls

| Action | Function |
|---------|----------|
| Left Click | Add Zone Point |
| Left Drag | Pan View |
| Right Click | Complete Zone |
| Mouse Wheel | Zoom In / Out |

## Output Files

### Recordings

```text
Recordings/
└── YYYY-MM-DD/
```

### Captures

```text
Captures/
└── YYYY-MM-DD/
```

### Zone Files

```text
danger_zones.json
ignore_zones.json
```

## Typical Use Cases

- Warehouse Safety Monitoring
- Forklift Operation Monitoring
- Logistics Centers
- Manufacturing Plants
- Construction Sites
- Industrial CCTV Monitoring

## Future Improvements

- Multi-Camera Support
- Web Dashboard
- Email Alerts
- SMS Notifications
- AI Event Search
- Heatmap Analytics
- Cloud Recording
- Centralized Monitoring Server

## License

This project is intended for educational and research purposes.
