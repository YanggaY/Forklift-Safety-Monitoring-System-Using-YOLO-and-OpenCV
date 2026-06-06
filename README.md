# CCTV Live Worker Safety Monitoring System
# CCTV 기반 작업자 안전 모니터링 시스템

---

## Project Summary
## 프로젝트 요약

This project is a real-time worker safety monitoring system developed using Python, OpenCV, and YOLO.

The system analyzes CCTV streams or recorded video files to detect workers and vehicles, identify dangerous situations, and provide visual warnings.

본 프로젝트는 Python, OpenCV 및 YOLO를 활용하여 개발한 실시간 작업자 안전 모니터링 시스템이다.

CCTV 영상 또는 녹화 영상을 분석하여 작업자와 차량을 검출하고 위험 상황을 판단하여 시각적 경고를 제공한다.

---

## Main Features
## 주요 기능

- YOLO-based real-time object detection
- Person and vehicle detection
- Danger Zone monitoring
- Ignore Zone support
- Driver false-alarm reduction logic
- 3-second confirmation rule
- RTSP live stream support
- Recorded video file support
- Automatic reconnection
- Video recording
- Image capture
- Zoom and Pan
- Zone Manager
- JSON-based zone storage
- Stream/UI/Detection FPS monitoring
- Graphical control panel

---

## How to Run
## 실행 방법

This program receives a video source through a command-line argument.

If no argument is given, the program uses the default video file:

```python
VIDEO_PATH = "sample.mp4"
```

본 프로그램은 실행할 때 명령행 인자(command-line argument)로 영상 소스를 입력받는다.

아무 인자도 입력하지 않으면 기본 영상 파일인 `sample.mp4`를 사용한다.

---

### 1. Run with Default Video
### 1. 기본 영상으로 실행

```bash
python app.py
```

This runs the program using:

```text
sample.mp4
```

위 명령어는 기본 설정된 `sample.mp4` 파일을 사용하여 실행한다.

---

### 2. Run with a Recorded Video File
### 2. 녹화 영상 파일로 실행

```bash
python app.py sample.mp4
```

Example:

```bash
python app.py forklift_test.mp4
```

This allows the professor or evaluator to test the program using a recorded video file.

교수님 또는 평가자가 녹화된 영상 파일을 이용하여 프로그램을 테스트할 수 있다.

---

### 3. Run with an RTSP CCTV Stream
### 3. RTSP CCTV 스트림으로 실행

```bash
python app.py rtsp://210.99.70.120:1935/live/cctv001.stream
```

The program automatically treats `rtsp://`, `rtmp://`, `http://`, and `https://` sources as live streams.

프로그램은 `rtsp://`, `rtmp://`, `http://`, `https://`로 시작하는 입력을 실시간 스트림으로 처리한다.

---

## Detailed Description
## 상세 기능 설명

### 1. Real-Time Object Detection
YOLO is used to detect persons and vehicles in real time.

YOLO를 이용하여 사람(Person) 및 차량(Car, Truck, Bus)을 실시간으로 검출한다.

---

### 2. Danger Zone Monitoring

Users can define custom danger zones.

When a worker enters a danger zone, the system changes its status and displays a warning.

사용자가 위험구역(Danger Zone)을 직접 지정할 수 있으며, 작업자가 해당 구역에 진입하면 경고 상태를 표시한다.

---

### 3. Ignore Zone Monitoring

Ignore zones exclude unnecessary regions from safety analysis.

Objects detected inside ignore zones are ignored.

무시구역(Ignore Zone)을 설정하여 불필요한 영역을 분석 대상에서 제외할 수 있다.

---

### 4. Driver Detection and False-Alarm Reduction

The system checks whether a detected person is located inside a vehicle bounding box.

This logic is designed to reduce false alarms caused by forklift drivers being incorrectly detected as workers inside dangerous areas.

차량 내부의 사람을 별도로 판별하여 지게차 운전자가 위험구역 작업자로 잘못 인식되는 오탐을 줄인다.

A warning is generated only when the detection remains valid continuously for approximately three seconds.

또한 약 3초 동안 연속 검출된 경우에만 경고를 발생시켜 순간적인 오검출을 줄인다.

---

### 5. RTSP Live CCTV Support

Supports RTSP-based CCTV streams.

Latest-frame processing is used to reduce latency.

RTSP 기반 CCTV 스트림을 지원하며 최신 프레임 기반 처리로 지연을 최소화한다.

---

### 6. Recorded Video File Support

The program can analyze recorded video files as well as live streams.

실시간 CCTV뿐 아니라 녹화된 영상 파일도 분석할 수 있다.

---

### 7. Automatic Reconnection

The system automatically reconnects when the RTSP connection is lost.

RTSP 연결이 끊어진 경우 자동으로 재연결을 시도한다.

---

### 8. Video Recording System

Supports video recording and automatic file saving.

영상 녹화 및 자동 저장 기능을 제공한다.

---

### 9. Image Capture System

The current monitoring screen can be saved as an image.

현재 모니터링 화면을 이미지로 저장할 수 있다.

---

### 10. Zoom and Pan Functions

Supports zooming with the mouse wheel and panning by dragging.

마우스 휠 확대/축소 및 드래그 이동 기능을 지원한다.

---

### 11. Zone Manager

Provides creation, deletion, and management of danger zones and ignore zones.

위험구역과 무시구역의 생성, 삭제 및 관리 기능을 제공한다.

Zone information is stored in JSON files and automatically loaded later.

구역 정보는 JSON 파일로 저장 및 재사용된다.

---

### 12. Performance Monitoring

Displays:

- Stream FPS
- UI FPS
- Detection FPS
- System status

Stream FPS, UI FPS, Detection FPS 및 현재 상태를 표시한다.

---

### 13. User Interface & Control Method

The system consists of a Control Panel and a Monitoring Window.

Most functions can be controlled using the mouse through the graphical control panel.

사용자 인터페이스는 Control Panel과 Monitoring Window로 구성된다.

대부분의 기능은 마우스를 이용하여 조작할 수 있다.

Mouse-supported functions:

- Record
- Capture
- Reconnect
- Pause View
- Zoom Control
- Zone Manager

마우스로 지원되는 기능:

- 녹화
- 캡처
- 재연결
- 화면 정지
- 확대/축소
- 구역 관리

Zone points are selected using the mouse.

However, saving, canceling, and resetting a newly created zone currently require keyboard input.

구역 점 선택은 마우스로 수행된다.

다만 새 구역의 저장, 취소, 초기화는 현재 키보드 입력이 필요하다.

---

## Folder Structure
## 폴더 구조

```text
Recordings/    -> Recorded videos
Captures/      -> Captured images
danger_zones.json
ignore_zones.json
```

---

## Required Libraries
## 필요 라이브러리

```bash
pip install opencv-python numpy ultralytics
```

---

## Expected Benefits
## 기대 효과

- Improved workplace safety
- Continuous monitoring
- Reduced manual supervision workload
- Faster identification of dangerous situations

- 산업현장 안전성 향상
- 지속적인 감시
- 관리 부담 감소
- 위험 상황 조기 발견

---

## Conclusion
## 결론

This project demonstrates a practical CCTV-based worker safety monitoring system using computer vision and deep learning.

본 프로젝트는 컴퓨터 비전과 딥러닝을 활용한 실용적인 CCTV 기반 작업자 안전 모니터링 시스템을 구현하였다.

---

## Author

Jun Seong Yang

2026
