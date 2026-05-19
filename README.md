# Forklift Safety Monitoring using YOLO and OpenCV

## 프로젝트 소개 (Project Description)

이 프로젝트는 산업현장 및 물류창고 환경에서 작업자의 안전을 보조하기 위한  
컴퓨터 비전 기반 안전 모니터링 프로그램을 만드는 것을 목표로 합니다.

프로그램은 YOLO와 OpenCV를 이용하여 영상 속 사람(person)과 지게차 및 차량(car, truck, bus)을 인식하고,  
작업자가 위험구역(Danger Zone)에 진입할 경우 화면에 경고를 표시합니다.

현재는 텀 프로젝트 제안(Proposal) 단계이며,  
기본적인 프로토타입 기능만 구현된 상태입니다.  
프로젝트 기간 동안 UI 개선, 위험구역 관리 기능, 오탐 감소 기능 등을 추가하여  
최종적으로 실제 산업현장에서도 활용 가능한 형태로 발전시키는 것이 목표입니다.

This project aims to build a computer vision-based safety monitoring system for industrial or warehouse environments using YOLO and OpenCV.  
The current repository is an early prototype for the term project proposal stage.

<br>

---

# 프로젝트 제작 계기 (Motivation)

산업현장에서 지게차 운전수로 근무하면서 느꼈던 위험한 상황들을 바탕으로 제작하게 되었습니다.

특히 다음과 같은 상황들은 실제 산업사고로 이어질 가능성이 높다고 생각했습니다.

- 지게차 작업구역에 작업자가 지나가는 상황
- 지게차 사각지대(blind spot)에 사람이 존재하는 상황
- 좁은 공간에 너무 많은 장비가 동시에 존재하는 상황

이 프로젝트는 이러한 위험 요소를 자동으로 감지하여  
현장 안전관리자 또는 CCTV 근무자가 위험 상황을 빠르게 인지하고 대응할 수 있도록 돕는 것을 목표로 합니다.

<br>

---

# 현재 프로토타입 기능 (Current Prototype Features)

- OpenCV를 이용한 비디오 파일 재생
- YOLO 기반 객체 인식(Object Detection)
- 사람(person) 및 차량 관련 클래스 감지
- 감지된 객체에 Bounding Box 표시
- 고정된 Danger Zone 표시
- 위험구역 내부 사람 감지 시 경고 출력
- 일시정지(Pause) 및 종료(Exit) 기능

### Prototype Features (English)

- Load a video file using OpenCV
- Run YOLO object detection
- Detect people and vehicle-related classes
- Draw bounding boxes on detected objects
- Display a fixed danger zone
- Show warning messages when a person enters the danger zone
- Basic pause and exit controls

<br>

---

# 최종 구현 예정 기능 (Planned Final Features)

- 마우스를 이용한 Danger Zone 생성 기능
- JSON 기반 Danger Zone / Ignore Zone 저장 및 불러오기
- Ignore Zone 설정 기능
- 재생, 정지, 배속, 녹화 등을 위한 UI Control Panel 제작
- 화면 녹화 및 결과 영상 저장 기능
- CCTV / RTSP 스트림 입력 지원
- 지게차 내부 작업자를 보행자로 인식하는 문제 개선
- Warning Delay Logic을 이용한 오탐(False Alarm) 감소
- 현장 환경에 맞는 YOLO 모델 테스트 및 최적화
- 사용자 친화적인 UI 구성

### Planned Features (English)

- Mouse-based danger zone selection
- Save and load danger zones using JSON files
- Ignore zone setting
- Better UI control panel for playback and recording
- Processed video recording and saving
- CCTV / RTSP stream support
- Reduce false alarms
- Test and optimize different YOLO models

<br>

---

# 사용 기술 (Technologies)

- Python
- OpenCV
- Ultralytics YOLO
- NumPy
- JSON (planned)

<br>

---

# 기대 결과 (Expected Final Result)

최종적으로는 현장 안전관리자 또는 CCTV 근무자가  
위험 상황을 빠르게 파악하고 대응할 수 있도록 보조하는 프로그램을 만드는 것이 목표입니다.

또한 처음 사용하는 사람도 쉽게 사용할 수 있도록  
직관적이고 사용자 친화적인 UI를 구성하는 것을 목표로 하고 있습니다.

The final goal is to create a practical safety monitoring system that helps workers or CCTV operators quickly identify dangerous situations in industrial environments.

<br>

---

# 실행 방법 (How to Run Prototype)

```bash
pip install opencv-python ultralytics numpy
python app_proposal.py
```

실행 전 테스트 영상 파일을 프로젝트 폴더 안에 넣고,  
코드 내부 `VIDEO_PATH` 경로를 올바르게 설정해야 합니다.

<br>

---

# 프로젝트 상태 (Project Status)

현재는 Proposal(제안서) 단계의 초기 프로토타입입니다.

현재 코드는 최종 제출 버전이 아니며,  
프로젝트 진행 과정에서 기능 추가 및 성능 개선이 계속 이루어질 예정입니다.
