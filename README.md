# YOLO_Open-Vocabulary-Detection
## 개요
YOLO를 활용한 객체 검지 웹사이트

## 구현 사항

- 80개의 **사전 학습된 객체**([참고](objectlist.txt)) 중 **사용자**가 **입력**한 **객체 검지**
- 웹 기반 인터페이스로 편리한 사용 가능

## 사용도구/기술
- **Python**: 개발언어
- **Flask**: 웹 기반 인터페이스
- **YOLOv8**: 객체검지
- **OpenCV**: 비디오처리
- **HTML, CSS**: 웹페이지 탬플릿
- **Visual Studio Code**: 코드 작성
- **Windows**: 운영체제

## 파일 목록
### app.py - Flask 파일, 다양한 기능들을 통합하고 웹을 통해 동작하도록 함
### index.html - 메인 페이지 템플릿
### events.html - 결과 조회 페이지 템플릿
### requirements.txt - 실행에 필요한 패키지 목록
### yolov8s-worldv2.pt - YOLOv8 World 객체 검지 모델

## 실행 방법
1. 프로젝트 클론: `git clone https://github.com/junny1117/YOLO_Open-Vocabulary-Detection`
2. 필요한 패키지 설치: `pip install -r requirements.txt`
3. Flask 서버 실행: `flask run`
4. 브라우저에서 `127.0.0.1:5000`으로 접속

## 실행 결과 이미지
![image](https://github.com/user-attachments/assets/542e844e-81cd-4300-b5b0-4370955f3c71)




