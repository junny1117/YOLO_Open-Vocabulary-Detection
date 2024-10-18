# YOLO_Open-Vocabulary-Detection
## 개요
YOLO를 활용한 객체 검지 웹사이트

## 주요 기능

- 80개의 사전 학습된 객체 목록([참고](objectlist.txt)) 중 사용자가 입력한 객체 검지
- 웹 기반 인터페이스로 편리한 사용 가능

## 사용도구/기술
- **Flask**: 웹 기반 인터페이스
- **YOLOv8**: 객체검지
- **OpenCV**: 비디오처리
- **HTML, CSS**: 웹페이지 탬플릿

## 파일 목록
### app.py - Flask 파일, 다양한 기능들을 통합하고 웹을 통해 동작하도록 함
### index.html - 메인 페이지 템플릿
### events.html - 결과 조회 페이지 템플릿
### requirements.txt - 실행에 필요한 패키지 목록

## 실행 방법
1. 프로젝트 클론: `git clone https://github.com/junny1117/Warehouse-System-with-YOLO`
2. 필요한 패키지 설치: `pip install -r requirements.txt`
3. Flask 서버 실행: `flask run`
4. 브라우저에서 `127.0.0.1:5000`으로 접속

## 실행 결과 이미지
![image](https://github.com/user-attachments/assets/20670f87-f84f-425c-9f9f-2307dc8033e8)



