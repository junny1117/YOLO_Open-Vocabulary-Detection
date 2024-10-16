from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLOWorld
import torch
import cv2
import time

app = Flask(__name__)

# GPU가 사용 가능한지 확인 (GPU가 없으면 CPU 사용)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# YOLOv8 모델 불러오기, 선택한 장치로 로드
model = YOLOWorld('yolov8s-worldv2.pt')
model.to(device)  # 모델을 선택한 장치로 이동

desired_classes = [""]
model.set_classes(desired_classes)  # 초기 필터 설정

def generate_frames():
    cap = cv2.VideoCapture(0)  # 웹캠 비디오 캡처 시작
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    prev_time = 0  # FPS 계산을 위한 이전 시간 초기화

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        # FPS 계산
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
        prev_time = current_time

        try:
            # YOLO 모델로 예측 수행
            results = model.predict(frame, device=device)  # YOLOv8 모델로 객체 탐지 수행
        except IndexError as e:
            print(f"IndexError 발생: {e}")
            continue  # 오류 발생 시 현재 프레임을 건너뛰고 다음 프레임으로 진행

        # 결과를 시각화하여 프레임에 그리기
        annotated_frame = results[0].plot()

        # FPS를 프레임에 추가
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 프레임을 JPEG로 인코딩
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()  # 웹캠 자원 해제

@app.route('/')
def index():
    return render_template('index.html', current_classes=desired_classes)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update_classes', methods=['POST'])
def update_classes():
    global desired_classes
    # 입력받은 객체 이름을 쉼표로 구분하여 리스트로 처리
    desired_classes_str = request.form.get('desired_class', '')
    desired_classes = [cls.strip() for cls in desired_classes_str.split(',') if cls.strip()]

    # YOLOWorld 모델의 클래스 필터 업데이트
    model.set_classes(desired_classes)
    
    print(f"Updated desired_classes: {desired_classes}")  # 디버깅을 위한 출력

    return jsonify({'status': 'success', 'classes': desired_classes})  # JSON 응답 반환

if __name__ == '__main__':
    app.run(debug=True)
