from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLOWorld
import torch
import cv2
import time

app = Flask(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

model = YOLOWorld('yolov8s-worldv2.pt')
model.to(device)  

desired_classes = [""]
model.set_classes(desired_classes)  

def generate_frames():
    cap = cv2.VideoCapture(0)  
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    prev_time = 0  

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
         
            results = model.predict(frame, device=device)  # YOLOv8 모델로 객체 탐지 수행
        except IndexError as e:
            print(f"IndexError 발생: {e}")
            continue  

        annotated_frame = results[0].plot()
    
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()  

@app.route('/')
def index():
    return render_template('index.html', current_classes=desired_classes)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update_classes', methods=['POST'])
def update_classes():
    global desired_classes
    desired_classes_str = request.form.get('desired_class', '')
    desired_classes = [cls.strip() for cls in desired_classes_str.split(',') if cls.strip()]

    model.set_classes(desired_classes)
    
    print(f"Updated desired_classes: {desired_classes}")  

    return jsonify({'status': 'success', 'classes': desired_classes}) 

if __name__ == '__main__':
    app.run(debug=True)
