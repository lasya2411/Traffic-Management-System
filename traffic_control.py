import cv2
import numpy as np
from ultralytics import YOLO
from flask import Flask, send_from_directory, request, Response, redirect, url_for, jsonify
from time import time, sleep
import threading
import pyttsx3
import base64

app = Flask(__name__)

# Initialize components
model = YOLO("yolov8n.pt")
vehicle_classes = ['car', 'bus', 'truck', 'motorcycle', 'bicycle', 'van']
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.8)

# Traffic signal parameters
signal_params = {
    'base_green_time': 30,
    'min_green_time': 10,
    'red_time': 60,
    'yellow_time': 5,
    'current_signal': 'red',
    'last_change': time(),
    'vehicle_count': 0,
    'is_running': False,
    'source': 'camera',
    'time_left': 60,
    'processed_image': None,
    'signal_thread': None,
    'current_green_time': 30
}

video_capture = None
processing_lock = threading.Lock()

def speak(message):
    def _speak():
        try:
            local_engine = pyttsx3.init()
            local_engine.setProperty('rate', 150)
            local_engine.setProperty('volume', 0.8)
            local_engine.say(message)
            local_engine.runAndWait()
        except Exception as e:
            print(f"Speech synthesis error: {e}")
    threading.Thread(target=_speak, daemon=True).start()

def calculate_green_time(vehicle_count):
    if vehicle_count <= 5:
        return signal_params['min_green_time']
    elif vehicle_count >= 15:
        return signal_params['base_green_time']
    else:
        return min(
            signal_params['base_green_time'],
            signal_params['min_green_time'] + (vehicle_count - 5) * 2
        )

def control_signals():
    print("[Signal Thread] Started")
    while signal_params['is_running']:
        try:
            current_time = time()
            elapsed = current_time - signal_params['last_change']

            with processing_lock:
                current_signal = signal_params['current_signal']

                if current_signal == "red":
                    if elapsed >= signal_params['red_time']:
                        signal_params['current_green_time'] = calculate_green_time(signal_params['vehicle_count'])
                        change_signal("green")
                        speak(f"Signal changed to green. Green time: {signal_params['current_green_time']} seconds")

                elif current_signal == "green":
                    if elapsed >= signal_params['current_green_time']:
                        change_signal("yellow")
                        speak("Signal changed to yellow")

                elif current_signal == "yellow":
                    if elapsed >= signal_params['yellow_time']:
                        change_signal("red")
                        speak("Signal changed to red")

                # Update time left
                if current_signal == "green":
                    signal_params['time_left'] = max(0, int(signal_params['current_green_time'] - elapsed))
                else:
                    key = f"{current_signal}_time"
                    signal_params['time_left'] = max(0, int(signal_params[key] - elapsed))

        except Exception as e:
            print(f"[Signal Thread] Error: {e}")

        sleep(0.1)

def change_signal(new_signal):
    signal_params['current_signal'] = new_signal
    signal_params['last_change'] = time()
    if new_signal == "green":
        signal_params['time_left'] = signal_params['current_green_time']
    else:
        signal_params['time_left'] = signal_params[f"{new_signal}_time"]

def detect_vehicles(frame):
    results = model(frame, verbose=False)[0]
    vehicle_count = 0

    for box in results.boxes:
        class_id = int(box.cls[0].item())
        class_name = model.names[class_id]
        if class_name in vehicle_classes:
            vehicle_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frame, vehicle_count

def generate_frames():
    global video_capture
    video_capture = cv2.VideoCapture(0)

    while signal_params['is_running'] and signal_params['source'] == 'camera':
        success, frame = video_capture.read()
        if not success:
            break

        processed_frame, count = detect_vehicles(frame)
        with processing_lock:
            signal_params['vehicle_count'] = count

        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    if video_capture and video_capture.isOpened():
        video_capture.release()
        video_capture = None

def process_uploaded_image(file_stream):
    try:
        file_bytes = np.frombuffer(file_stream.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if frame is not None:
            processed_frame, count = detect_vehicles(frame)

            with processing_lock:
                signal_params['vehicle_count'] = count
                signal_params['processed_image'] = image_to_base64(processed_frame)
                signal_params['current_green_time'] = calculate_green_time(count)
                signal_params['current_signal'] = 'red'
                signal_params['last_change'] = time()
                signal_params['time_left'] = signal_params['red_time']

            speak(f"Detected {count} vehicles. Green time will be {signal_params['current_green_time']} seconds")

            if not signal_params['is_running']:
                signal_params['is_running'] = True
                signal_params['source'] = 'image'
                signal_params['signal_thread'] = threading.Thread(target=control_signals, daemon=True)
                signal_params['signal_thread'].start()
    except Exception as e:
        print(f"Image processing error: {e}")

def image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/video_feed')
def video_feed():
    if signal_params['is_running'] and signal_params['source'] == 'camera':
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(status=204)

@app.route('/control', methods=['POST'])
def control():
    global video_capture
    action = request.form.get('action')
    source = request.form.get('source')

    with processing_lock:
        if action == 'start':
            if not signal_params['is_running']:
                signal_params['is_running'] = True
                signal_params['source'] = source
                signal_params['last_change'] = time()
                signal_params['current_signal'] = 'red'
                signal_params['time_left'] = signal_params['red_time']
                if source == 'camera':
                    signal_params['processed_image'] = None
                    signal_params['signal_thread'] = threading.Thread(target=control_signals, daemon=True)
                    signal_params['signal_thread'].start()
                speak("System started. Signal is red")

        elif action == 'stop':
            signal_params['is_running'] = False
            signal_params['processed_image'] = None
            if video_capture and video_capture.isOpened():
                video_capture.release()
                video_capture = None
            speak("System stopped")

    return redirect(url_for('index'))

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return redirect(url_for('index'))
    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('index'))

    process_uploaded_image(file.stream)
    return redirect(url_for('index'))

@app.route('/get_status')
def get_status():
    with processing_lock:
        return jsonify({
            'current_signal': signal_params['current_signal'],
            'time_left': signal_params['time_left'],
            'vehicle_count': signal_params['vehicle_count'],
            'current_green_time': signal_params['current_green_time'],
            'base_green_time': signal_params['base_green_time'],
            'red_time': signal_params['red_time'],
            'yellow_time': signal_params['yellow_time'],
            'processed_image': signal_params['processed_image'],
            'is_running': signal_params['is_running'],
            'source': signal_params['source']
        })

if __name__ == '__main__':
    app.run(debug=True, threaded=True, host='0.0.0.0')
