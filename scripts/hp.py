import cv2
from ultralytics import YOLO
import time
import os
from flask import Flask, Response

app = Flask(__name__)

# ======================
# 🔧 KONFIGURASI
# ======================
url = "http://192.168.106.190:4747/video"
model_path = r"D:\Quant_ML_Project\ML.py\EasyPark\Model\parking_detection2\weights\best.pt"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model tidak ditemukan: {model_path}")

# ======================
# 🚀 LOAD MODEL
# ======================
print("🔥 Loading model...")
try:
    model = YOLO(model_path).to("cuda")
    print("✅ Model di GPU\n")
except:
    model = YOLO(model_path)
    print("✅ Model di CPU\n")

# ======================
# 🎥 KAMERA
# ======================
print("📸 Connecting to camera...")
cap = cv2.VideoCapture(url)
if not cap.isOpened():
    raise Exception("❌ Cannot connect to camera")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ======================
# 🎬 STREAM GENERATOR
# ======================
def generate():
    prev_time = time.time()
    fps_counter = 0
    fps_display = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        
        # Deteksi
        results = model(frame, conf=0.5, verbose=False)
        annotated_frame = results[0].plot()
        
        # FPS
        fps_counter += 1
        current_time = time.time()
        if current_time - prev_time >= 1:
            fps_display = fps_counter / (current_time - prev_time)
            fps_counter = 0
            prev_time = current_time
        
        # Info
        cv2.putText(annotated_frame, f"FPS: {fps_display:.1f}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        num_detections = len(results[0].boxes)
        cv2.putText(annotated_frame, f"Objects: {num_detections}", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # Encode
        ret, buffer = cv2.imencode('.jpg', annotated_frame, 
                                   [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ======================
# 🌐 ROUTES
# ======================
@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Parking Detection</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                background: linear-gradient(135deg, #0f0f0f 0%, #1a1a2e 100%);
                color: #fff;
                font-family: 'Segoe UI', sans-serif;
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 20px;
                min-height: 100vh;
            }
            h1 {
                color: #00ff88;
                text-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
                margin-bottom: 10px;
                font-size: 2rem;
            }
            .status {
                background: linear-gradient(90deg, #ff0000, #ff4444);
                padding: 5px 15px;
                border-radius: 20px;
                font-size: 0.9rem;
                margin-bottom: 20px;
                animation: pulse 2s infinite;
            }
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.7; }
            }
            .container {
                position: relative;
                max-width: 95%;
                width: 100%;
                max-width: 1200px;
                box-shadow: 0 20px 60px rgba(0, 255, 136, 0.2);
                border-radius: 15px;
                overflow: hidden;
                border: 2px solid #00ff88;
            }
            img {
                width: 100%;
                display: block;
            }
            .info {
                margin-top: 30px;
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                max-width: 1200px;
                width: 100%;
            }
            .card {
                background: rgba(0, 255, 136, 0.1);
                padding: 20px;
                border-radius: 10px;
                border: 1px solid rgba(0, 255, 136, 0.3);
                text-align: center;
            }
            .card-icon { font-size: 2rem; margin-bottom: 10px; }
            .card-title { color: #00ff88; font-weight: bold; margin-bottom: 5px; }
            .card-text { color: #aaa; font-size: 0.9rem; }
        </style>
    </head>
    <body>
        <h1>🚗 Parking Detection System</h1>
        <div class="status">🔴 LIVE</div>
        <div class="container">
            <img src="/video" alt="Stream">
        </div>
        <div class="info">
            <div class="card">
                <div class="card-icon">📡</div>
                <div class="card-title">Source</div>
                <div class="card-text">IP Webcam Stream</div>
            </div>
            <div class="card">
                <div class="card-icon">🎯</div>
                <div class="card-title">Model</div>
                <div class="card-text">YOLOv8 Nano</div>
            </div>
            <div class="card">
                <div class="card-icon">⚡</div>
                <div class="card-title">Acceleration</div>
                <div class="card-text">CUDA GPU</div>
            </div>
        </div>
    </body>
    </html>
    """

@app.route('/video')
def video():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ======================
# 🚀 RUN
# ======================
if __name__ == '__main__':
    print("\n" + "="*50)
    print("🌐 Web Server Running!")
    print("="*50)
    print("🔗 Open in browser:")
    print("   http://localhost:5000")
    print("\n⌨️  Press Ctrl+C to stop")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)