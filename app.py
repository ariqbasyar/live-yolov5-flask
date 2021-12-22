import json
from os import error
import re
import cv2

from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, Response, request, jsonify
from models.experimental import attempt_load
from model import get_device, box_label, detect, preprocess
from utils.general import methods

FILE = Path(__file__).resolve()
ROOT = FILE.parent
WEIGHTS = ROOT / 'weights'

device = get_device()

app = Flask(__name__)

weights = {
    'none': None,
    'yolov5s': 'yolov5s',
    'yolov5m': 'yolov5m',
    'yolov5l': 'yolov5l',
    'yolov5x': 'yolov5x',
}

model_key = 'none'
model = weights[model_key]
model_changed = False

cnt = 0
fps = 0
last_fps = 0
avg_fps = 0
min_fps = 999
max_fps = 0
start_time = datetime.now()

def gen_frames():  # generate frame by frame from camera
    global fps, cnt, avg_fps, min_fps, max_fps, start_time, last_fps,\
        model, model_changed
    video = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        success, frame = video.read()  # read the camera frame
        if not success:
            print('not success')
            print(frame)
            break

        preprocessed = preprocess(frame, device)
        if model_changed:
            model = weights[model_key]
            model_changed = False
        if model is not None:
            pred = detect(model,preprocessed)
            data = box_label(pred, frame)
        else:
            data = frame
        ret, buffer = cv2.imencode('.jpg', data)
        bytes_frame = buffer.tobytes()
        # concat frame one by one and show result
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + bytes_frame + b'\r\n')
        fps += 1
        if (datetime.now() - start_time).seconds >= 1:
            last_fps = fps
            avg_fps = (avg_fps * cnt + fps) / (cnt + 1)
            cnt += 1
            max_fps = max(max_fps, fps)
            min_fps = min(min_fps, fps)
            fps = 0
            start_time = datetime.now()


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


@app.route('/analytics')
def get_analytics():
    return {
        'fps': last_fps,
        'avg': avg_fps,
        'min': min_fps,
        'max': max_fps,
    }

@app.errorhandler(403)
@app.route('/change-model', methods=['POST'])
def change_model():
    global model_key, model_changed
    data = request.get_json()
    if data is None:
        return jsonify(error='No json were given'), 403
    _type = data.get('modelType')
    if _type not in weights.keys():
        return jsonify(error='Invalid Model Key'), 403
    model_key = _type
    model_changed = True
    return jsonify(result=200)

def load_models(device):
    for key,val in weights.items():
        if val is None: continue
        m = attempt_load(WEIGHTS / val, map_location=device)
        weights[key] = m

if __name__ == '__main__':
    load_models(device)
    app.run(debug=True, host='0.0.0.0')
