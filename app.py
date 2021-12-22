from datetime import datetime
from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

# video = cv2.VideoCapture('rtsp://freja.hiof.no:1935/rtplive/_definst_/hessdalen03.stream')  # use 0 for web camera
#  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera
# for local webcam use cv2.VideoCapture(0)
cnt = 0
fps = 0
last_fps = 0
avg_fps = 0
min_fps = 999
max_fps = 0
a = datetime.now()

def gen_frames():  # generate frame by frame from camera
    global fps, cnt, avg_fps, min_fps, max_fps, a, last_fps
    video = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        success, frame = video.read()  # read the camera frame
        if not success:
            print('not success')
            print(frame)
            break

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        # concat frame one by one and show result
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        fps += 1
        if (datetime.now() - a).seconds >= 1:
            last_fps = fps
            avg_fps = (avg_fps * cnt + fps) / (cnt + 1)
            cnt += 1
            max_fps = max(max_fps, fps)
            min_fps = min(min_fps, fps)
            fps = 0
            a = datetime.now()


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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
