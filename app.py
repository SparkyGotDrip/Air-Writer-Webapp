"""
Air Writer — Web App (Optimized Server Version)
"""

import cv2
import mediapipe as mp
import numpy as np
import random
import math
import time
import base64
import os
import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template_string, request
from flask_socketio import SocketIO
from threading import Lock

# --------------------------------------------------
# Thread-safe client state
# --------------------------------------------------

state_lock = Lock()
client_states = {}

# --------------------------------------------------
# Flask + SocketIO
# --------------------------------------------------

app = Flask(__name__)
app.config["SECRET_KEY"] = "airwriter-secret"

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="eventlet",
    transports=["websocket", "polling"],
    ping_timeout=60,
    ping_interval=25,
    logger=False,
    engineio_logger=False,
)

# --------------------------------------------------
# MediaPipe Setup
# --------------------------------------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# --------------------------------------------------
# Drawing Settings
# --------------------------------------------------

COLORS = {
    "Red": (0,0,255),
    "Green": (0,255,0),
    "Blue": (255,0,0),
    "Yellow": (0,255,255),
    "Purple": (255,0,255),
    "Cyan": (255,255,0),
    "White": (255,255,255),
    "Orange": (0,165,255),
}

HOLD_DURATION = 1.0

# --------------------------------------------------
# Client State
# --------------------------------------------------

def make_state():
    return {
        "canvas": None,
        "prev_point": None,
        "draw_color": COLORS["Red"],
        "color_name": "Red",
        "mirror_mode": False,
        "particle_mode": False,
        "particles": [],
        "brush_size": 12,
        "pinky_was_up": False,
        "last_gesture": "none",
        "hold_start": None,
        "hold_triggered": False,
        "last_frame_time": 0
    }

# --------------------------------------------------
# Gesture Detection
# --------------------------------------------------

def finger_states(lm):
    return (
        lm[8].y < lm[6].y,
        lm[12].y < lm[10].y,
        lm[16].y < lm[14].y,
        lm[20].y < lm[18].y
    )

def detect_gesture(lm):
    i,m,r,p = finger_states(lm)

    if not i and not m and not r and not p:
        return "fist"

    if i and not m and not r and not p:
        return "index"

    if i and m and not r and not p:
        return "two"

    if i and m and r and not p:
        return "three"

    if not i and not m and not r and p:
        return "pinky"

    if i and m and r and p:
        return "open"

    return "none"

# --------------------------------------------------
# Frame Processing
# --------------------------------------------------

def process_frame(sid, jpg_bytes):

    with state_lock:
        s = client_states.get(sid)

    if not s:
        return

    now = time.time()
    if now - s["last_frame_time"] < 0.05:
        return

    s["last_frame_time"] = now

    try:
        arr = np.frombuffer(jpg_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except:
        return

    if frame is None:
        return

    h,w = frame.shape[:2]

    if s["canvas"] is None:
        s["canvas"] = np.zeros_like(frame)

    # Resize for faster MediaPipe
    small = cv2.resize(frame,(0,0),fx=0.75,fy=0.75)

    result = hands.process(
        cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    )

    mode_text = "No Hand"

    if result.multi_hand_landmarks:

        lm = result.multi_hand_landmarks[0].landmark

        mp_draw.draw_landmarks(
            frame,
            result.multi_hand_landmarks[0],
            mp_hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style(),
        )

        gesture = detect_gesture(lm)

        tip_x = int(lm[8].x * w)
        tip_y = int(lm[8].y * h)

        # Drawing
        if gesture == "index":

            mode_text = "✏ Drawing"

            if s["prev_point"]:

                cv2.line(
                    s["canvas"],
                    s["prev_point"],
                    (tip_x,tip_y),
                    s["draw_color"],
                    s["brush_size"]
                )

            s["prev_point"] = (tip_x,tip_y)

        # Erasing
        elif gesture == "two":

            mode_text = "✌ Erasing"
            s["prev_point"] = None

            cv2.circle(
                s["canvas"],
                (tip_x,tip_y),
                40,
                (0,0,0),
                -1
            )

        # Clear
        elif gesture == "open":

            mode_text = "🖐 Clear"
            s["canvas"][:] = 0

        # Color cycle
        elif gesture == "pinky":

            if not s["pinky_was_up"]:

                names = list(COLORS.keys())
                idx = (names.index(s["color_name"])+1)%len(names)

                s["color_name"] = names[idx]
                s["draw_color"] = COLORS[names[idx]]

                s["pinky_was_up"] = True

            mode_text = f"🎨 {s['color_name']}"

        else:
            s["prev_point"] = None
            s["pinky_was_up"] = False

    else:
        s["prev_point"] = None

    # Blend drawing with frame
    output = cv2.addWeighted(frame,1,s["canvas"],1,0)

    # JPEG encode
    encode = [int(cv2.IMWRITE_JPEG_QUALITY),65]

    _,buf = cv2.imencode(".jpg",output,encode)

    b64 = base64.b64encode(buf).decode()

    socketio.emit(
        "frame",
        {
            "image": b64,
            "gesture": mode_text,
            "color": s["color_name"],
            "mirror": s["mirror_mode"],
            "particles": s["particle_mode"]
        },
        to=sid
    )

# --------------------------------------------------
# Socket Events
# --------------------------------------------------

@socketio.on("connect")
def connect():

    with state_lock:
        client_states[request.sid] = make_state()

    print("Client connected", request.sid)


@socketio.on("disconnect")
def disconnect():

    with state_lock:
        client_states.pop(request.sid,None)

    print("Client disconnected", request.sid)


@socketio.on("frame")
def receive_frame(data):

    try:
        jpg = base64.b64decode(data["image"])
        process_frame(request.sid, jpg)
    except Exception as e:
        print("Frame error", e)


# --------------------------------------------------
# Web Page
# --------------------------------------------------

HTML = """<h1>Air Writer Server Running</h1>"""

@app.route("/")
def index():
    return render_template_string(HTML)

# --------------------------------------------------
# Server Start
# --------------------------------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT",5000))

    socketio.run(
        app,
        host="0.0.0.0",
        port=port,
        debug=False
    )