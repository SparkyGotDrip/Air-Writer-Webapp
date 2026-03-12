"""
Air Writer — Web App Backend
Requirements: pip install flask flask-socketio opencv-python mediapipe numpy eventlet
Run: python app.py
Then open: http://localhost:5000
"""

import cv2
import mediapipe as mp
import numpy as np
import random
import math
import time
import base64
import threading

from flask import Flask, render_template_string, send_from_directory
from flask_socketio import SocketIO, emit

# ── Flask setup ───────────────────────────────────────────────────
app    = Flask(__name__)
app.config["SECRET_KEY"] = "airwriter-secret"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ── MediaPipe ─────────────────────────────────────────────────────
mp_hands       = mp.solutions.hands
mp_draw        = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles
hands          = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# ── State (shared across frames) ──────────────────────────────────
state = {
    "canvas":        None,
    "prev_point":    None,
    "mirror_mode":   False,
    "particle_mode": False,
    "particles":     [],
    "draw_color":    (0, 0, 255),
    "color_name":    "Red",
    "hold_gesture":  None,
    "hold_start":    None,
    "hold_triggered":False,
    "last_gesture":  "none",
    "pinky_was_up":  False,
    "brush_size":    12,
    "running":       False,
    "cap":           None,
}

COLORS = {
    "Red":    (0,   0,   255),
    "Green":  (0,   255, 0  ),
    "Blue":   (255, 0,   0  ),
    "Yellow": (0,   255, 255),
    "Purple": (255, 0,   255),
    "Cyan":   (255, 255, 0  ),
    "White":  (255, 255, 255),
    "Orange": (0,   165, 255),
}
HOLD_DURATION = 1.0

# ── Gesture detection ─────────────────────────────────────────────
def finger_states(lm):
    return (lm[8].y < lm[6].y, lm[12].y < lm[10].y,
            lm[16].y < lm[14].y, lm[20].y < lm[18].y)

def detect_gesture(lm):
    i, m, r, p  = finger_states(lm)
    thumb_tucked = (lm[4].x - lm[5].x) < 0
    if not i and not m and not r and not p: return "fist"
    if i and not m and not r and not p:     return "index"
    if i and m and not r and not p:         return "two_fingers"
    if i and m and r and not p:             return "three_fingers"
    if not i and not m and not r and p:     return "pinky"
    if i and m and r and p:
        return "four_fingers" if thumb_tucked else "open_palm"
    return "none"

PALM_PTS = [0, 1, 5, 9, 13, 17]
def palm_center(lm, w, h):
    cx = int(sum(lm[i].x for i in PALM_PTS) / len(PALM_PTS) * w)
    cy = int(sum(lm[i].y for i in PALM_PTS) / len(PALM_PTS) * h)
    return cx, cy

def update_hold(gesture):
    s = state
    if gesture != s["last_gesture"]:
        s["last_gesture"]    = gesture
        s["hold_triggered"]  = False
        if gesture in ("three_fingers", "four_fingers"):
            s["hold_gesture"] = gesture
            s["hold_start"]   = time.time()
        else:
            s["hold_gesture"] = None
            s["hold_start"]   = None

    if gesture not in ("three_fingers", "four_fingers") or s["hold_start"] is None:
        return 0.0, False

    elapsed  = time.time() - s["hold_start"]
    progress = min(elapsed / HOLD_DURATION, 1.0)
    if progress >= 1.0 and not s["hold_triggered"]:
        s["hold_triggered"] = True
        return 1.0, True
    return progress, False

# ── Drawing helpers ───────────────────────────────────────────────
def draw_line(canvas, p1, p2, color, thickness, w):
    cv2.line(canvas, p1, p2, color, thickness)
    if state["mirror_mode"]:
        cv2.line(canvas, (w - p1[0], p1[1]), (w - p2[0], p2[1]), color, thickness)

def draw_hold_arc(frame, cx, cy, progress, color):
    radius = 68
    angle  = int(360 * progress)
    prev   = None
    for deg in range(-90, -90 + angle, 3):
        rad = math.radians(deg)
        pt  = (int(cx + radius * math.cos(rad)), int(cy + radius * math.sin(rad)))
        if prev:
            cv2.line(frame, prev, pt, color, 3)
        prev = pt

def draw_palm_ring(frame, cx, cy, gesture, color, hold_progress=0.0):
    styles = {
        "index":         [(52, color, 2), (44, color, 1)],
        "two_fingers":   [(52, (255, 255, 0), 2)],
        "three_fingers": [(52, (255, 0, 255), 2)],
        "four_fingers":  [(52, (0, 255, 255), 2)],
        "open_palm":     [(52, (0, 140, 255), 2)],
        "fist":          [(52, (80, 80, 80), 1)],
        "pinky":         [(52, color, 2), (60, color, 1)],
    }
    for radius, col, thickness in styles.get(gesture, []):
        cv2.circle(frame, (cx, cy), radius, col, thickness)
    if hold_progress > 0.0:
        arc_color = (255, 0, 255) if gesture == "three_fingers" else (0, 255, 255)
        draw_hold_arc(frame, cx, cy, hold_progress, arc_color)

def spawn_particles(x, y):
    for _ in range(6):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 4)
        hue   = random.randint(0, 179)
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]),
                             cv2.COLOR_HSV2BGR)[0][0].tolist()
        state["particles"].append([
            float(x), float(y), random.randint(4, 10), color, 255.0,
            math.cos(angle) * speed, math.sin(angle) * speed,
        ])

def tick_particles(frame):
    dead = []
    for idx, p in enumerate(state["particles"]):
        p[0] += p[5]; p[1] += p[6]
        p[2] = max(0.0, p[2] - 0.2)
        p[4] = max(0.0, p[4] - 8.0)
        if p[4] <= 0 or p[2] <= 0:
            dead.append(idx); continue
        alpha   = p[4] / 255.0
        overlay = frame.copy()
        cv2.circle(overlay, (int(p[0]), int(p[1])), int(p[2]), p[3], -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    for idx in reversed(dead):
        state["particles"].pop(idx)

# ── Video loop (runs in background thread) ────────────────────────
def video_loop():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    state["cap"] = cap

    last_time = time.time()

    while state["running"]:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.03)
            continue

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        if state["canvas"] is None or state["canvas"].shape != frame.shape:
            state["canvas"] = np.zeros_like(frame)

        result  = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        gesture = "none"
        mode_text = "No Hand"

        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0].landmark
            mp_draw.draw_landmarks(
                frame, result.multi_hand_landmarks[0],
                mp_hands.HAND_CONNECTIONS,
                mp_draw_styles.get_default_hand_landmarks_style(),
                mp_draw_styles.get_default_hand_connections_style(),
            )

            gesture = detect_gesture(lm)
            tip_x   = int(lm[8].x * w)
            tip_y   = int(lm[8].y * h)
            cx, cy  = palm_center(lm, w, h)
            hold_progress, just_triggered = update_hold(gesture)
            draw_palm_ring(frame, cx, cy, gesture, state["draw_color"], hold_progress)

            if gesture == "fist":
                mode_text = "✊ Idle"
                state["prev_point"] = None; state["pinky_was_up"] = False

            elif gesture == "index":
                mode_text = "✏️ Drawing"
                brush = state["brush_size"]
                cv2.circle(frame, (tip_x, tip_y), brush // 2, state["draw_color"], -1)
                cv2.circle(frame, (tip_x, tip_y), brush // 2, (255, 255, 255), 1)
                if state["prev_point"]:
                    draw_line(state["canvas"], state["prev_point"],
                              (tip_x, tip_y), state["draw_color"], brush, w)
                if state["particle_mode"]:
                    spawn_particles(tip_x, tip_y)
                state["prev_point"] = (tip_x, tip_y); state["pinky_was_up"] = False

            elif gesture == "two_fingers":
                mode_text = "✌️ Erasing"
                state["prev_point"] = None; state["pinky_was_up"] = False
                ix, iy = int(lm[8].x * w),  int(lm[8].y * h)
                mx, my = int(lm[12].x * w), int(lm[12].y * h)
                ex = (ix + mx) // 2; ey = (iy + my) // 2
                spread = int(math.hypot(mx - ix, my - iy) // 2)
                radius = max(20, min(spread, 80))
                cv2.circle(state["canvas"], (ex, ey), radius, (0, 0, 0), -1)
                cv2.circle(frame, (ex, ey), radius, (255, 255, 0), 2)

            elif gesture == "three_fingers":
                state["prev_point"] = None; state["pinky_was_up"] = False
                if just_triggered:
                    state["particle_mode"] = not state["particle_mode"]
                    mode_text = "✨ Particles toggled!"
                else:
                    mode_text = "⏳ Hold for Particles..."

            elif gesture == "four_fingers":
                state["prev_point"] = None; state["pinky_was_up"] = False
                if just_triggered:
                    state["mirror_mode"] = not state["mirror_mode"]
                    mode_text = "🪞 Mirror toggled!"
                else:
                    mode_text = "⏳ Hold for Mirror..."

            elif gesture == "open_palm":
                mode_text = "🖐️ Cleared!"
                state["prev_point"] = None; state["pinky_was_up"] = False
                state["canvas"] = np.zeros_like(frame)
                state["particles"].clear()

            elif gesture == "pinky":
                state["prev_point"] = None
                if not state["pinky_was_up"]:
                    names = list(COLORS.keys())
                    idx   = (names.index(state["color_name"]) + 1) % len(names)
                    state["color_name"]  = names[idx]
                    state["draw_color"]  = COLORS[names[idx]]
                    state["pinky_was_up"] = True
                mode_text = f"🎨 Color: {state['color_name']}"

            else:
                mode_text = "No Hand"
                state["prev_point"] = None; state["pinky_was_up"] = False
        else:
            state["prev_point"] = None; state["pinky_was_up"] = False
            state["hold_gesture"] = None; state["hold_start"] = None
            state["hold_triggered"] = False; state["last_gesture"] = "none"

        output = cv2.addWeighted(frame, 1.0, state["canvas"], 1.0, 0)
        if state["particle_mode"] or state["particles"]:
            tick_particles(output)
        if state["mirror_mode"]:
            cv2.line(output, (w // 2, 0), (w // 2, h), (0, 255, 255), 1)

        # FPS
        now = time.time()
        fps = 1.0 / max(now - last_time, 1e-9)
        last_time = now

        # Encode and send via WebSocket
        _, buf = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, 70])
        b64 = base64.b64encode(buf).decode("utf-8")

        socketio.emit("frame", {
            "image": b64,
            "gesture": mode_text,
            "fps": f"{fps:.0f}",
            "mirror": state["mirror_mode"],
            "particles": state["particle_mode"],
            "color": state["color_name"],
        })
        time.sleep(0.033)

    cap.release()

# ── Socket events ─────────────────────────────────────────────────
@socketio.on("connect")
def on_connect():
    print("Client connected")
    if not state["running"]:
        state["running"] = True
        t = threading.Thread(target=video_loop, daemon=True)
        t.start()

@socketio.on("disconnect")
def on_disconnect():
    print("Client disconnected")

@socketio.on("set_color")
def on_set_color(data):
    name = data.get("color", "Red")
    if name in COLORS:
        state["draw_color"] = COLORS[name]
        state["color_name"] = name

@socketio.on("set_brush")
def on_set_brush(data):
    state["brush_size"] = max(4, min(int(data.get("size", 12)), 40))

@socketio.on("toggle_mirror")
def on_toggle_mirror():
    state["mirror_mode"] = not state["mirror_mode"]

@socketio.on("toggle_particles")
def on_toggle_particles():
    state["particle_mode"] = not state["particle_mode"]

@socketio.on("clear_canvas")
def on_clear():
    if state["canvas"] is not None:
        state["canvas"][:] = 0
    state["particles"].clear()

# ── HTML page (served inline) ─────────────────────────────────────
HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>✏️ Air Writer</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.6.1/socket.io.min.js"></script>
<style>
  :root {
    --bg: #0F0F1A; --sidebar: #1A1A2E; --card: #252540;
    --accent: #5B5BFF; --text: #E0E0FF; --muted: #6060AA;
    --green: #3BFF6A; --purple: #CC3BFF;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { display: flex; height: 100vh; background: var(--bg); color: var(--text);
         font-family: 'Segoe UI', sans-serif; overflow: hidden; }

  /* Sidebar */
  #sidebar {
    width: 230px; min-width: 230px; background: var(--sidebar);
    display: flex; flex-direction: column; padding: 20px 14px;
    gap: 12px; border-right: 1px solid #2a2a4a; overflow-y: auto;
  }
  #sidebar h1 { font-size: 18px; font-weight: 700; }
  #sidebar .sub { font-size: 11px; color: var(--muted); margin-top: -8px; }
  .sep { border: none; border-top: 1px solid #2a2a4a; }

  .section-label { font-size: 10px; color: var(--muted); font-weight: 700;
                   letter-spacing: 0.08em; text-transform: uppercase; }

  /* Gesture badge */
  #gesture-badge {
    background: var(--card); border-radius: 8px; padding: 10px 12px;
    font-size: 14px; font-weight: 700; text-align: center; min-height: 42px;
    display: flex; align-items: center; justify-content: center; gap: 6px;
  }
  #fps { font-size: 10px; color: var(--muted); text-align: center; margin-top: -6px; }

  /* Color grid */
  .color-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; }
  .color-btn {
    width: 100%; aspect-ratio: 1; border-radius: 6px; border: 2px solid transparent;
    cursor: pointer; transition: transform .15s, border-color .15s;
  }
  .color-btn:hover { transform: scale(1.15); }
  .color-btn.active { border-color: white; }

  /* Color pill */
  #color-pill {
    text-align: center; padding: 6px; border-radius: 6px;
    font-weight: 700; font-size: 12px; transition: background .2s;
  }

  /* Brush slider */
  input[type=range] { width: 100%; accent-color: var(--accent); cursor: pointer; }

  /* Toggle buttons */
  .toggle-btn {
    width: 100%; padding: 8px; border-radius: 8px; border: none;
    background: var(--card); color: var(--muted); font-size: 13px;
    cursor: pointer; transition: background .2s, color .2s; font-family: inherit;
  }
  .toggle-btn.on { background: #1A3A1A; color: var(--green); }
  .toggle-btn.on.purple { background: #2A1A3A; color: var(--purple); }

  /* Danger button */
  .danger-btn {
    width: 100%; padding: 9px; border-radius: 8px; border: none;
    background: #3B1A1A; color: #FF6666; font-size: 13px; font-weight: 700;
    cursor: pointer; font-family: inherit;
  }
  .danger-btn:hover { background: #5A2A2A; }

  /* Gesture guide */
  .guide-row { display: flex; justify-content: space-between; font-size: 11px; }
  .guide-row .g-name { color: #AAAACC; }
  .guide-row .g-action { color: var(--muted); }

  /* Main video area */
  #main {
    flex: 1; display: flex; flex-direction: column; align-items: center;
    justify-content: center; padding: 12px; gap: 10px; overflow: hidden;
  }
  #video-container {
    position: relative; width: 100%; flex: 1;
    display: flex; align-items: center; justify-content: center;
  }
  #video-feed {
    max-width: 100%; max-height: 100%; border-radius: 12px;
    box-shadow: 0 0 40px #5B5BFF33; object-fit: contain;
  }
  #no-cam {
    color: var(--muted); font-size: 18px; text-align: center;
    display: flex; flex-direction: column; gap: 10px; align-items: center;
  }
  #no-cam span { font-size: 48px; }
</style>
</head>
<body>

<div id="sidebar">
  <h1>✏️ Air Writer</h1>
  <p class="sub">Draw with your hand!</p>
  <hr class="sep">

  <span class="section-label">Gesture</span>
  <div id="gesture-badge">👋 Connecting...</div>
  <div id="fps">FPS: --</div>

  <hr class="sep">
  <span class="section-label">Color</span>
  <div class="color-grid" id="color-grid"></div>
  <div id="color-pill">Red</div>

  <hr class="sep">
  <span class="section-label">Brush Size</span>
  <input type="range" id="brush-slider" min="4" max="40" value="12">

  <hr class="sep">
  <span class="section-label">Modes</span>
  <button class="toggle-btn" id="mirror-btn" onclick="toggleMirror()">🪞 Mirror: OFF</button>
  <button class="toggle-btn" id="particle-btn" onclick="toggleParticles()">✨ Particles: OFF</button>

  <button class="danger-btn" onclick="clearCanvas()">🗑️ Clear Canvas</button>

  <hr class="sep">
  <span class="section-label">Gesture Guide</span>
  <div class="guide-row"><span class="g-name">☝️ Index</span><span class="g-action">Draw</span></div>
  <div class="guide-row"><span class="g-name">✌️ Two fingers</span><span class="g-action">Erase</span></div>
  <div class="guide-row"><span class="g-name">🤟 Pinky</span><span class="g-action">Cycle color</span></div>
  <div class="guide-row"><span class="g-name">🖐️ Open palm</span><span class="g-action">Clear</span></div>
  <div class="guide-row"><span class="g-name">🤘 Three (hold)</span><span class="g-action">Particles</span></div>
  <div class="guide-row"><span class="g-name">🖖 Four (hold)</span><span class="g-action">Mirror</span></div>
</div>

<div id="main">
  <div id="video-container">
    <div id="no-cam"><span>📷</span>Waiting for camera stream...</div>
    <img id="video-feed" style="display:none;" alt="Live Feed">
  </div>
</div>

<script>
const COLORS = {
  Red:    "#FF3B3B", Green: "#3BFF6A", Blue:   "#3B7FFF",
  Yellow: "#FFE83B", Purple:"#CC3BFF", Cyan:   "#3BFFFF",
  White:  "#FFFFFF", Orange:"#FFA53B"
};

const socket = io();
let mirrorOn    = false;
let particleOn  = false;
let activeColor = "Red";

// Build color grid
const grid = document.getElementById("color-grid");
Object.entries(COLORS).forEach(([name, hex]) => {
  const btn = document.createElement("button");
  btn.className = "color-btn" + (name === "Red" ? " active" : "");
  btn.style.background = hex;
  btn.title = name;
  btn.onclick = () => setColor(name);
  btn.id = "color-" + name;
  grid.appendChild(btn);
});

function setColor(name) {
  socket.emit("set_color", { color: name });
  document.querySelectorAll(".color-btn").forEach(b => b.classList.remove("active"));
  document.getElementById("color-" + name).classList.add("active");
  const pill = document.getElementById("color-pill");
  const hex  = COLORS[name];
  pill.style.background = hex;
  pill.style.color = ["White","Yellow","Cyan"].includes(name) ? "#000" : "#fff";
  pill.textContent = name;
  activeColor = name;
}

document.getElementById("brush-slider").addEventListener("input", e => {
  socket.emit("set_brush", { size: parseInt(e.target.value) });
});

function toggleMirror() {
  socket.emit("toggle_mirror");
  mirrorOn = !mirrorOn;
  const btn = document.getElementById("mirror-btn");
  btn.textContent = mirrorOn ? "🪞 Mirror: ON" : "🪞 Mirror: OFF";
  btn.className = "toggle-btn" + (mirrorOn ? " on" : "");
}

function toggleParticles() {
  socket.emit("toggle_particles");
  particleOn = !particleOn;
  const btn = document.getElementById("particle-btn");
  btn.textContent = particleOn ? "✨ Particles: ON" : "✨ Particles: OFF";
  btn.className = "toggle-btn" + (particleOn ? " on purple" : "");
}

function clearCanvas() {
  socket.emit("clear_canvas");
}

// Receive frames
socket.on("frame", data => {
  const img   = document.getElementById("video-feed");
  const noCam = document.getElementById("no-cam");
  img.src = "data:image/jpeg;base64," + data.image;
  if (img.style.display === "none") {
    img.style.display = "block";
    noCam.style.display = "none";
  }
  document.getElementById("gesture-badge").textContent = data.gesture || "No Hand";
  document.getElementById("fps").textContent = "FPS: " + (data.fps || "--");

  // Sync mirror/particle from server
  if (data.mirror !== mirrorOn) {
    mirrorOn = data.mirror;
    const btn = document.getElementById("mirror-btn");
    btn.textContent = mirrorOn ? "🪞 Mirror: ON" : "🪞 Mirror: OFF";
    btn.className = "toggle-btn" + (mirrorOn ? " on" : "");
  }
  if (data.particles !== particleOn) {
    particleOn = data.particles;
    const btn = document.getElementById("particle-btn");
    btn.textContent = particleOn ? "✨ Particles: ON" : "✨ Particles: OFF";
    btn.className = "toggle-btn" + (particleOn ? " on purple" : "");
  }
  if (data.color && data.color !== activeColor) {
    setColor(data.color);
  }
});

socket.on("connect", () => {
  document.getElementById("gesture-badge").textContent = "🤝 Connected!";
});
socket.on("disconnect", () => {
  document.getElementById("gesture-badge").textContent = "❌ Disconnected";
});
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True)
