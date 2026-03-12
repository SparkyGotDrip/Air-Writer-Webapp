"""
Air Writer — Web App (Browser Camera Edition)
Browser captures webcam frames and sends to server for MediaPipe processing.
"""

import cv2
import mediapipe as mp
import numpy as np
import random
import math
import time
import base64

from flask import Flask, render_template_string
from flask_socketio import SocketIO

app = Flask(__name__)
app.config["SECRET_KEY"] = "airwriter-secret"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading",
                    max_http_buffer_size=10 * 1024 * 1024,
                    transports=["polling", "websocket"],
                    ping_timeout=60, ping_interval=25)

mp_hands       = mp.solutions.hands
mp_draw        = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles
hands          = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

client_states = {}

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

def make_state():
    return {
        "canvas": None, "prev_point": None,
        "mirror_mode": False, "particle_mode": False, "particles": [],
        "draw_color": (0, 0, 255), "color_name": "Red",
        "hold_gesture": None, "hold_start": None,
        "hold_triggered": False, "last_gesture": "none",
        "pinky_was_up": False, "brush_size": 12,
    }

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

def update_hold(s, gesture):
    if gesture != s["last_gesture"]:
        s["last_gesture"]   = gesture
        s["hold_triggered"] = False
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

def draw_line(canvas, p1, p2, color, thickness, w, mirror):
    cv2.line(canvas, p1, p2, color, thickness)
    if mirror:
        cv2.line(canvas, (w - p1[0], p1[1]), (w - p2[0], p2[1]), color, thickness)

def draw_hold_arc(frame, cx, cy, progress, color):
    radius = 68; angle = int(360 * progress); prev = None
    for deg in range(-90, -90 + angle, 3):
        rad = math.radians(deg)
        pt  = (int(cx + radius * math.cos(rad)), int(cy + radius * math.sin(rad)))
        if prev: cv2.line(frame, prev, pt, color, 3)
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

def spawn_particles(s, x, y):
    for _ in range(6):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1, 4)
        hue   = random.randint(0, 179)
        color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]),
                             cv2.COLOR_HSV2BGR)[0][0].tolist()
        s["particles"].append([float(x), float(y), random.randint(4, 10),
                                color, 255.0, math.cos(angle)*speed, math.sin(angle)*speed])

def tick_particles(s, frame):
    dead = []
    for idx, p in enumerate(s["particles"]):
        p[0] += p[5]; p[1] += p[6]
        p[2] = max(0.0, p[2] - 0.2); p[4] = max(0.0, p[4] - 8.0)
        if p[4] <= 0 or p[2] <= 0: dead.append(idx); continue
        alpha = p[4] / 255.0; overlay = frame.copy()
        cv2.circle(overlay, (int(p[0]), int(p[1])), int(p[2]), p[3], -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    for idx in reversed(dead): s["particles"].pop(idx)

def process_frame(sid, jpg_bytes):
    s = client_states.get(sid)
    if s is None: return
    arr   = np.frombuffer(jpg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None: return

    h, w  = frame.shape[:2]
    if s["canvas"] is None or s["canvas"].shape != frame.shape:
        s["canvas"] = np.zeros_like(frame)

    result    = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    mode_text = "No Hand"

    if result.multi_hand_landmarks:
        lm = result.multi_hand_landmarks[0].landmark
        mp_draw.draw_landmarks(frame, result.multi_hand_landmarks[0],
            mp_hands.HAND_CONNECTIONS,
            mp_draw_styles.get_default_hand_landmarks_style(),
            mp_draw_styles.get_default_hand_connections_style())
        gesture = detect_gesture(lm)
        tip_x   = int(lm[8].x * w); tip_y = int(lm[8].y * h)
        cx, cy  = palm_center(lm, w, h)
        hold_progress, just_triggered = update_hold(s, gesture)
        draw_palm_ring(frame, cx, cy, gesture, s["draw_color"], hold_progress)

        if gesture == "fist":
            mode_text = "✊ Idle"; s["prev_point"] = None; s["pinky_was_up"] = False
        elif gesture == "index":
            mode_text = "✏️ Drawing"; brush = s["brush_size"]
            cv2.circle(frame, (tip_x, tip_y), brush//2, s["draw_color"], -1)
            cv2.circle(frame, (tip_x, tip_y), brush//2, (255,255,255), 1)
            if s["prev_point"]:
                draw_line(s["canvas"], s["prev_point"], (tip_x, tip_y),
                          s["draw_color"], brush, w, s["mirror_mode"])
            if s["particle_mode"]: spawn_particles(s, tip_x, tip_y)
            s["prev_point"] = (tip_x, tip_y); s["pinky_was_up"] = False
        elif gesture == "two_fingers":
            mode_text = "✌️ Erasing"; s["prev_point"] = None; s["pinky_was_up"] = False
            ix,iy = int(lm[8].x*w),int(lm[8].y*h); mx,my = int(lm[12].x*w),int(lm[12].y*h)
            ex=(ix+mx)//2; ey=(iy+my)//2
            spread=int(math.hypot(mx-ix,my-iy)//2); radius=max(20,min(spread,80))
            cv2.circle(s["canvas"],(ex,ey),radius,(0,0,0),-1)
            cv2.circle(frame,(ex,ey),radius,(255,255,0),2)
        elif gesture == "three_fingers":
            s["prev_point"] = None; s["pinky_was_up"] = False
            if just_triggered: s["particle_mode"] = not s["particle_mode"]; mode_text = "✨ Particles toggled!"
            else: mode_text = "⏳ Hold for Particles..."
        elif gesture == "four_fingers":
            s["prev_point"] = None; s["pinky_was_up"] = False
            if just_triggered: s["mirror_mode"] = not s["mirror_mode"]; mode_text = "🪞 Mirror toggled!"
            else: mode_text = "⏳ Hold for Mirror..."
        elif gesture == "open_palm":
            mode_text = "🖐️ Cleared!"; s["prev_point"] = None; s["pinky_was_up"] = False
            s["canvas"] = np.zeros_like(frame); s["particles"].clear()
        elif gesture == "pinky":
            s["prev_point"] = None
            if not s["pinky_was_up"]:
                names = list(COLORS.keys())
                idx   = (names.index(s["color_name"]) + 1) % len(names)
                s["color_name"] = names[idx]; s["draw_color"] = COLORS[names[idx]]; s["pinky_was_up"] = True
            mode_text = f"🎨 Color: {s['color_name']}"
        else:
            mode_text = "No Hand"; s["prev_point"] = None; s["pinky_was_up"] = False
    else:
        s["prev_point"] = None; s["pinky_was_up"] = False
        s["hold_gesture"] = None; s["hold_start"] = None
        s["hold_triggered"] = False; s["last_gesture"] = "none"

    output = cv2.addWeighted(frame, 1.0, s["canvas"], 1.0, 0)
    if s["particle_mode"] or s["particles"]: tick_particles(s, output)
    if s["mirror_mode"]: cv2.line(output, (w//2, 0), (w//2, h), (0,255,255), 1)

    _, buf = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, 70])
    b64 = base64.b64encode(buf).decode("utf-8")
    socketio.emit("frame", {
        "image": b64, "gesture": mode_text,
        "mirror": s["mirror_mode"], "particles": s["particle_mode"], "color": s["color_name"],
    }, to=sid)

@socketio.on("connect")
def on_connect():
    from flask import request
    client_states[request.sid] = make_state()
    print(f"[+] Client connected: {request.sid}")

@socketio.on("disconnect")
def on_disconnect():
    from flask import request
    client_states.pop(request.sid, None)
    print(f"[-] Client disconnected: {request.sid}")

@socketio.on("frame")
def on_frame(data):
    from flask import request
    try:
        jpg = base64.b64decode(data["image"])
        process_frame(request.sid, jpg)
    except Exception as e:
        print(f"Frame error: {e}")

@socketio.on("set_color")
def on_set_color(data):
    from flask import request
    s = client_states.get(request.sid)
    if s:
        name = data.get("color","Red")
        if name in COLORS: s["draw_color"] = COLORS[name]; s["color_name"] = name

@socketio.on("set_brush")
def on_set_brush(data):
    from flask import request
    s = client_states.get(request.sid)
    if s: s["brush_size"] = max(4, min(int(data.get("size", 12)), 40))

@socketio.on("toggle_mirror")
def on_toggle_mirror():
    from flask import request
    s = client_states.get(request.sid)
    if s: s["mirror_mode"] = not s["mirror_mode"]

@socketio.on("toggle_particles")
def on_toggle_particles():
    from flask import request
    s = client_states.get(request.sid)
    if s: s["particle_mode"] = not s["particle_mode"]

@socketio.on("clear_canvas")
def on_clear():
    from flask import request
    s = client_states.get(request.sid)
    if s and s["canvas"] is not None: s["canvas"][:] = 0; s["particles"].clear()

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Air Writer</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>✏️</text></svg>">
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.6.1/socket.io.min.js"></script>
<style>
  :root{--bg:#0F0F1A;--sidebar:#1A1A2E;--card:#252540;--accent:#5B5BFF;--text:#E0E0FF;--muted:#6060AA;--green:#3BFF6A;--purple:#CC3BFF}
  *{box-sizing:border-box;margin:0;padding:0}
  body{display:flex;height:100vh;background:var(--bg);color:var(--text);font-family:'Segoe UI',sans-serif;overflow:hidden}
  #sidebar{width:230px;min-width:230px;background:var(--sidebar);display:flex;flex-direction:column;padding:20px 14px;gap:12px;border-right:1px solid #2a2a4a;overflow-y:auto}
  #sidebar h1{font-size:18px;font-weight:700}
  .sub{font-size:11px;color:var(--muted);margin-top:-8px}
  .sep{border:none;border-top:1px solid #2a2a4a}
  .section-label{font-size:10px;color:var(--muted);font-weight:700;letter-spacing:.08em;text-transform:uppercase}
  #gesture-badge{background:var(--card);border-radius:8px;padding:10px 12px;font-size:13px;font-weight:700;text-align:center;min-height:42px;display:flex;align-items:center;justify-content:center}
  #debug-bar{background:#111128;border-radius:6px;padding:6px 10px;font-size:10px;color:#4040AA;font-family:monospace;line-height:1.6}
  .color-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:6px}
  .color-btn{width:100%;aspect-ratio:1;border-radius:6px;border:2px solid transparent;cursor:pointer;transition:transform .15s,border-color .15s}
  .color-btn:hover{transform:scale(1.15)}
  .color-btn.active{border-color:white}
  #color-pill{text-align:center;padding:6px;border-radius:6px;font-weight:700;font-size:12px}
  input[type=range]{width:100%;accent-color:var(--accent);cursor:pointer}
  .toggle-btn{width:100%;padding:8px;border-radius:8px;border:none;background:var(--card);color:var(--muted);font-size:13px;cursor:pointer;font-family:inherit}
  .toggle-btn.on{background:#1A3A1A;color:var(--green)}
  .toggle-btn.on.purple{background:#2A1A3A;color:var(--purple)}
  .danger-btn{width:100%;padding:9px;border-radius:8px;border:none;background:#3B1A1A;color:#FF6666;font-size:13px;font-weight:700;cursor:pointer;font-family:inherit}
  .danger-btn:hover{background:#5A2A2A}
  .guide-row{display:flex;justify-content:space-between;font-size:11px}
  .guide-row .g-name{color:#AAAACC}
  .guide-row .g-action{color:var(--muted)}
  #main{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:12px;gap:10px;overflow:hidden}
  #video-container{position:relative;width:100%;flex:1;display:flex;align-items:center;justify-content:center;background:#0a0a14;border-radius:12px}
  #video-feed{max-width:100%;max-height:100%;border-radius:12px;box-shadow:0 0 40px #5B5BFF33;object-fit:contain}
  #no-cam{color:var(--muted);font-size:16px;text-align:center;display:flex;flex-direction:column;gap:12px;align-items:center}
  #no-cam span{font-size:48px}
  #start-btn{padding:12px 28px;background:var(--accent);color:white;border:none;border-radius:10px;font-size:15px;font-weight:700;cursor:pointer;font-family:inherit}
  #start-btn:hover{background:#7B7BFF}
  #local-video{display:none}
  #capture-canvas{display:none}
</style>
</head>
<body>
<div id="sidebar">
  <h1>✏️ Air Writer</h1>
  <p class="sub">Draw with your hand!</p>
  <hr class="sep">
  <span class="section-label">Status</span>
  <div id="gesture-badge">👋 Start camera</div>
  <div id="debug-bar">
    socket: <span id="d-socket">—</span><br>
    frames sent: <span id="d-sent">0</span><br>
    frames recv: <span id="d-recv">0</span>
  </div>
  <hr class="sep">
  <span class="section-label">Color</span>
  <div class="color-grid" id="color-grid"></div>
  <div id="color-pill" style="background:#FF3B3B;color:#fff;">Red</div>
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
    <div id="no-cam">
      <span>📷</span>
      <p>Click below to start your camera</p>
      <button id="start-btn" onclick="startCamera()">▶ Start Camera</button>
    </div>
    <img id="video-feed" style="display:none;" alt="Processed Feed">
  </div>
</div>

<video id="local-video" autoplay playsinline muted></video>
<canvas id="capture-canvas"></canvas>

<script>
const COLORS={Red:"#FF3B3B",Green:"#3BFF6A",Blue:"#3B7FFF",Yellow:"#FFE83B",Purple:"#CC3BFF",Cyan:"#3BFFFF",White:"#FFFFFF",Orange:"#FFA53B"};
const socket=io({transports:["polling","websocket"],upgrade:true});
let mirrorOn=false,particleOn=false,activeColor="Red";
let framesSent=0,framesRecv=0;

// Debug helpers
function dbg(key,val){document.getElementById("d-"+key).textContent=val;}
function setSocketStatus(s){dbg("socket",s);}

// Color grid
const grid=document.getElementById("color-grid");
Object.entries(COLORS).forEach(([name,hex])=>{
  const btn=document.createElement("button");
  btn.className="color-btn"+(name==="Red"?" active":"");
  btn.style.background=hex; btn.title=name;
  btn.onclick=()=>setColor(name); btn.id="color-"+name;
  grid.appendChild(btn);
});

function setColor(name){
  socket.emit("set_color",{color:name});
  document.querySelectorAll(".color-btn").forEach(b=>b.classList.remove("active"));
  document.getElementById("color-"+name).classList.add("active");
  const pill=document.getElementById("color-pill");
  pill.style.background=COLORS[name];
  pill.style.color=["White","Yellow","Cyan"].includes(name)?"#000":"#fff";
  pill.textContent=name; activeColor=name;
}
document.getElementById("brush-slider").addEventListener("input",e=>socket.emit("set_brush",{size:parseInt(e.target.value)}));
function toggleMirror(){socket.emit("toggle_mirror");mirrorOn=!mirrorOn;const b=document.getElementById("mirror-btn");b.textContent=mirrorOn?"🪞 Mirror: ON":"🪞 Mirror: OFF";b.className="toggle-btn"+(mirrorOn?" on":"");}
function toggleParticles(){socket.emit("toggle_particles");particleOn=!particleOn;const b=document.getElementById("particle-btn");b.textContent=particleOn?"✨ Particles: ON":"✨ Particles: OFF";b.className="toggle-btn"+(particleOn?" on purple":"");}
function clearCanvas(){socket.emit("clear_canvas");}

// ── Camera ────────────────────────────────────────────────────────
async function startCamera(){
  document.getElementById("start-btn").textContent="Starting...";
  document.getElementById("start-btn").disabled=true;
  try{
    const stream=await navigator.mediaDevices.getUserMedia({
      video:{width:{ideal:640},height:{ideal:480},facingMode:"user"},audio:false
    });
    const video=document.getElementById("local-video");
    video.srcObject=stream;
    await new Promise(res=>{video.onloadedmetadata=res;});
    await video.play();

    // Wait a moment for video to be truly ready
    await new Promise(res=>setTimeout(res,500));

    document.getElementById("no-cam").style.display="none";
    document.getElementById("gesture-badge").textContent="✅ Camera active";

    const canvas=document.getElementById("capture-canvas");
    const ctx=canvas.getContext("2d");

    // Use actual video dimensions
    canvas.width=video.videoWidth||640;
    canvas.height=video.videoHeight||480;

    // Send frames at 15fps
    setInterval(()=>{
      if(video.readyState<2)return;
      canvas.width=video.videoWidth; canvas.height=video.videoHeight;
      ctx.drawImage(video,0,0);
      const b64=canvas.toDataURL("image/jpeg",0.65).split(",")[1];
      socket.emit("frame",{image:b64});
      framesSent++;
      dbg("sent",framesSent);
    },67);

  }catch(err){
    document.getElementById("gesture-badge").textContent="❌ "+err.message;
    document.getElementById("start-btn").textContent="▶ Retry";
    document.getElementById("start-btn").disabled=false;
    console.error(err);
  }
}

// ── Receive processed frames ──────────────────────────────────────
socket.on("frame",data=>{
  framesRecv++;
  dbg("recv",framesRecv);
  const img=document.getElementById("video-feed");
  img.src="data:image/jpeg;base64,"+data.image;
  if(img.style.display==="none"){
    img.style.display="block";
    document.getElementById("no-cam").style.display="none";
  }
  document.getElementById("gesture-badge").textContent=data.gesture||"No Hand";
  if(data.mirror!==mirrorOn){mirrorOn=data.mirror;const b=document.getElementById("mirror-btn");b.textContent=mirrorOn?"🪞 Mirror: ON":"🪞 Mirror: OFF";b.className="toggle-btn"+(mirrorOn?" on":"");}
  if(data.particles!==particleOn){particleOn=data.particles;const b=document.getElementById("particle-btn");b.textContent=particleOn?"✨ Particles: ON":"✨ Particles: OFF";b.className="toggle-btn"+(particleOn?" on purple":"");}
  if(data.color&&data.color!==activeColor)setColor(data.color);
});

socket.on("connect",()=>{setSocketStatus("✅ connected");});
socket.on("disconnect",reason=>{setSocketStatus("❌ "+reason);document.getElementById("gesture-badge").textContent="❌ Disconnected";});
socket.on("connect_error",err=>{setSocketStatus("❌ "+err.message);});
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True)