import streamlit as st
import cv2
import time
import os
import tempfile
import numpy as np
from ultralytics import YOLO
from collections import deque
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="TrafficSense", page_icon="🚦", layout="wide", initial_sidebar_state="expanded")

# ── Session State ─────────────────────────────────────────
if 'running'         not in st.session_state: st.session_state.running         = False 
if 'total_detected'  not in st.session_state: st.session_state.total_detected  = 0
if 'peak_congestion' not in st.session_state: st.session_state.peak_congestion = 0
if 'speed_history'   not in st.session_state: st.session_state.speed_history   = []
if 'session_log'     not in st.session_state: st.session_state.session_log     = []
if 'start_time'      not in st.session_state: st.session_state.start_time      = None

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0d1b3e, #1a3a6b, #0d1b3e); }
    .main .block-container { padding-top: 1rem; }

    .title-box {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(126,184,247,0.3);
        border-left: 5px solid #7eb8f7;
        padding: 20px 30px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    .title-box h1 { color: #7eb8f7; font-size: 2.2rem; margin: 0; }
    .title-box p  { color: #aac8f0; margin: 5px 0 0 0; font-size: 1rem; }

    .metric-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(126,184,247,0.2);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        margin: 4px 0;
    }
    .metric-card h2 { color: #7eb8f7; font-size: 2rem; margin: 0; font-weight: 500; }
    .metric-card p  { color: #aac8f0; margin: 4px 0 0 0; font-size: 0.85rem; }

    .green-signal {
        background: rgba(0,255,136,0.08);
        border: 2px solid #00ff88;
        border-radius: 10px;
        padding: 12px;
        text-align: center;
        color: #00ff88;
        font-size: 1rem;
        font-weight: 500;
        margin: 4px 0;
    }
    .red-signal {
        background: rgba(255,68,68,0.08);
        border: 2px solid #ff4444;
        border-radius: 10px;
        padding: 12px;
        text-align: center;
        color: #ff4444;
        font-size: 1rem;
        font-weight: 500;
        margin: 4px 0;
    }
    .paused-signal {
        background: rgba(255,200,0,0.08);
        border: 2px solid #ffc800;
        border-radius: 10px;
        padding: 12px;
        text-align: center;
        color: #ffc800;
        font-size: 1rem;
        font-weight: 500;
        margin: 4px 0;
    }
    .congestion-low    { background: rgba(0,255,136,0.1);  border: 1px solid #00ff88; border-radius: 8px; padding: 10px; text-align: center; color: #00ff88;  font-weight: 500; margin: 4px 0; }
    .congestion-medium { background: rgba(255,200,0,0.1);  border: 1px solid #ffc800; border-radius: 8px; padding: 10px; text-align: center; color: #ffc800;  font-weight: 500; margin: 4px 0; }
    .congestion-high   { background: rgba(255,68,68,0.1);  border: 1px solid #ff4444; border-radius: 8px; padding: 10px; text-align: center; color: #ff4444;  font-weight: 500; margin: 4px 0; }

    .stat-box {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(126,184,247,0.2);
        border-radius: 10px;
        padding: 12px;
        margin: 4px 0;
    }
    .stat-box h3 { color: #7eb8f7; font-size: 1rem; margin: 0 0 8px 0; }
    .stat-box p  { color: #aac8f0; font-size: 0.9rem; margin: 3px 0; }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1628, #0d1b3e);
        border-right: 1px solid rgba(126,184,247,0.2);
    }
    div[data-testid="stSidebar"] * { color: #aac8f0 !important; }
    div[data-testid="stSidebar"] h1,
    div[data-testid="stSidebar"] h2,
    div[data-testid="stSidebar"] h3 { color: #7eb8f7 !important; }

    .stButton > button {
        background: rgba(126,184,247,0.15) !important;
        color: #7eb8f7 !important;
        border: 1px solid rgba(126,184,247,0.4) !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        width: 100% !important;
        padding: 10px !important;
    }
    .stButton > button:hover {
        background: rgba(126,184,247,0.3) !important;
        border-color: #7eb8f7 !important;
    }
    .stMetric { background: rgba(255,255,255,0.04); border-radius: 10px; padding: 10px; border: 1px solid rgba(126,184,247,0.15); }
    .stMetric label { color: #aac8f0 !important; }
    .stTabs [data-baseweb="tab"] { color: #aac8f0 !important; }
    .stTabs [aria-selected="true"] { color: #7eb8f7 !important; border-bottom: 2px solid #7eb8f7 !important; }
</style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("yolov8m.pt")

model = load_model()
vehicle_classes = {2: "Car", 3: "Motorcycle", 5: "Heavy Vehicle", 7: "Heavy Vehicle"}

# ── Helper Functions ──────────────────────────────────────
def get_density(count):
    if count <= 5:    return "Low", 15
    elif count <= 15: return "Medium", 30
    else:             return "High", 45

def get_congestion(total):
    if total <= 5:    return "congestion-low",    "🟢 Low Congestion"
    elif total <= 15: return "congestion-medium", "🟡 Moderate Congestion"
    else:             return "congestion-high",   "🔴 High Congestion"

def estimate_speed(box_width, box_height, frame_width):
    ratio = (box_width * box_height) / max(frame_width * frame_width * 0.5, 1)
    return max(5, min(80, int(60 - ratio * 200)))

def apply_heatmap(frame, detections):
    heat = np.zeros(frame.shape[:2], dtype=np.float32)
    for (x1,y1,x2,y2) in detections:
        cv2.circle(heat, ((x1+x2)//2,(y1+y2)//2), 60, 1.0, -1)
    heat = cv2.GaussianBlur(heat, (61,61), 0)
    heat = np.clip(heat*3, 0, 1)
    heatmap = cv2.applyColorMap((heat*255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 0.65, heatmap, 0.35, 0)

def process_frame(frame, show_heatmap=False):
    h, w = frame.shape[:2]
    results = model(frame, verbose=False)
    total_count = 0
    breakdown = {"Car":0,"Motorcycle":0,"Heavy Vehicle":0}
    detections = []
    speeds = []
    for result in results:
        for box in result.boxes:
            cls = int(box.cls)
            if cls in vehicle_classes and float(box.conf[0]) > 0.45:
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                detections.append((x1,y1,x2,y2))
                spd = estimate_speed(x2-x1, y2-y1, w)
                speeds.append(spd)
                total_count += 1
                breakdown[vehicle_classes[cls]] += 1
                color = (126, 184, 247)
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                cv2.putText(frame,f"{vehicle_classes[cls]} {spd}km/h",(x1,y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX,0.4,color,1)
    if show_heatmap and detections:
        frame = apply_heatmap(frame, detections)
    avg_speed = int(np.mean(speeds)) if speeds else 0
    return frame, total_count, breakdown, avg_speed

def draw_minimap(total, density, signal_active):
    fig = go.Figure()

    # Road background
    fig.add_shape(type="rect", x0=0, y0=0, x1=10, y1=10, fillcolor="#1a1a2e", line=dict(color="#7eb8f7", width=1))

    # Road markings
    fig.add_shape(type="rect", x0=4, y0=0, x1=6, y1=10, fillcolor="#2a2a3e", line_color="rgba(0,0,0,0)")
    fig.add_shape(type="rect", x0=0, y0=4, x1=10, y1=6, fillcolor="#2a2a3e", line_color="rgba(0,0,0,0)")

    # Intersection box
    fig.add_shape(type="rect", x0=4, y0=4, x1=6, y1=6, fillcolor="#333355", line=dict(color="#7eb8f7", width=1))

    # Congestion color
    if density == "Low":     ccolor = "#00ff88"
    elif density == "Medium": ccolor = "#ffc800"
    else:                     ccolor = "#ff4444"

    # Vehicles dots
    np.random.seed(total)
    for _ in range(min(total, 15)):
        x = np.random.uniform(4.2, 5.8)
        y = np.random.uniform(0.5, 3.5)
        fig.add_shape(type="circle", x0=x-0.15, y0=y-0.15, x1=x+0.15, y1=y+0.15,
                     fillcolor=ccolor, line_color=ccolor)

    # Signal light
    sig_color = "#00ff88" if signal_active else "#ff4444"
    fig.add_shape(type="circle", x0=5.8, y0=3.5, x1=6.3, y1=4.0,
                 fillcolor=sig_color, line_color=sig_color)

    fig.add_annotation(x=5, y=5, text="🚦", font=dict(size=16), showarrow=False)
    fig.add_annotation(x=5, y=9, text="↓ Approach", font=dict(size=9, color="#aac8f0"), showarrow=False)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=200, margin=dict(l=0,r=0,t=0,b=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0,10]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0,10]),
    )
    return fig

# ── SIDEBAR ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚦 TrafficSense")
    st.markdown("*Real-Time Vehicle Detection*")
    st.markdown("---")

    st.markdown("### 📹 Video Source")
    source = st.radio("", ["Upload Video", "Sample Videos"], label_visibility="collapsed")
    video_file = None
    if source == "Upload Video":
        video_file = st.file_uploader("Upload traffic video", type=["mp4","avi","mov"])
    else:
        videos = [f for f in os.listdir(".") if f.endswith((".mp4",".avi",".mov",".mkv"))]
        if videos:
            video_file = st.selectbox("Select Video", videos)
        else:
            st.warning("No videos found in folder!")

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    frame_skip   = st.slider("Frame Skip", 1, 5, 2)
    show_heatmap = st.toggle("Heatmap Overlay", value=False)

    st.markdown("---")
    st.markdown("### 🎮 Controls")
    start_btn = st.button("▶  Start Detection", use_container_width=True)
    stop_btn  = st.button("⏹  Stop",            use_container_width=True)
    reset_btn = st.button("🔄  Reset Stats",     use_container_width=True)

    if start_btn:
        st.session_state.running = True
        st.session_state.paused = False


    if stop_btn:
        st.session_state.running = False
        st.session_state.paused = False

    st.markdown("---")
    

# ── RESET ─────────────────────────────────────────────────
if reset_btn:
    st.session_state.total_detected  = 0
    st.session_state.peak_congestion = 0
    st.session_state.speed_history   = []
    st.session_state.session_log     = []
    st.session_state.start_time      = None
    st.toast("Stats reset!", icon="🔄")

# ── MAIN PAGE ─────────────────────────────────────────────
st.markdown("""
<div class="title-box">
    <h1>🚦 TrafficSense</h1>
    <p>Real-Time Vehicle Detection & Adaptive Signal Control System — DY Patil RAIT, Mumbai</p>
</div>
""", unsafe_allow_html=True)

# ── TOP STATS ─────────────────────────────────────────────
s1,s2,s3,s4,s5 = st.columns(5)
total_ph  = s1.empty()
fps_ph    = s2.empty()
speed_ph  = s3.empty()
cong_ph   = s4.empty()
timer_ph  = s5.empty()

total_ph.markdown('<div class="metric-card"><h2>0</h2><p>Total Vehicles</p></div>',   unsafe_allow_html=True)
fps_ph.markdown('<div class="metric-card"><h2>--</h2><p>FPS</p></div>',               unsafe_allow_html=True)
speed_ph.markdown('<div class="metric-card"><h2>--</h2><p>Avg Speed km/h</p></div>',  unsafe_allow_html=True)
cong_ph.markdown('<div class="congestion-low">🟢 Low Congestion</div>',               unsafe_allow_html=True)
timer_ph.markdown('<div class="metric-card"><h2>--</h2><p>Signal Timer</p></div>',    unsafe_allow_html=True)

st.markdown("---")

# ── TABS ──────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📷 Live Detection", "📊 Analytics", "📋 Session Stats"])

with tab1:
    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown("### 📷 Live Detection Feed")
        video_ph = st.empty()
        video_ph.markdown("""
        <div style='background:rgba(255,255,255,0.04);border:1px solid rgba(126,184,247,0.2);
        border-radius:12px;height:420px;display:flex;align-items:center;justify-content:center;'>
            <p style='color:#aac8f0;font-size:1.1rem'>Select a video and click Start Detection</p>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("### 🚦 Signal Status")
        sig_ph   = st.empty()
        sig_ph.markdown('<div class="metric-card"><p style="color:#aac8f0">Waiting...</p></div>', unsafe_allow_html=True)

        st.markdown("### 🗺️ Intersection Map")
        map_ph = st.empty()
        map_ph.plotly_chart(draw_minimap(0, "Low", True), use_container_width=True, key="minimap_init")

        st.markdown("### 📋 Vehicle Breakdown")
        break_ph = st.empty()
        break_ph.markdown('<div class="metric-card"><p style="color:#aac8f0">Waiting for detection...</p></div>', unsafe_allow_html=True)

with tab2:
    st.markdown("### 📊 Live Vehicle Count Chart")
    chart_ph = st.empty()
    st.markdown("### ⚡ Speed Distribution")
    speed_chart_ph = st.empty()

with tab3:
    st.markdown("### 📋 Session Statistics")
    session_ph = st.empty()
    session_ph.markdown("""
    <div class="stat-box">
        <h3>No session data yet</h3>
        <p>Start detection to begin recording session statistics.</p>
    </div>""", unsafe_allow_html=True)

# ── Chart History ─────────────────────────────────────────
count_history = deque(maxlen=50)
time_labels   = deque(maxlen=50)

def update_charts(speeds_all, frame_count=0):
    # Vehicle count chart
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        y=list(count_history), name="Vehicle Count",
        line=dict(color="#7eb8f7", width=2),
        fill="tozeroy", fillcolor="rgba(126,184,247,0.1)"
    ))
    fig1.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#aac8f0"), height=200,
        margin=dict(l=0,r=0,t=10,b=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(126,184,247,0.1)", zeroline=False),
    )
    chart_ph.plotly_chart(fig1, use_container_width=True)

    # Speed distribution
    if speeds_all:
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=speeds_all[-100:], nbinsx=15,
            marker_color="rgba(126,184,247,0.7)",
            marker_line=dict(color="#7eb8f7", width=1)
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#aac8f0"), height=180,
            margin=dict(l=0,r=0,t=10,b=0),
            xaxis=dict(title="Speed (km/h)", showgrid=False, color="#aac8f0"),
            yaxis=dict(title="Count", showgrid=True, gridcolor="rgba(126,184,247,0.1)"),
        )
        speed_chart_ph.plotly_chart(fig2, use_container_width=True)

def update_session_stats():
    if not st.session_state.session_log:
        return
    duration = int(time.time() - st.session_state.start_time) if st.session_state.start_time else 0
    avg_spd  = int(np.mean(st.session_state.speed_history)) if st.session_state.speed_history else 0
    session_ph.markdown(f"""
    <div class="stat-box">
        <h3>📊 Session Summary</h3>
        <p>⏱ Duration: {duration} seconds</p>
        <p>🚗 Total Vehicles Detected: {st.session_state.total_detected}</p>
        <p>🔴 Peak Congestion: {st.session_state.peak_congestion} vehicles</p>
        <p>⚡ Average Speed: {avg_spd} km/h</p>
        <p>📅 Started: {datetime.fromtimestamp(st.session_state.start_time).strftime('%H:%M:%S') if st.session_state.start_time else '--'}</p>
    </div>""", unsafe_allow_html=True)

update_charts([])
update_session_stats()

# ── DETECTION LOOP ────────────────────────────────────────
if st.session_state.running and video_file is not None:
    if isinstance(video_file, str):
        video_path = video_file
    else:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_file.read())
        video_path = tfile.name

    cap         = cv2.VideoCapture(video_path)
    active_lane = 0
    last_switch = time.time()
    frame_count = 0
    fps_time    = time.time()
    fps         = 0
    all_speeds  = []

    st.session_state.paused     = False
    st.session_state.start_time = time.time()
    st.toast("Detection started! 🚦", icon="🚦")

    while cap.isOpened():

    

        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        # FPS
        now      = time.time()
        fps      = round(1.0 / max(now - fps_time, 0.001))
        fps_time = now

        frame = cv2.resize(frame, (800, 450))
        frame, total, breakdown, avg_spd = process_frame(frame, show_heatmap)

        # Session tracking
        st.session_state.total_detected  += total
        st.session_state.peak_congestion  = max(st.session_state.peak_congestion, total)
        if avg_spd > 0:
            st.session_state.speed_history.append(avg_spd)
            all_speeds.append(avg_spd)

        density, sig_time = get_density(total)
        cong_class, cong_text = get_congestion(total)

        # Round robin
        if time.time() - last_switch >= sig_time:
            active_lane = 1 - active_lane
            last_switch = time.time()

        signal_active = active_lane == 0

        # Draw signal on frame
        sig_label = "GREEN" if signal_active else "RED"
        sig_color = (0,255,136) if signal_active else (68,68,255)
        cv2.putText(frame, f"Signal: {sig_label}", (10,35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, sig_color, 2)
        cv2.putText(frame, f"Vehicles: {total}",   (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (126,184,247), 2)
        cv2.putText(frame, f"Density: {density}",  (10,100),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (126,184,247), 2)

        # Update video
        video_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        # Update top stats
        active_time = max(0, sig_time - int(time.time() - last_switch))
        total_ph.markdown(f'<div class="metric-card"><h2>{total}</h2><p>Total Vehicles</p></div>',        unsafe_allow_html=True)
        fps_ph.markdown(f'<div class="metric-card"><h2>{min(fps,60)}</h2><p>FPS</p></div>',              unsafe_allow_html=True)
        speed_ph.markdown(f'<div class="metric-card"><h2>{avg_spd}</h2><p>Avg Speed km/h</p></div>',     unsafe_allow_html=True)
        cong_ph.markdown(f'<div class="{cong_class}">{cong_text}</div>',                                 unsafe_allow_html=True)
        timer_ph.markdown(f'<div class="metric-card"><h2>{active_time}s</h2><p>Signal Timer</p></div>',  unsafe_allow_html=True)

        # Update signal status
        if signal_active:
            sig_ph.markdown(f'<div class="green-signal">🟢 SIGNAL — GREEN<br><small>Green time: {sig_time}s | Density: {density}</small></div>', unsafe_allow_html=True)
        else:
            sig_ph.markdown(f'<div class="red-signal">🔴 SIGNAL — RED<br><small>Next green in: {active_time}s</small></div>', unsafe_allow_html=True)

        # Update minimap
        map_ph.plotly_chart(draw_minimap(total, density, signal_active), use_container_width=True, key=f"minimap_{frame_count}")

        # Update breakdown
        break_ph.markdown(f"""
        <div class="metric-card" style="text-align:left">
            <p style="color:#7eb8f7;font-weight:500;margin:0 0 8px">Vehicle Breakdown</p>
            <p style="color:#aac8f0;margin:3px 0">🚗 Cars: {breakdown['Car']}</p>
            <p style="color:#aac8f0;margin:3px 0">🏍 Motorcycles: {breakdown['Motorcycle']}</p>
            <p style="color:#aac8f0;margin:3px 0">🚌 Heavy Vehicles: {breakdown['Heavy Vehicle']}</p>
            <p style="color:#aac8f0;margin:3px 0">⚡ Avg Speed: {avg_spd} km/h</p>
        </div>""", unsafe_allow_html=True)

        # Update charts
        count_history.append(total)
        update_charts(all_speeds)

        # Update session stats
        update_session_stats()

        if stop_btn:
            st.session_state.paused = False
            st.toast("Detection stopped!", icon="⏹")
            break

    cap.release()

elif start_btn and video_file is None:
    st.error("Please select a video first from the sidebar!")