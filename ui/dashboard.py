"""
AImSafe Dashboard — Unified single-viewport design.
Zone overview + per-person tracking + camera toggle + worker detail navigation.
"""

import asyncio
import os
import time
from datetime import datetime

import cv2
import numpy as np
from nicegui import ui, app
from starlette.responses import StreamingResponse

import config
from vision.camera import Camera
from vision.pose_analyzer import PoseAnalyzer, PoseResult
from vision.face_analyzer import FaceAnalyzer, FaceResult
from engine.risk_engine import RiskEngine
from engine.zone_manager import ZoneManager
from engine.worker_state import WorkerState, RiskLevel, Alert
from alerts.alert_manager import AlertManager
from integration.wearable_sim import WearableSim
from integration.mqtt_bridge import MQTTBridge
from llm.openrouter_client import OpenRouterClient
from llm.safety_coach import SafetyCoach
from llm.shift_reporter import ShiftReporter

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
]

UI_FPS = 10
TREND_HISTORY = 120
CAPTURE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "captures")
CAPTURE_COOLDOWN = 30
CAPTURE_THRESHOLD = 35

PERSON_COLORS = [
    (0, 255, 0), (255, 165, 0), (255, 0, 255), (0, 255, 255), (128, 128, 255),
]

# Module-level shared state for cross-page access
_shared = {
    "all_states": [],
    "all_poses": [],
    "person_hrs": {},
    "face": None,
    "area_safety": 100.0,
    "session_start": 0.0,
    "alert_count": 0,
}

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

* { box-sizing: border-box; }
body { font-family: 'Inter', system-ui, sans-serif; overflow: hidden; }
.nicegui-content { padding: 0 !important; }

.app-shell {
    height: calc(100vh - 44px);
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

/* Glass cards */
.gc {
    background: rgba(30,41,59,0.7);
    border: 1px solid rgba(100,116,139,0.2);
    border-radius: 10px;
    padding: 10px;
    backdrop-filter: blur(8px);
}
.gc:hover { border-color: rgba(99,102,241,0.4); }

/* Feed */
.feed-box {
    border-radius: 10px; overflow: hidden;
    border: 2px solid rgba(100,116,139,0.2);
    background: #0a0f1a;
}
.feed-box img { width: 100%; height: 100%; object-fit: cover; display: block; }

/* Alerts */
.alert-row { border-radius: 6px; padding: 6px 10px; font-size: 0.75rem; }
.a-crit { background: rgba(127,29,29,0.4); border-left: 4px solid #f87171; }
.a-warn { background: rgba(113,63,18,0.3); border-left: 4px solid #facc15; }
.a-info { background: rgba(30,58,138,0.25); border-left: 3px solid #60a5fa; }

/* Big notification popups */
.q-notification { font-size: 1.3rem !important; min-width: 450px !important; padding: 16px 20px !important; border-radius: 12px !important; }
.q-notification .q-notification__message { font-size: 1.2rem !important; line-height: 1.5 !important; font-weight: 600 !important; }
.q-notification .q-notification__actions { font-size: 1rem !important; }

/* Coach */
.coach-box {
    background: linear-gradient(135deg, rgba(30,58,138,0.2), rgba(88,28,135,0.1));
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 10px;
    padding: 10px;
}

/* Pulsing */
@keyframes plive { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }
.live-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #4ade80;
    animation: plive 2s ease-in-out infinite;
    display: inline-block;
}
@keyframes dpulse {
    0%,100% { border-color: rgba(248,113,113,0.2); }
    50% { border-color: rgba(248,113,113,0.7); }
}
.danger-border { animation: dpulse 1.5s ease-in-out infinite; }

/* Score colors */
.sc-safe { color: #4ade80; }
.sc-caution { color: #facc15; }
.sc-warning { color: #fb923c; }
.sc-danger { color: #f87171; }

/* Person card — clickable */
.person-card {
    background: rgba(15,23,42,0.6);
    border: 1px solid rgba(100,116,139,0.2);
    border-radius: 10px;
    padding: 8px 10px;
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
}
.person-card:hover {
    border-color: rgba(99,102,241,0.5);
    background: rgba(30,41,59,0.7);
}

/* Responsive */
@media (max-width: 900px) {
    .main-grid { flex-direction: column !important; }
    .side-panel { width: 100% !important; max-height: 40vh; overflow-y: auto; }
}

/* Voice */
@keyframes vpulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(248,113,113,0.4); }
    50% { box-shadow: 0 0 0 8px rgba(248,113,113,0); }
}
.voice-listening { animation: vpulse 1.5s ease-in-out infinite; background: rgba(248,113,113,0.2) !important; }
</style>

<script>
// Safety alarm sound using Web Audio API
let _alarmCtx = null;
let _alarmPlaying = false;
function playSafetyAlarm(type, level) {
    if (_alarmPlaying) return;
    _alarmPlaying = true;
    if (!_alarmCtx) _alarmCtx = new (window.AudioContext || window.webkitAudioContext)();
    const ctx = _alarmCtx;
    const now = ctx.currentTime;
    // Different sounds per alert type
    let freq, beeps, dur, msg;
    if (type === 'collision') {
        freq = level === 'CRITICAL' ? 880 : 660;
        beeps = level === 'CRITICAL' ? 4 : 2;
        dur = level === 'CRITICAL' ? 1.2 : 0.6;
        msg = level === 'CRITICAL' ? 'Danger! Workers too close!' : 'Caution, workers approaching.';
    } else if (type === 'microsleep') {
        freq = 1000; beeps = 5; dur = 1.5;
        msg = 'Alert! Microsleep detected! Worker falling asleep!';
    } else if (type === 'posture') {
        freq = 550; beeps = 3; dur = 0.9;
        msg = 'Warning. Unsafe lifting posture detected.';
    } else if (type === 'drowsy') {
        freq = 750; beeps = 3; dur = 1.0;
        msg = 'Warning. Worker showing signs of drowsiness.';
    } else {
        freq = 660; beeps = 2; dur = 0.6;
        msg = 'Safety alert.';
    }
    for (let i = 0; i < beeps; i++) {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.type = type === 'microsleep' ? 'sawtooth' : 'square';
        osc.frequency.value = freq;
        gain.gain.setValueAtTime(0.3, now + i * (dur / beeps));
        gain.gain.exponentialRampToValueAtTime(0.01, now + i * (dur / beeps) + dur / beeps * 0.8);
        osc.connect(gain).connect(ctx.destination);
        osc.start(now + i * (dur / beeps));
        osc.stop(now + i * (dur / beeps) + dur / beeps * 0.9);
    }
    const utt = new SpeechSynthesisUtterance(msg);
    utt.rate = 1.2; utt.volume = 1.0;
    speechSynthesis.speak(utt);
    setTimeout(() => { _alarmPlaying = false; }, dur * 1000 + 2500);
}
// Legacy wrapper
function playCollisionAlarm(level) { playSafetyAlarm('collision', level); }

let voiceListening = false;
let recognition = null;

function toggleVoice() {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        alert('Voice not supported in this browser. Use Chrome or Edge.');
        return;
    }
    if (voiceListening) { if (recognition) recognition.stop(); return; }
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SR();
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;
    recognition.onstart = () => {
        voiceListening = true;
        document.querySelectorAll('.q-btn--round').forEach(b => {
            if (b.querySelector('.q-icon') && b.querySelector('.q-icon').textContent.trim() === 'mic')
                b.classList.add('voice-listening');
        });
    };
    recognition.onresult = async (event) => {
        const text = event.results[0][0].transcript;
        try {
            const resp = await fetch('/api/voice', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text: text})
            });
            const data = await resp.json();
            const utterance = new SpeechSynthesisUtterance(data.response);
            utterance.rate = 1.0;
            speechSynthesis.speak(utterance);
        } catch(e) { console.error('Voice API error:', e); }
    };
    recognition.onend = () => {
        voiceListening = false;
        document.querySelectorAll('.voice-listening').forEach(b => b.classList.remove('voice-listening'));
    };
    recognition.onerror = (e) => {
        console.error('Speech error:', e.error);
        voiceListening = false;
        document.querySelectorAll('.voice-listening').forEach(b => b.classList.remove('voice-listening'));
    };
    recognition.start();
}
</script>
"""


def build_dashboard():
    """Build unified single-viewport dashboard with camera toggle and worker navigation."""

    # ── Subsystems ──
    camera = Camera().start()
    pose_analyzer = PoseAnalyzer()
    face_analyzer = FaceAnalyzer()
    zone_manager = ZoneManager()
    alert_manager = AlertManager()
    mqtt_bridge = MQTTBridge()
    llm_client = OpenRouterClient()
    safety_coach = SafetyCoach(llm_client)
    shift_reporter = ShiftReporter(llm_client)

    session_start = time.time()
    safety_scores: list[float] = []
    fatigue_scores: list[float] = []
    blink_rates: list[float] = []
    trend_data = {"times": [], "safety": [], "fatigue": [], "hr": []}
    _last_trend = [0.0]

    latest_state = {"state": WorkerState(), "frame_jpg": b"", "hr": 72.0, "pose": PoseResult()}
    _coaching_active = {"running": False}

    # ── Multi-person tracking ──
    _person_engines: dict[int, RiskEngine] = {}
    _person_wearables: dict[int, WearableSim] = {}
    _person_hrs: dict[int, tuple[float, float]] = {}  # idx → (hr, hrv)
    _person_count = [0]
    _area_safety = [100.0]
    _show_camera = [True]  # togglable
    _collision_timestamps: dict[str, float] = {}  # cooldown tracking for collision alerts
    _collision_prev_dists: dict[str, float] = {}  # previous distances for direction tracking
    _pending_alarm: list[str] = []  # queued alarm levels to play in UI thread

    os.makedirs(CAPTURE_DIR, exist_ok=True)
    _capture_state = {"last_time": 0.0, "was_below": False}

    # ── MJPEG ──
    def _mjpeg():
        while True:
            jpg = latest_state.get("frame_jpg", b"")
            if jpg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            import time as _t
            _t.sleep(1 / UI_FPS)

    @app.get("/video_feed")
    def video_feed():
        return StreamingResponse(_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")

    from starlette.responses import FileResponse as _FileResponse

    @app.get("/captures/{filename}")
    def serve_capture(filename: str):
        fpath = os.path.join(CAPTURE_DIR, filename)
        if os.path.exists(fpath):
            return _FileResponse(fpath, media_type="image/jpeg")
        return StreamingResponse(iter([b""]), status_code=404)

    # ── Voice assistant ──
    async def handle_voice_command(text: str) -> str:
        if not text.strip():
            return "I didn't catch that. Please try again."
        state = latest_state["state"]
        hr = latest_state.get("hr", 0)
        elapsed = int(time.time() - session_start)
        mins, secs = divmod(elapsed, 60)
        context = (
            f"Area safety: {_area_safety[0]:.0f}/100, people: {_person_count[0]}, "
            f"fatigue={state.fatigue_score:.0f}, PERCLOS={state.perclos:.0%}, "
            f"stress={state.stress_level:.0%}, REBA={state.reba_score}, "
            f"HR={hr:.0f}bpm, session={mins}m{secs}s, alerts={len(alert_manager.history)}"
        )
        response = await llm_client.chat(
            system=(
                "You are a factory floor AI safety assistant. Answer briefly (1-2 sentences max). "
                "Be friendly and helpful. The worker may have their hands full."
            ),
            user=f"Worker asks: \"{text}\"\n\nCurrent data: {context}",
        )
        if response:
            return response
        text_lower = text.lower()
        if "status" in text_lower or "how am i" in text_lower:
            return f"Safety score is {_area_safety[0]:.0f}. Fatigue at {state.fatigue_score:.0f}."
        if "break" in text_lower:
            return "Your fatigue is elevated. I recommend a break." if state.fatigue_score > 30 else "Vitals look good."
        return f"Area safety is at {_area_safety[0]:.0f}."

    from starlette.requests import Request
    from starlette.responses import JSONResponse

    @app.post("/api/voice")
    async def voice_api(request: Request):
        body = await request.json()
        return JSONResponse({"response": await handle_voice_command(body.get("text", ""))})

    # ── Theme ──
    ui.dark_mode(True)
    ui.colors(primary='#818cf8', positive='#4ade80', negative='#f87171', warning='#facc15')
    ui.add_head_html(CUSTOM_CSS)

    # ════════════════════════════════════════
    #  HEADER
    # ════════════════════════════════════════
    with ui.header().classes("items-center justify-between px-4 py-1").style(
        "background: #0a0f1a; border-bottom: 1px solid rgba(100,116,139,0.15); min-height: 36px; height: 44px;"
    ):
        with ui.row().classes("items-center gap-2"):
            ui.icon("shield", size="sm").classes("text-indigo-400")
            ui.label("AImSafe").classes("text-sm font-bold tracking-wide")
        with ui.row().classes("items-center gap-2"):
            if camera.demo_mode:
                ui.badge("DEMO", color="orange").classes("text-[8px]")
            with ui.row().classes("items-center gap-1"):
                ui.element("span").classes("live-dot")
                ui.label("LIVE").classes("text-[8px] text-green-400 font-bold tracking-widest")
            ui.badge(
                "LLM ON" if llm_client.is_available else "LLM OFF",
                color="green" if llm_client.is_available else "gray"
            ).classes("text-[8px]")

            # Camera toggle
            def toggle_camera():
                _show_camera[0] = not _show_camera[0]
                if _show_camera[0]:
                    cam_toggle.props("icon=videocam color=green")
                    feed_container.style("flex: 3; min-height: 0; overflow: hidden; display: flex;")
                    gallery_full_container.style("flex: 3; min-height: 0; overflow: hidden; display: none;")
                else:
                    cam_toggle.props("icon=videocam_off color=grey")
                    feed_container.style("flex: 3; min-height: 0; overflow: hidden; display: none;")
                    gallery_full_container.style("flex: 3; min-height: 0; overflow: hidden; display: flex; flex-direction: column;")
                    _update_gallery_full()

            cam_toggle = ui.button(icon="videocam", on_click=toggle_camera).props(
                "round dense flat color=green size=sm"
            ).tooltip("Toggle camera")
            ui.button(icon="mic", on_click=lambda: ui.run_javascript('toggleVoice()')).props(
                "round dense flat color=white size=sm"
            ).classes("text-slate-400")
            uptime_label = ui.label("00:00").classes("text-[9px] text-slate-500 font-mono")

    # ════════════════════════════════════════
    #  MAIN APP
    # ════════════════════════════════════════
    with ui.column().classes("app-shell w-full"):

        # ── ZONE OVERVIEW BAR ──
        with ui.row().classes("w-full items-center gap-2 px-3 py-1 flex-nowrap").style(
            "background: rgba(15,23,42,0.8); border-bottom: 1px solid rgba(100,116,139,0.1); flex-shrink: 0;"
        ):
            gauge = ui.echart({
                "series": [{
                    "type": "gauge", "startAngle": 220, "endAngle": -40, "min": 0, "max": 100,
                    "progress": {"show": True, "roundCap": True, "width": 8},
                    "pointer": {"show": False},
                    "axisLine": {"roundCap": True, "lineStyle": {"width": 8,
                        "color": [[0.2, "#f87171"], [0.4, "#fb923c"], [0.7, "#facc15"], [1, "#4ade80"]]}},
                    "axisTick": {"show": False}, "splitLine": {"show": False}, "axisLabel": {"show": False},
                    "title": {"fontSize": 7, "color": "#64748b", "offsetCenter": [0, "72%"]},
                    "detail": {"fontSize": 24, "fontWeight": "bold", "color": "#f1f5f9",
                               "valueAnimation": True, "formatter": "{value}", "offsetCenter": [0, "5%"]},
                    "data": [{"value": 100, "name": "ZONE"}]
                }]
            }).style("width: 100px; height: 100px; min-width: 100px;")

            risk_badge = ui.badge("SAFE", color="green").classes("text-[10px] px-2 font-bold tracking-widest")

            # Zone metrics
            with ui.row().classes("flex-1 gap-1.5 flex-wrap justify-center items-center"):
                def _zm(icon, lbl, default):
                    with ui.column().classes("items-center px-2 py-0.5").style(
                        "background: rgba(30,41,59,0.5); border: 1px solid rgba(100,116,139,0.12); "
                        "border-radius: 8px; min-width: 75px;"
                    ):
                        with ui.row().classes("items-center gap-1"):
                            ui.icon(icon, size="xs").classes("text-slate-500")
                            ui.label(lbl).classes("text-[7px] text-slate-500 uppercase tracking-widest font-semibold")
                        v = ui.label(default).classes("text-base font-bold text-slate-100 font-mono")
                    return v

                h_people = _zm("groups", "People", "0")
                h_fatigue = _zm("bedtime", "Avg Fatigue", "0")
                h_stress = _zm("mood_bad", "Avg Stress", "0%")
                h_reba = _zm("fitness_center", "Avg REBA", "1")
                h_hr = _zm("favorite", "Avg HR", "--")

            # Zone active alerts count
            with ui.column().classes("items-center px-2"):
                h_active_flags = ui.label("0").classes("text-lg font-bold text-slate-100 font-mono")
                ui.label("ALERTS").classes("text-[7px] text-slate-500 uppercase tracking-widest")

        # ══════════════════════════════════════════
        #  MAIN CONTENT — single column, 2 rows
        # ══════════════════════════════════════════
        with ui.row().classes("flex-1 w-full gap-2 px-2 py-1").style("overflow: hidden; min-height: 0; height: 100%;"):

            # ── LEFT: Camera + Trend — fill full height ──
            with ui.column().classes("gap-1").style("width: 55%; min-width: 0; height: 100%;"):

                # Camera feed
                feed_container = ui.column().classes("w-full").style("flex: 3; min-height: 0; overflow: hidden;")
                feed_card = None
                with feed_container:
                    with ui.card().classes("feed-box w-full p-0").style("height: 100%;") as feed_card:
                        ui.image("/video_feed").classes("w-full h-full")

                # Large incident gallery — shown when camera is off (replaces camera space)
                gallery_full_container = ui.column().classes("gc w-full p-2").style(
                    "flex: 3; min-height: 0; overflow: hidden; display: none;"
                )
                with gallery_full_container:
                    with ui.row().classes("items-center gap-1 mb-1"):
                        ui.icon("photo_library", size="sm").classes("text-amber-400")
                        ui.label("INCIDENT GALLERY").classes("text-[8px] tracking-[0.2em] text-slate-400 font-semibold")
                        gallery_full_count = ui.label("0").classes("text-[9px] text-slate-500")
                    gallery_full_grid = ui.row().classes("w-full gap-2 flex-wrap").style(
                        "overflow-y: auto; flex: 1; min-height: 0; align-content: flex-start;"
                    )

                # (Mini gallery removed — full gallery shows when camera is off)

                # Zone trend — fills remaining space
                with ui.card().classes("gc w-full p-1").style("flex: 1; min-height: 0; overflow: hidden;"):
                    ui.label("ZONE TREND").classes("text-[6px] tracking-[0.2em] text-slate-500 font-semibold")
                    trend_chart = ui.echart({
                        "grid": {"left": 28, "right": 8, "top": 18, "bottom": 16},
                        "legend": {"data": ["Safety", "Fatigue", "HR"],
                                   "textStyle": {"color": "#94a3b8", "fontSize": 7}, "top": 0, "right": 0},
                        "tooltip": {"trigger": "axis", "backgroundColor": "#1e293b", "borderColor": "#334155",
                                    "textStyle": {"color": "#f1f5f9", "fontSize": 9}},
                        "xAxis": {"type": "category", "data": [], "show": True,
                                  "axisLabel": {"color": "#475569", "fontSize": 6, "interval": "auto"},
                                  "axisLine": {"lineStyle": {"color": "#334155"}}},
                        "yAxis": {"type": "value", "min": 0, "max": 100, "show": True,
                                  "axisLabel": {"color": "#475569", "fontSize": 6},
                                  "splitLine": {"lineStyle": {"color": "rgba(100,116,139,0.08)"}}},
                        "series": [
                            {"name": "Safety", "type": "line", "smooth": True, "data": [],
                             "showSymbol": False, "lineStyle": {"color": "#4ade80", "width": 2},
                             "areaStyle": {"color": {"type": "linear", "x": 0, "y": 0, "x2": 0, "y2": 1,
                                 "colorStops": [{"offset": 0, "color": "rgba(74,222,128,0.12)"},
                                                {"offset": 1, "color": "rgba(74,222,128,0)"}]}}},
                            {"name": "Fatigue", "type": "line", "smooth": True, "data": [],
                             "showSymbol": False, "lineStyle": {"color": "#fb923c", "width": 1.5}},
                            {"name": "HR", "type": "line", "smooth": True, "data": [],
                             "showSymbol": False, "lineStyle": {"color": "#f472b6", "width": 1, "type": "dashed"}},
                        ]
                    }).style("width: 100%; height: 100%; min-height: 70px;")

            # ── RIGHT COLUMN: Alerts + Coach + Workers — fill full height ──
            with ui.column().classes("flex-1 gap-1").style("min-width: 0; height: 100%; overflow-y: auto;"):

                # Alerts — larger panel
                with ui.card().classes("gc w-full p-2").style("flex-shrink: 0;"):
                    with ui.row().classes("items-center justify-between mb-1"):
                        ui.icon("notifications_active", size="xs").classes("text-red-400")
                        ui.label("ALERTS").classes("text-[8px] tracking-[0.2em] text-slate-400 font-semibold")
                        alert_count_label = ui.label("0").classes("text-[10px] text-slate-400 font-bold")
                    alert_box = ui.column().classes("w-full gap-1").style("max-height: 160px; overflow-y: auto;")

                # Coach + Incidents side by side — equal height
                with ui.row().classes("w-full gap-1").style("min-height: 0;"):
                    # AI Coach
                    with ui.card().classes("coach-box p-2").style(
                        "flex: 1; min-width: 0; display: flex; flex-direction: column; max-height: 130px;"
                    ):
                        with ui.row().classes("items-center gap-1 mb-1"):
                            ui.icon("psychology", size="xs").classes("text-indigo-400")
                            ui.label("AI COACH").classes("text-[6px] tracking-[0.2em] text-slate-500 font-semibold")
                            coach_spinner = ui.spinner("dots", size="xs", color="indigo")
                            coach_spinner.set_visibility(False)
                        coach_label = ui.label(
                            "Waiting for data..." if llm_client.is_available else "LLM disabled"
                        ).classes("text-[9px] text-slate-300 leading-snug").style(
                            "overflow-y: auto; flex: 1; min-height: 0;"
                        )

                    # Incidents
                    with ui.card().classes("gc p-2").style(
                        "flex: 1; min-width: 0; display: flex; flex-direction: column; max-height: 130px;"
                        "background: linear-gradient(135deg, rgba(113,63,18,0.1), rgba(30,41,59,0.7));"
                        "border: 1px solid rgba(251,146,60,0.2);"
                    ):
                        with ui.row().classes("items-center gap-1 mb-1"):
                            ui.icon("warning_amber", size="xs").classes("text-amber-400")
                            ui.label("INCIDENTS").classes("text-[6px] tracking-[0.2em] text-slate-500 font-semibold")
                            incident_spinner = ui.spinner("dots", size="xs", color="amber")
                            incident_spinner.set_visibility(False)
                        incident_label = ui.label("No incidents yet.").classes(
                            "text-[9px] text-slate-400 leading-snug"
                        ).style("overflow-y: auto; flex: 1; min-height: 0;")

                # Workers list panel
                with ui.card().classes("gc w-full p-2").style("flex: 1; min-height: 0; display: flex; flex-direction: column;"):
                    with ui.row().classes("w-full items-center justify-between mb-1"):
                        with ui.row().classes("items-center gap-1"):
                            ui.icon("groups", size="xs").classes("text-indigo-400")
                            ui.label("WORKERS").classes("text-[7px] tracking-[0.2em] text-slate-500 font-semibold")
                            people_count_badge = ui.badge("0", color="indigo").classes("text-[8px]")

                        with ui.row().classes("items-center gap-1"):
                            async def on_report():
                                report_spin.set_visibility(True)
                                dur = (time.time() - session_start) / 60
                                rpt = await shift_reporter.generate(
                                    alerts=alert_manager.get_recent(50),
                                    avg_safety_score=sum(safety_scores[-100:]) / max(len(safety_scores[-100:]), 1),
                                    avg_fatigue=sum(fatigue_scores[-100:]) / max(len(fatigue_scores[-100:]), 1),
                                    total_incidents=len(alert_manager.history),
                                    duration_minutes=dur,
                                    peak_blink_rate=max(blink_rates) if blink_rates else 0,
                                    total_repetitions=sum(1 for a in alert_manager.history if a.rule_name == "repetition"),
                                )
                                report_md.set_content(rpt)
                                report_dialog.open()
                                report_spin.set_visibility(False)
                            ui.button(icon="summarize", on_click=on_report).props(
                                "round dense flat color=indigo size=xs"
                            ).tooltip("Shift Report")
                            report_spin = ui.spinner("dots", size="xs", color="indigo")
                            report_spin.set_visibility(False)

                    # Scrollable worker list
                    people_cards_box = ui.column().classes("w-full gap-1").style(
                        "overflow-y: auto; flex: 1; min-height: 0;"
                    )

    # ── Report dialog ──
    with ui.dialog() as report_dialog:
        with ui.card().classes("gc w-full max-w-2xl").style("max-height: 80vh; overflow-y: auto;"):
            ui.label("Shift Safety Report").classes("text-lg font-bold mb-2")
            report_md = ui.markdown("*Generating...*").classes("text-sm leading-relaxed")
            ui.button("Close", on_click=report_dialog.close).props("color=indigo rounded flat")

    # ════════════════════════════════════════
    #  DRAWING HELPERS
    # ════════════════════════════════════════

    def draw_skeleton(frame, landmarks, color=(0, 255, 0)):
        h, w = frame.shape[:2]
        points = {}
        for i, lm in enumerate(landmarks):
            px, py = int(lm.x * w), int(lm.y * h)
            points[i] = (px, py)
            if getattr(lm, "visibility", 1.0) > 0.5:
                cv2.circle(frame, (px, py), 4, color, -1)
        for s, e in POSE_CONNECTIONS:
            if s in points and e in points:
                cv2.line(frame, points[s], points[e], (255, 255, 255), 2)

    def draw_overlays_multi(frame, poses, states, area_safety):
        h, w = frame.shape[:2]
        for idx, (pose, state) in enumerate(zip(poses, states)):
            if not pose.visible or not pose.landmarks:
                continue
            color = PERSON_COLORS[idx % len(PERSON_COLORS)]
            draw_skeleton(frame, pose.landmarks, color=color)
            xs = [int(lm.x * w) for lm in pose.landmarks if getattr(lm, "visibility", 1.0) > 0.5]
            ys = [int(lm.y * h) for lm in pose.landmarks if getattr(lm, "visibility", 1.0) > 0.5]
            if xs and ys:
                pad = 20
                x1, y1 = max(0, min(xs)-pad), max(0, min(ys)-pad)
                x2, y2 = min(w, max(xs)+pad), min(h, max(ys)+pad)
                box_color = {RiskLevel.SAFE: (0,220,0), RiskLevel.CAUTION: (0,220,220),
                             RiskLevel.WARNING: (0,165,255), RiskLevel.DANGER: (0,0,255)}.get(state.risk_level, (200,200,200))
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                hr_val = _person_hrs.get(idx, (72.0, 50.0))[0]
                label = f"P{idx+1}: {state.overall_safety_score:.0f} | HR:{hr_val:.0f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+8, y1), box_color, -1)
                cv2.putText(frame, label, (x1+4, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)

        # Draw collision warning lines between close workers (two levels)
        COLLISION_VIS_WARN = 0.65    # yellow line — approaching (~4m)
        COLLISION_VIS_DANGER = 0.45  # red line — dangerously close (~2-3m)
        for i in range(len(poses)):
            for j in range(i + 1, len(poses)):
                if not poses[i].visible or not poses[j].visible:
                    continue
                cx1, cy1 = poses[i].body_centroid
                cx2, cy2 = poses[j].body_centroid
                dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
                if dist < COLLISION_VIS_WARN:
                    p1 = (int(cx1 * w), int(cy1 * h))
                    p2 = (int(cx2 * w), int(cy2 * h))
                    mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                    if dist < COLLISION_VIS_DANGER:
                        # DANGER — thick red line + flashing warning
                        cv2.line(frame, p1, p2, (0, 0, 255), 3)
                        cv2.putText(frame, "!! DANGER !!", (mid[0] - 40, mid[1] - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                    else:
                        # WARNING — yellow dashed line
                        cv2.line(frame, p1, p2, (0, 200, 255), 2)
                        cv2.putText(frame, "CAUTION", (mid[0] - 28, mid[1] - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1, cv2.LINE_AA)
                        dist_pct = int(dist * 100)
                        cv2.putText(frame, f"{dist_pct}%", (mid[0] - 10, mid[1] + 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 255), 1, cv2.LINE_AA)

        area_level = RiskEngine._score_to_level(area_safety)
        bar_color = {RiskLevel.SAFE: (0,180,0), RiskLevel.CAUTION: (0,180,180),
                     RiskLevel.WARNING: (0,140,255), RiskLevel.DANGER: (0,0,230)}.get(area_level, (200,200,200))
        ov = frame.copy()
        cv2.rectangle(ov, (0,0), (w, 24), (0,0,0), -1)
        cv2.addWeighted(ov, 0.6, frame, 0.4, 0, frame)
        cv2.rectangle(frame, (0,0), (int(w * area_safety / 100), 3), bar_color, -1)
        n = len([p for p in poses if p.visible])
        cv2.putText(frame, f"Zone: {area_safety:.0f} | {area_level.value} | {n} people",
                    (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
        return frame

    # ════════════════════════════════════════
    #  SCREENSHOT CAPTURE
    # ════════════════════════════════════════

    def maybe_capture(state, jpg):
        now = time.time()
        back_angle = state.angles.get("back", 0) if hasattr(state, "angles") else 0
        bad_posture = back_angle > 35 and state.pose_flags.get("unsafe_lift", False)
        should_capture = (
            state.overall_safety_score < CAPTURE_THRESHOLD
            or state.perclos > 0.50
            or bad_posture
        )
        if should_capture:
            if not _capture_state["was_below"] and now - _capture_state["last_time"] > CAPTURE_COOLDOWN:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                if bad_posture:
                    reason = "posture"
                elif state.perclos > 0.50:
                    reason = "perclos"
                else:
                    reason = "safety"
                fname = f"incident_{ts}_{reason}_score{int(state.overall_safety_score)}.jpg"
                with open(os.path.join(CAPTURE_DIR, fname), "wb") as f:
                    f.write(jpg)
                _capture_state["last_time"] = now
                _capture_state["was_below"] = True
                return fname
            _capture_state["was_below"] = True
        else:
            _capture_state["was_below"] = False
        return None

    # ════════════════════════════════════════
    #  VISION LOOP
    # ════════════════════════════════════════

    _demo_tick = [0]

    def _demo_pose(tick):
        import math as m
        t = tick * 0.05
        back = 15 + 40 * abs(m.sin(t * 0.3))
        arm = 90 + 50 * m.sin(t * 0.7)
        flags = {}
        if back > 45: flags["unsafe_lift"] = True
        return PoseResult(visible=True, angles={"back": back, "shoulder_tilt": 5 + 3*m.sin(t),
            "left_knee": 170, "right_knee": 170, "left_arm_extension": arm,
            "right_arm_extension": 90+30*m.cos(t*0.5)},
            body_centroid=(0.35+0.15*m.sin(t*0.5), 0.5), flags=flags)

    def _demo_pose_2(tick):
        import math as m
        t = tick * 0.05
        back = 10 + 20 * abs(m.sin(t * 0.2 + 1.5))
        arm = 80 + 30 * m.sin(t * 0.5 + 2.0)
        flags = {}
        if back > 45: flags["unsafe_lift"] = True
        return PoseResult(visible=True,
            angles={"back": back, "shoulder_tilt": 3 + 2 * m.sin(t + 1),
                     "left_knee": 165, "right_knee": 165,
                     "left_arm_extension": arm,
                     "right_arm_extension": 85 + 20 * m.cos(t * 0.4)},
            body_centroid=(0.65 + 0.1 * m.sin(t * 0.3), 0.5), flags=flags)

    def _demo_pose_3(tick):
        import math as m
        t = tick * 0.05
        back = 8 + 15 * abs(m.sin(t * 0.15 + 3.0))
        arm = 95 + 25 * m.sin(t * 0.6 + 1.0)
        flags = {}
        if back > 45: flags["unsafe_lift"] = True
        return PoseResult(visible=True,
            angles={"back": back, "shoulder_tilt": 4 + 2.5 * m.sin(t + 2),
                     "left_knee": 168, "right_knee": 172,
                     "left_arm_extension": arm,
                     "right_arm_extension": 88 + 15 * m.cos(t * 0.3)},
            body_centroid=(0.50 + 0.08 * m.cos(t * 0.4), 0.45 + 0.05 * m.sin(t * 0.2)), flags=flags)

    def _demo_face_for(tick, person_idx):
        """Generate distinct face/fatigue data per person so each worker is independent."""
        import math as m
        t = tick * 0.05
        # Each person has a different fatigue cycle offset and pattern
        offset = person_idx * 900  # stagger drowsy episodes
        cycle = tick + offset
        perclos_base = 0.04 + 0.02 * m.sin(t * 0.1 + person_idx)
        # Different drowsy episode timing per person
        in_drowsy = (cycle % 3000) > (2600 + person_idx * 50)
        perclos = min(perclos_base + (0.22 if in_drowsy else 0.0), 0.55)
        drowsy = perclos > 0.20
        microsleep = (cycle % 3000) > 2980
        yawn = (1200 + person_idx * 100) < (cycle % 1400) < (1400)
        stress = 0.12 + 0.08 * m.sin(t * 0.08 + person_idx * 1.5) + (0.25 if in_drowsy else 0.0)
        blink = 16 + 6 * m.sin(t * 0.2 + person_idx * 0.7) + (5 if in_drowsy else 0)
        return FaceResult(visible=True, blink_rate=blink,
                          microsleep=microsleep, head_pitch=3 * m.sin(t * 0.4 + person_idx),
                          perclos=perclos, drowsy=drowsy,
                          stress_level=min(stress, 1.0), yawn_detected=yawn,
                          gaze_focus=0.9 - 0.2 * abs(m.sin(t * 0.15 + person_idx * 0.5)))

    async def vision_loop():
        while True:
            frame = camera.read()
            if frame is None:
                await asyncio.sleep(0.05)
                continue

            if camera.demo_mode:
                _demo_tick[0] += 1
                tick = _demo_tick[0]
                poses = [_demo_pose(tick), _demo_pose_2(tick), _demo_pose_3(tick)]
                per_person_faces = [_demo_face_for(tick, i) for i in range(3)]
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                poses = pose_analyzer.analyze_multi(rgb)
                if not poses:
                    poses = [PoseResult()]
                real_face = face_analyzer.analyze(rgb)
                # In real mode, only first person gets face data (single camera)
                per_person_faces = [real_face] + [FaceResult()] * (len(poses) - 1)

            # Evaluate each person with their own RiskEngine + WearableSim
            all_states = []
            all_poses = []
            for idx, pose in enumerate(poses):
                if not pose.visible:
                    continue
                if idx not in _person_engines:
                    _person_engines[idx] = RiskEngine(zone_manager)
                    _person_wearables[idx] = WearableSim(
                        base_hr=72 + idx * 5,
                        base_hrv=50 - idx * 3,
                    )
                # Per-person face data
                person_face = per_person_faces[idx] if idx < len(per_person_faces) else FaceResult()

                # Per-person wearable
                wearable = _person_wearables[idx]
                if person_face.visible:
                    wearable.set_stress(min(1.0, person_face.blink_rate / 40))
                hr, hrv = wearable.update()
                _person_hrs[idx] = (hr, hrv)

                state = _person_engines[idx].evaluate(
                    pose, person_face,
                    heart_rate=hr, hrv=hrv,
                )
                all_states.append(state)
                all_poses.append(pose)

            # ── Collision detection between workers ──
            # Two-level + direction-aware: only warn when approaching, not separating
            COLLISION_WARN = 0.65   # ~65% of frame — early warning (~4m)
            COLLISION_DANGER = 0.45 # ~45% of frame — danger zone (~2-3m)
            for i in range(len(all_poses)):
                for j in range(i + 1, len(all_poses)):
                    if not all_poses[i].visible or not all_poses[j].visible:
                        continue
                    cx1, cy1 = all_poses[i].body_centroid
                    cx2, cy2 = all_poses[j].body_centroid
                    dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
                    collision_key = f"collision_{i}_{j}"

                    # Check direction: approaching or separating
                    prev_dist = _collision_prev_dists.get(collision_key, dist)
                    _collision_prev_dists[collision_key] = dist
                    approaching = dist < prev_dist - 0.002  # small threshold to ignore jitter
                    # Workers are separating — skip alert
                    if not approaching and dist > COLLISION_DANGER:
                        continue

                    if dist < COLLISION_WARN:
                        collision_cooldown = 8  # seconds — warn more often
                        now_t = time.time()
                        last_collision = _collision_timestamps.get(collision_key, 0)
                        if now_t - last_collision >= collision_cooldown:
                            if dist < COLLISION_DANGER:
                                severity = "CRITICAL"
                                msg = f"Workers {i+1} & {j+1} DANGEROUSLY close — collision risk! (dist: {dist:.2f})"
                            else:
                                severity = "WARNING"
                                msg = f"Workers {i+1} & {j+1} approaching each other — caution (dist: {dist:.2f})"
                            collision_alert = Alert(
                                rule_name="collision",
                                severity=severity,
                                message=msg,
                            )
                            all_states[i].active_alerts.append(collision_alert)
                            _collision_timestamps[collision_key] = now_t
                            # Queue alarm for UI thread
                            _pending_alarm.append(severity)

            # Clean up stale person entries
            active_ids = set(range(len([p for p in poses if p.visible])))
            for old_id in list(_person_engines.keys()):
                if old_id not in active_ids:
                    del _person_engines[old_id]
                    _person_wearables.pop(old_id, None)
                    _person_hrs.pop(old_id, None)

            # Area safety = average
            if all_states:
                area_safety = sum(s.overall_safety_score for s in all_states) / len(all_states)
                primary_state = all_states[0]
            else:
                area_safety = 100.0
                primary_state = WorkerState()

            _person_count[0] = len(all_states)
            _area_safety[0] = area_safety
            primary_state.overall_safety_score = area_safety
            primary_state.risk_level = RiskEngine._score_to_level(area_safety)

            safety_scores.append(area_safety)
            fatigue_scores.append(primary_state.fatigue_score)
            primary_face = per_person_faces[0] if per_person_faces else FaceResult()
            if primary_face.visible:
                blink_rates.append(primary_face.blink_rate)

            all_alerts = []
            _ALLOWED_ALERTS = {"fatigue", "unsafe_lift", "collision"}
            for s in all_states:
                for a in s.active_alerts:
                    if a.rule_name in _ALLOWED_ALERTS:
                        all_alerts.append(a)
            alert_manager.process(all_alerts)
            mqtt_bridge.publish(primary_state)

            # Queue sound alarms for microsleep and posture
            for a in all_alerts:
                if a.rule_name == "fatigue" and "microsleep" in a.message.lower():
                    _pending_alarm.append("microsleep")
                elif a.rule_name == "fatigue" and "drowsiness" in a.message.lower():
                    _pending_alarm.append("drowsy")
                elif a.rule_name == "unsafe_lift":
                    _pending_alarm.append("posture")

            if all_alerts and not _coaching_active.get("running"):
                asyncio.create_task(_coach(primary_state))

            annotated = draw_overlays_multi(frame.copy(), all_poses, all_states, area_safety)
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
            jpg = buf.tobytes()

            captured = maybe_capture(primary_state, jpg)
            # Also capture on collision
            collision_alerts = [a for a in all_alerts if a.rule_name == "collision" and a.severity == "CRITICAL"]
            if collision_alerts and time.time() - _capture_state.get("last_collision_cap", 0) > CAPTURE_COOLDOWN:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"incident_{ts}_collision_score{int(area_safety)}.jpg"
                with open(os.path.join(CAPTURE_DIR, fname), "wb") as f:
                    f.write(jpg)
                _capture_state["last_collision_cap"] = time.time()
                captured = captured or fname
            if captured:
                latest_state["last_capture"] = captured

            avg_hr = sum(h for h, _ in _person_hrs.values()) / max(len(_person_hrs), 1) if _person_hrs else 72.0
            latest_state.update({
                "state": primary_state, "frame_jpg": jpg, "hr": avg_hr,
                "pose": all_poses[0] if all_poses else PoseResult(),
                "face": primary_face, "hrv": 50.0,
                "all_states": list(all_states),
                "all_poses": list(all_poses),
            })
            # Update shared state for worker detail pages
            _shared["all_states"] = list(all_states)
            _shared["person_hrs"] = dict(_person_hrs)
            _shared["face"] = primary_face
            _shared["area_safety"] = area_safety
            _shared["session_start"] = session_start
            _shared["alert_count"] = len(alert_manager.history)
            await asyncio.sleep(1 / config.TARGET_FPS)

    # ════════════════════════════════════════
    #  UI UPDATE
    # ════════════════════════════════════════

    _prev_alerts = {"count": 0, "cap": ""}

    def update_ui():
        state = latest_state["state"]
        if not latest_state.get("frame_jpg"):
            return

        now = time.time()
        elapsed = int(now - session_start)
        m, s = divmod(elapsed, 60)
        uptime_label.set_text(f"{m:02d}:{s:02d}")

        # Play queued safety alarms (pick most urgent)
        if _pending_alarm:
            alarms = list(_pending_alarm)
            _pending_alarm.clear()
            # Priority: microsleep > CRITICAL collision > posture > drowsy > WARNING collision
            if "microsleep" in alarms:
                ui.run_javascript("playSafetyAlarm('microsleep', 'CRITICAL')")
            elif "CRITICAL" in alarms:
                ui.run_javascript("playSafetyAlarm('collision', 'CRITICAL')")
            elif "posture" in alarms:
                ui.run_javascript("playSafetyAlarm('posture', 'WARNING')")
            elif "drowsy" in alarms:
                ui.run_javascript("playSafetyAlarm('drowsy', 'WARNING')")
            elif "WARNING" in alarms:
                ui.run_javascript("playSafetyAlarm('collision', 'WARNING')")

        all_person_states = latest_state.get("all_states", [])
        n_people = len(all_person_states)

        # ── Zone gauge ──
        score = _area_safety[0]
        gauge.options["series"][0]["data"][0]["value"] = round(score)
        gauge.update()

        lvl = RiskEngine._score_to_level(score)
        risk_badge.set_text(lvl.value)
        badge_color = {"SAFE": "green", "CAUTION": "yellow", "WARNING": "orange", "DANGER": "red"}.get(lvl.value, "gray")
        risk_badge.props(f"color={badge_color}")
        if feed_card is not None:
            if lvl == RiskLevel.DANGER:
                feed_card.classes(add="danger-border")
            else:
                feed_card.classes(remove="danger-border")

        # ── Zone aggregates ──
        h_people.set_text(str(n_people))
        if all_person_states:
            avg_fatigue = sum(s.fatigue_score for s in all_person_states) / n_people
            avg_stress = sum(s.stress_level for s in all_person_states) / n_people
            avg_reba = sum(s.reba_score for s in all_person_states) / n_people
            avg_hr = sum(h for h, _ in _person_hrs.values()) / max(len(_person_hrs), 1) if _person_hrs else 72.0
        else:
            avg_fatigue, avg_stress, avg_reba, avg_hr = 0, 0, 0, 72.0
        h_fatigue.set_text(f"{avg_fatigue:.0f}")
        h_stress.set_text(f"{avg_stress:.0%}")
        h_reba.set_text(f"{avg_reba:.0f}")
        h_hr.set_text(f"{avg_hr:.0f}")

        # ── Zone active alerts count ──
        h_active_flags.set_text(str(len(alert_manager.history)))

        # ── Worker list ──
        people_count_badge.set_text(str(n_people))
        people_cards_box.clear()
        color_map = {"SAFE": "green", "CAUTION": "yellow", "WARNING": "orange", "DANGER": "red"}
        border_map = {"SAFE": "#4ade80", "CAUTION": "#facc15", "WARNING": "#fb923c", "DANGER": "#f87171"}
        sc_class = {"SAFE": "sc-safe", "CAUTION": "sc-caution", "WARNING": "sc-warning", "DANGER": "sc-danger"}

        with people_cards_box:
            if n_people == 0:
                ui.label("No workers detected").classes("text-xs text-slate-500 italic py-2")
            else:
                for idx in range(n_people):
                    ps = all_person_states[idx]
                    ps_level = ps.risk_level.value
                    hr_val, hrv_val = _person_hrs.get(idx, (72.0, 50.0))
                    border_color = border_map.get(ps_level, "#64748b")

                    with ui.card().classes("w-full p-2").style(
                        f"background: rgba(15,23,42,0.6); border: 1px solid {border_color}; "
                        f"border-left: 3px solid {border_color}; border-radius: 8px;"
                    ):
                        # Row 1: Name + score + status + nav button
                        with ui.row().classes("w-full items-center justify-between"):
                            with ui.row().classes("items-center gap-2"):
                                p_color = ['#4ade80', '#fb923c', '#e879f9', '#22d3ee', '#a78bfa'][idx % 5]
                                ui.icon("person", size="xs").style(f"color: {p_color};")
                                ui.label(f"Worker {idx + 1}").classes("text-xs font-bold text-slate-200")
                                ui.badge(ps_level, color=color_map.get(ps_level, "gray")).classes("text-[7px] px-1")
                            with ui.row().classes("items-center gap-2"):
                                ui.label(f"{ps.overall_safety_score:.0f}").classes(
                                    f"text-xl font-black {sc_class.get(ps_level, 'text-slate-100')}"
                                )
                                ui.button(icon="open_in_new", on_click=lambda s=idx: ui.navigate.to(f"/worker/{s}")).props(
                                    "round dense flat size=xs color=indigo"
                                ).tooltip(f"Full details for Worker {idx + 1}")

                        # Row 2: Key metrics
                        with ui.row().classes("w-full gap-1 flex-wrap mt-1"):
                            def _wm(lbl, val, w_idx=idx):
                                with ui.column().classes("items-center px-1.5 py-0.5").style(
                                    "background: rgba(30,41,59,0.4); border-radius: 5px; min-width: 48px; flex: 1;"
                                ):
                                    ui.label(val).classes("text-[11px] font-bold text-slate-200 font-mono")
                                    ui.label(lbl).classes("text-[6px] text-slate-500 uppercase")

                            _wm("Back", f"{ps.angles.get('back', 0):.0f}\u00b0")
                            _wm("Fatigue", f"{ps.fatigue_score:.0f}")
                            _wm("PERCLOS", f"{ps.perclos:.0%}")
                            _wm("HR", f"{hr_val:.0f}")
                            _wm("REBA", f"{ps.reba_score}")
                            _wm("Stress", f"{ps.stress_level:.0%}")

                        # Row 3: Active flags (if any)
                        active_flags = []
                        if ps.pose_flags.get("unsafe_lift"): active_flags.append(("UNSAFE LIFT", "red"))
                        if getattr(ps, "microsleep", False): active_flags.append(("MICROSLEEP", "red"))
                        if getattr(ps, "drowsy", False): active_flags.append(("DROWSY", "orange"))
                        if getattr(ps, "yawn_detected", False): active_flags.append(("YAWN", "yellow"))
                        if hasattr(ps, "zone_status") and str(ps.zone_status) == "ZoneStatus.RESTRICTED":
                            active_flags.append(("IN ZONE", "purple"))
                        elif hasattr(ps, "zone_status") and str(ps.zone_status) == "ZoneStatus.MACHINE_PROXIMITY":
                            active_flags.append(("NEAR MACHINE", "blue"))

                        if active_flags:
                            with ui.row().classes("w-full gap-1 mt-0.5"):
                                for flabel, fcolor in active_flags:
                                    ui.badge(flabel, color=fcolor).classes("text-[7px] px-1")

        # ── Trend (1/sec) ──
        if now - _last_trend[0] >= 1.0:
            _last_trend[0] = now
            trend_data["times"].append(time.strftime("%H:%M:%S"))
            trend_data["safety"].append(round(score, 1))
            trend_data["fatigue"].append(round(avg_fatigue, 1))
            trend_data["hr"].append(round(avg_hr, 1))
            for k in trend_data:
                if len(trend_data[k]) > TREND_HISTORY:
                    trend_data[k] = trend_data[k][-TREND_HISTORY:]
            trend_chart.options["xAxis"]["data"] = trend_data["times"]
            trend_chart.options["series"][0]["data"] = trend_data["safety"]
            trend_chart.options["series"][1]["data"] = trend_data["fatigue"]
            if len(trend_chart.options["series"]) > 2:
                trend_chart.options["series"][2]["data"] = trend_data["hr"]
            trend_chart.update()

        # ── Alerts ──
        cnt = len(alert_manager.history)
        if cnt != _prev_alerts["count"]:
            new = cnt - _prev_alerts["count"]
            _prev_alerts["count"] = cnt
            alert_count_label.set_text(str(cnt))

            for a in alert_manager.get_recent(min(new, 3)):
                # Only show microsleep, posture, collision popups
                if a["rule"] not in ("fatigue", "unsafe_lift", "collision"):
                    continue
                sev = a["severity"]
                if sev in ("CRITICAL", "WARNING"):
                    # Big centered notification for important alerts
                    ui.notify(a["message"],
                              type={"CRITICAL": "negative", "WARNING": "warning"}.get(sev, "info"),
                              position="top", close_button=True,
                              timeout=8000 if sev == "CRITICAL" else 5000,
                              classes="text-lg font-bold",
                              multi_line=True)
                else:
                    ui.notify(a["message"],
                              type="info",
                              position="top-right", close_button=True,
                              timeout=3000)

            alert_box.clear()
            for a in alert_manager.get_recent(8):
                if a["rule"] not in ("fatigue", "unsafe_lift", "collision"):
                    continue
                css = {"CRITICAL": "a-crit", "WARNING": "a-warn", "INFO": "a-info"}.get(a["severity"], "a-info")
                with alert_box:
                    with ui.row().classes(f"w-full items-center gap-2 alert-row {css}"):
                        ui.label(a["time"]).classes("text-[10px] text-slate-500 font-mono")
                        ui.badge(a["severity"], color={"CRITICAL": "red", "WARNING": "yellow", "INFO": "blue"}.get(
                            a["severity"], "gray")).classes("text-[9px]")
                        ui.label(a["message"]).classes("text-[11px] flex-1")

        # Capture notification + gallery update
        cap = latest_state.get("last_capture", "")
        if cap and cap != _prev_alerts.get("cap", ""):
            _prev_alerts["cap"] = cap
            ui.notify(f"Incident captured: {cap}", type="negative", position="bottom-right", timeout=4000)
            if not _show_camera[0]:
                _update_gallery_full()

        # Refresh gallery every 10 seconds (only when camera is off)
        if not _show_camera[0] and now - _prev_alerts.get("gallery_refresh", 0) > 10:
            _prev_alerts["gallery_refresh"] = now
            _update_gallery_full()

    def _update_gallery_full():
        """Refresh the large incident gallery (shown when camera is off)."""
        try:
            files = sorted(
                [f for f in os.listdir(CAPTURE_DIR) if f.endswith(".jpg")],
                reverse=True
            )[:40]  # show last 40 in full view
        except OSError:
            files = []

        gallery_full_count.set_text(str(len(files)))
        gallery_full_grid.clear()
        with gallery_full_grid:
            if not files:
                ui.label("No incident captures yet. Captures appear automatically when safety drops or drowsiness is detected.").classes(
                    "text-xs text-slate-500 italic py-4"
                )
            else:
                for fname in files:
                    parts = fname.replace(".jpg", "").split("_")
                    reason = parts[3] if len(parts) > 3 else "?"
                    score_str = parts[4] if len(parts) > 4 else ""
                    score_num = score_str.replace("score", "") if "score" in score_str else ""
                    date_str = parts[1] if len(parts) > 1 else ""
                    time_str = parts[2] if len(parts) > 2 else ""
                    time_fmt = f"{time_str[:2]}:{time_str[2:4]}:{time_str[4:]}" if len(time_str) == 6 else time_str

                    reason_color = {"perclos": "#f87171", "safety": "#fb923c", "posture": "#c084fc", "collision": "#ef4444"}.get(reason, "#94a3b8")
                    reason_label = {"perclos": "DROWSY", "safety": "LOW SCORE", "posture": "BAD POSTURE", "collision": "COLLISION"}.get(reason, reason.upper())

                    with ui.column().classes("items-center gap-0.5 cursor-pointer").style(
                        f"min-width: 100px; border: 1px solid {reason_color}; border-radius: 8px; "
                        "padding: 4px; background: rgba(15,23,42,0.6); flex-shrink: 0;"
                    ).on("click", lambda f=fname: _open_capture_dialog(f)):
                        ui.image(f"/captures/{fname}").style(
                            "width: 100px; height: 66px; object-fit: cover; border-radius: 6px;"
                        )
                        ui.badge(reason_label, color="red" if reason == "perclos" else "orange").classes("text-[7px]")
                        with ui.row().classes("items-center gap-1"):
                            ui.label(time_fmt).classes("text-[7px] text-slate-500 font-mono")
                            if score_num:
                                ui.label(f"Score: {score_num}").classes("text-[7px] text-slate-500")

    def _open_capture_dialog(filename):
        """Open a dialog showing the full capture image."""
        with ui.dialog() as dlg:
            with ui.card().classes("gc p-3").style("max-width: 90vw; max-height: 90vh;"):
                ui.label(f"Incident: {filename}").classes("text-sm font-bold text-slate-200 mb-2")
                ui.image(f"/captures/{filename}").style(
                    "max-width: 80vw; max-height: 70vh; object-fit: contain; border-radius: 8px;"
                )
                ui.button("Close", on_click=dlg.close).props("color=indigo rounded flat").classes("mt-2")
        dlg.open()

    async def _coach(state):
        if _coaching_active.get("running"):
            return
        _coaching_active["running"] = True
        try:
            coach_spinner.set_visibility(True)
            tip = await safety_coach.maybe_coach(state)
            coach_spinner.set_visibility(False)
            if tip:
                coach_label.set_text(tip)
            crits = [a for a in state.active_alerts if a.severity == "CRITICAL"]
            if crits:
                incident_spinner.set_visibility(True)
                analysis = await safety_coach.analyze_incident(list(alert_manager.history)[-10:])
                incident_spinner.set_visibility(False)
                if analysis:
                    incident_label.set_text(analysis)
        except Exception:
            coach_spinner.set_visibility(False)
            incident_spinner.set_visibility(False)
        finally:
            _coaching_active["running"] = False

    # ── Cleanup ──
    app.on_shutdown(lambda: (camera.stop(), pose_analyzer.close(), face_analyzer.close()))

    # ── Start ──
    ui.timer(0.1, lambda: asyncio.create_task(vision_loop()), once=True)
    ui.timer(1 / UI_FPS, update_ui)


def build_worker_page(worker_id: int):
    """Full-page worker detail view with live-updating metrics."""
    ui.dark_mode(True)
    ui.colors(primary='#818cf8', positive='#4ade80', negative='#f87171', warning='#facc15')
    ui.add_head_html("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    body { font-family: 'Inter', system-ui, sans-serif; }
    .nicegui-content { padding: 0 !important; }
    .gc { background: rgba(30,41,59,0.7); border: 1px solid rgba(100,116,139,0.2);
          border-radius: 10px; padding: 12px; backdrop-filter: blur(8px); }
    .sc-safe { color: #4ade80; } .sc-caution { color: #facc15; }
    .sc-warning { color: #fb923c; } .sc-danger { color: #f87171; }
    </style>""")

    # Header with back navigation
    with ui.header().classes("items-center justify-between px-4 py-1").style(
        "background: #0a0f1a; border-bottom: 1px solid rgba(100,116,139,0.15); height: 44px;"
    ):
        with ui.row().classes("items-center gap-2"):
            ui.button(icon="arrow_back", on_click=lambda: ui.navigate.to("/")).props(
                "round dense flat color=white size=sm"
            ).tooltip("Back to Zone Overview")
            ui.icon("person", size="sm").classes("text-indigo-400")
            ui.label(f"Worker {worker_id + 1}").classes("text-sm font-bold tracking-wide")
        with ui.row().classes("items-center gap-2"):
            ui.element("span").style(
                "width:6px;height:6px;border-radius:50%;background:#4ade80;"
                "animation:plive 2s ease-in-out infinite;display:inline-block;"
            )
            ui.label("LIVE").classes("text-[8px] text-green-400 font-bold tracking-widest")

    # Main content
    with ui.column().classes("w-full p-4 gap-3").style("height: calc(100vh - 44px); overflow-y: auto;"):

        # Score header
        score_row = ui.row().classes("w-full items-center justify-between")
        with score_row:
            w_score_label = ui.label("--").classes("text-5xl font-black sc-safe")
            w_risk_badge = ui.badge("SAFE", color="green").classes("text-sm px-4 font-bold")

        ui.separator()

        # Metrics grid
        with ui.row().classes("w-full gap-3 flex-wrap"):
            def _metric_card(icon, label, default):
                with ui.card().classes("gc").style("min-width: 140px; flex: 1;"):
                    with ui.row().classes("items-center gap-1 mb-1"):
                        ui.icon(icon, size="xs").classes("text-slate-500")
                        ui.label(label).classes("text-[8px] text-slate-500 uppercase tracking-widest font-semibold")
                    v = ui.label(default).classes("text-xl font-bold text-slate-100 font-mono")
                return v

            w_hr = _metric_card("favorite", "Heart Rate", "--")
            w_hrv = _metric_card("monitor_heart", "HRV", "--")
            w_fatigue = _metric_card("bedtime", "Fatigue", "0")
            w_stress = _metric_card("mood_bad", "Stress", "0%")
            w_perclos = _metric_card("visibility", "PERCLOS", "0%")
            w_reba = _metric_card("fitness_center", "REBA", "1")

        with ui.row().classes("w-full gap-3 flex-wrap"):
            w_back = _metric_card("accessibility", "Back Angle", "0\u00b0")
            w_shoulder = _metric_card("straighten", "Shoulder Tilt", "0\u00b0")
            w_left_arm = _metric_card("pan_tool", "L Arm Extension", "0\u00b0")
            w_right_arm = _metric_card("pan_tool", "R Arm Extension", "0\u00b0")
            w_left_knee = _metric_card("directions_walk", "L Knee", "0\u00b0")
            w_right_knee = _metric_card("directions_walk", "R Knee", "0\u00b0")

        with ui.row().classes("w-full gap-3 flex-wrap"):
            w_gaze = _metric_card("center_focus_strong", "Gaze Focus", "100%")
            w_head_pitch = _metric_card("swap_vert", "Head Pitch", "0\u00b0")
            w_reps = _metric_card("repeat", "Repetitions", "0")

        # Active flags
        with ui.card().classes("gc w-full"):
            ui.label("ACTIVE FLAGS").classes("text-[8px] tracking-[0.2em] text-slate-500 font-semibold mb-2")
            w_flags_row = ui.row().classes("w-full gap-2 flex-wrap")
            with w_flags_row:
                ui.label("No active flags").classes("text-sm text-green-400")

    # Live update
    def update_worker():
        all_states = _shared.get("all_states", [])
        if worker_id >= len(all_states):
            w_score_label.set_text("--")
            w_risk_badge.set_text("OFFLINE")
            w_risk_badge.props("color=gray")
            return

        ps = all_states[worker_id]
        face = _shared.get("face", FaceResult()) if worker_id == 0 else FaceResult()
        hr_val, hrv_val = _shared.get("person_hrs", {}).get(worker_id, (72.0, 50.0))
        ps_level = ps.risk_level.value
        sc = {"SAFE": "sc-safe", "CAUTION": "sc-caution", "WARNING": "sc-warning", "DANGER": "sc-danger"}
        color_map = {"SAFE": "green", "CAUTION": "yellow", "WARNING": "orange", "DANGER": "red"}

        w_score_label.set_text(f"{ps.overall_safety_score:.0f}")
        w_score_label.classes(replace=sc.get(ps_level, ""))
        w_risk_badge.set_text(ps_level)
        w_risk_badge.props(f"color={color_map.get(ps_level, 'gray')}")

        w_hr.set_text(f"{hr_val:.0f} bpm")
        w_hrv.set_text(f"{hrv_val:.0f} ms")
        w_fatigue.set_text(f"{ps.fatigue_score:.0f}")
        w_stress.set_text(f"{ps.stress_level:.0%}")
        w_perclos.set_text(f"{ps.perclos:.0%}")
        w_reba.set_text(f"{ps.reba_score}")
        w_back.set_text(f"{ps.angles.get('back', 0):.0f}\u00b0")
        w_shoulder.set_text(f"{ps.angles.get('shoulder_tilt', 0):.1f}\u00b0")
        w_left_arm.set_text(f"{ps.angles.get('left_arm_extension', 0):.0f}\u00b0")
        w_right_arm.set_text(f"{ps.angles.get('right_arm_extension', 0):.0f}\u00b0")
        w_left_knee.set_text(f"{ps.angles.get('left_knee', 0):.0f}\u00b0")
        w_right_knee.set_text(f"{ps.angles.get('right_knee', 0):.0f}\u00b0")
        w_gaze.set_text(f"{ps.gaze_focus:.0%}")
        w_head_pitch.set_text(f"{getattr(ps, 'head_pitch', 0):.1f}\u00b0")
        w_reps.set_text(f"{ps.repetition_count}")

        # Flags
        active_flags = []
        if ps.pose_flags.get("unsafe_lift"): active_flags.append(("UNSAFE LIFT", "red"))
        if getattr(ps, "microsleep", False): active_flags.append(("MICROSLEEP", "red"))
        if getattr(ps, "drowsy", False): active_flags.append(("DROWSY", "orange"))
        if getattr(ps, "yawn_detected", False): active_flags.append(("YAWNING", "yellow"))

        w_flags_row.clear()
        with w_flags_row:
            if active_flags:
                for flabel, fcolor in active_flags:
                    ui.badge(flabel, color=fcolor).classes("text-sm px-3 py-1")
            else:
                ui.label("No active flags").classes("text-sm text-green-400")

    ui.timer(1 / UI_FPS, update_worker)
