# 🛡️ AImSafe

### Real-time factory floor safety co-pilot powered by computer vision, ergonomic scoring, and AI coaching.

`Python 3.11+` · `MediaPipe` · `NiceGUI` · `OpenCV` · `OpenRouter LLM` · `MQTT` · `ECharts`

Built for the **MERIThON 2026** hackathon — Industry 5.0 challenge.

---

## 🎯 Project Aim

**Problem:** Workplace injuries in manufacturing remain alarmingly common. Unsafe lifting, repetitive strain, fatigue-related accidents, and zone violations account for millions of lost workdays every year. Traditional safety monitoring relies on periodic human audits — which can't catch hazards as they happen.

**Our goal:** Build a **real-time, non-invasive safety co-pilot** that continuously watches over factory workers using nothing more than a standard webcam. AImSafe embodies the **Industry 5.0** vision — technology that works *alongside* humans, enhancing their safety and well-being rather than replacing them.

**What makes it different:**
- **No wearables required** — workers don't need to carry or wear any devices
- **No identity tracking** — the system doesn't know *who* you are, only *how* you're moving
- **Instant feedback** — hazards are detected in real time, not discovered in quarterly reports
- **AI that coaches, not punishes** — friendly, actionable tips ("bend your knees!") instead of cold warnings
- **Full automation** — from detection to alert to machine stop, the entire safety loop is closed

---

## 💡 What Is AImSafe?

AImSafe is a **computer-vision-based worker safety monitoring system** designed for factory floor environments. It watches a live camera feed and, in real time:

- **Tracks up to 5 workers simultaneously** with independent safety scoring per person
- **Detects unsafe postures** — bad lifting form, overreaching, sustained poor posture
- **Monitors drowsiness & fatigue** — eye closure (PERCLOS), microsleep, yawning, gaze drift
- **Detects worker-to-worker collision risk** — pairwise proximity alerts with visual warning lines
- **Scores ergonomic risk** — simplified REBA (Rapid Entire Body Assessment) score per worker
- **Fuses all signals** into a single **safety score (0–100)** per person
- **Delivers AI coaching** — natural-language safety tips via LLM when hazards are detected
- **Controls factory equipment** — publishes MQTT commands (e.g. stop conveyor, change lighting) on critical events
- **Captures incidents automatically** — screenshots saved when safety drops or drowsiness spikes

**Who it's for:** Factory supervisors, EHS (Environment Health & Safety) teams, and industrial workstation designers who need continuous, non-invasive safety monitoring.

---

## 🔬 How Tracking Works

AImSafe uses **Google's MediaPipe** machine learning framework to extract body and face landmarks from video frames. No images are stored, no faces are identified — the system only processes **geometric points** and **movement patterns** in real time.

### Body Tracking — MediaPipe Pose (33 Landmarks)

MediaPipe PoseLandmarker places **33 landmark points** on each person's body. AImSafe supports **multi-person tracking** (up to 5 simultaneous workers) using `num_poses=5`.

```
                  0 (nose)
                  |
          11 ----12        ← shoulders
         / |      | \
       13  |      |  14    ← elbows
       /   |      |   \
     15    |      |    16   ← wrists
           |      |
          23 ----24        ← hips
           |      |
          25     26        ← knees
           |      |
          27     28        ← ankles
```

From these 33 points, AImSafe computes **angles** and **positions** every frame:

| Measurement | How It's Computed | What It Detects |
|---|---|---|
| **Back angle** | Angle between mid-shoulder → mid-hip → vertical | Unsafe lifting (> 40° while hands below hips) |
| **Shoulder tilt** | Y-difference between left and right shoulder | Lateral lean / asymmetric loading |
| **Knee angle** | Hip → knee → ankle angle (both legs) | Whether knees are bent during lifting |
| **Arm extension** | Shoulder → elbow → wrist angle (both arms) | Overreaching beyond safe range |
| **Body centroid** | Average of left and right hip positions | Zone proximity & collision detection |

#### Unsafe Lift Detection

```
Good lift (back < 30°):        Bad lift (back > 40°):

    O  ← head                         O
    |                                 /
    |  ← back upright               /  ← back bending forward
    |                              /
   / \  ← knees bent             |
  /   \                          / \
```

**Rule:** Back angle > 40° AND hands below hip level → unsafe lift. Uses **temporal filtering** (8 consecutive frames ≈ 260ms) to avoid false positives from brief natural movements.

#### Repetitive Motion Detection

Tracks cyclical arm and body motion patterns over a **5-minute rolling window**. When motion cycles exceed **50 repetitions**, a repetitive strain risk alert fires.

```
Arm height
  ↑
  |   /\    /\    /\    /\
  |  /  \  /  \  /  \  /  \
  | /    \/    \/    \/    \
  +————————————————————————→ time
```

---

### Face Tracking — MediaPipe Face Mesh (468 Landmarks + 52 Blendshapes)

MediaPipe FaceLandmarker places **468 landmark points** across the face and outputs **52 blendshape scores** (neural-network-based facial expression coefficients). Each tracked worker gets their own independent `FaceAnalyzer` instance.

#### Eye Aspect Ratio (EAR)

The system tracks 6 points around each eye to compute the **Eye Aspect Ratio**:

```
        p2    p3
   p1              p4
        p6    p5

EAR = (dist(p2,p6) + dist(p3,p5)) / (2 × dist(p1,p4))
```

| EAR Value | Meaning |
|---|---|
| ~0.25–0.30 | Eyes open |
| < threshold | Eyes closed |
| < threshold for > 0.5s | Microsleep |

#### Adaptive Baseline Calibration

AImSafe doesn't use a fixed threshold — it **learns each person's open-eye EAR** during the first 60 frames (~2 seconds):

1. Collects EAR samples while head pitch < 15° (filtering out downward looks)
2. Uses the **75th percentile** as baseline (robust to blinks during warmup)
3. Sets threshold at **72% of baseline**, clamped to [0.15, 0.28]

This means the system adapts to different eye shapes and facial structures automatically.

#### PERCLOS (Percentage of Eye Closure)

Instead of binary open/closed, AImSafe computes a **continuous closure degree** (0.0–1.0) for each frame — how far EAR dropped below threshold — and averages it over a **900-frame sliding window (~30 seconds)**.

- **Drowsy:** PERCLOS > 20% (eyes are closed more than 20% of the time)
- **Auto-reset:** If eyes stay open for ~5 consecutive seconds, the PERCLOS window resets

#### Noise-Tolerant Microsleep Detection

Real webcam data is noisy. AImSafe uses an **8-frame sliding window** with ratio-based state transitions:

- **Enter closed state:** 75% of recent frames show closed eyes (6 out of 8)
- **Exit closed state:** Only 25% show closed eyes (2 out of 8)
- **Glitch tolerance:** Up to 2 "open" frames during a sustained closure don't reset the counter
- **Microsleep trigger:** Sustained closure > 2 seconds

```
Timeline:
  Normal blink:    ___/‾‾‾\___    (< 0.5s closed)
  Microsleep:      ___/‾‾‾‾‾‾‾‾‾\___    (> 2s closed) ← CRITICAL ALERT
```

#### Signal Fusion — Blendshape + Geometry

AImSafe fuses **two independent signals** for robust closure detection:

1. **Geometric EAR** — computed from raw landmark positions
2. **Neural blendshapes** — `eyeBlinkLeft` / `eyeBlinkRight` from MediaPipe's ML model

Closure is confirmed when **both signals agree**, OR when blendshape confidence > 0.7.

#### Pitch Gating

Looking down at work can make eyes appear partially closed. AImSafe suppresses false positives:

- **Normal closure detection:** Suppressed when head pitch > 20°
- **Microsleep detection:** Uses a relaxed 35° gate (people nod off with head tilted down)

#### Additional Face Metrics

| Metric | How It Works |
|---|---|
| **Stress level** | Composite of brow tension, worry, frown, and squint blendshapes with EMA smoothing |
| **Yawn detection** | `jawOpen` blendshape > 0.6 |
| **Gaze tracking** | Iris position (landmark 468+) relative to eye corners; variance-based focus score |
| **Head pose** | Pitch (nodding) and yaw (turning) estimated from nose/chin/forehead/eye-corner positions |

---

### Fatigue Scoring

All signals feed into a **composite fatigue score** (0–100) via the fatigue tracker:

| Signal | Contribution |
|---|---|
| PERCLOS > 20% | Proportional to closure percentage |
| Microsleep events | +2 per frame during microsleep |
| Yawn detected | Additive boost |
| High stress | Weighted contribution |
| Head nod (pitch spike > 25°) | Indicates drowsy head-drooping |
| Movement slowdown | Velocity drop vs. baseline |

The score uses **asymmetric EMA smoothing**: fast rise (α=0.15) so fatigue is detected quickly, slow decay (α=0.85) so the score doesn't drop too fast after a brief alert moment.

---

### Risk Engine — Score Fusion

Each tracked worker has their own independent `RiskEngine` instance. Every frame, all safety rules are evaluated and scores are **fused into a single weighted safety score**:

```
Safety Score = 100 − weighted risk sum

                    ┌─────────────────┐
  Pose Risk ──(×0.35)──┐              │
  Fatigue   ──(×0.30)──┼──→  Overall  │──→  0-100 Safety Score
  Proximity ──(×0.25)──┤    Score     │      + Risk Level
  Repetition──(×0.10)──┘              │      + Active Alerts
                    └─────────────────┘
```

| Safety Score | Level | Color | What Happens |
|---|---|---|---|
| ≥ 70 | **SAFE** | 🟢 Green | Normal operation |
| 40–69 | **CAUTION** | 🟡 Yellow | Gentle dashboard reminders |
| 20–39 | **WARNING** | 🟠 Orange | Audio alert, AI coaching tip |
| < 20 | **DANGER** | 🔴 Red | Critical alert, incident capture, MQTT machine stop |

**Area Safety Score** = average of all individual worker safety scores (displayed as the zone-wide metric).

---

### Collision Detection

AImSafe computes **pairwise Euclidean distance** between every pair of tracked workers' body centroids:

- **Alert threshold:** 0.12 normalized distance → collision warning alert (15s cooldown per pair)
- **Visual threshold:** 0.15 normalized distance → red warning lines drawn on video with "! CLOSE !" label

---

### Zone Management

Danger zones are defined as polygons in `data/zones.json` using normalized camera coordinates (0.0–1.0). The zone manager uses **ray-casting point-in-polygon** testing to check if a worker's centroid falls inside any zone.

Zone types: `RESTRICTED` (score +40), `MACHINE_PROXIMITY` (score +20), `SAFE_AREA` (no penalty).

---

### Ergonomic Assessment (REBA)

A simplified **REBA (Rapid Entire Body Assessment)** score (1–15) is computed from pose angles. REBA is a standard ergonomic risk assessment tool used in occupational health:

| REBA Score | Risk Level |
|---|---|
| 1 | Negligible |
| 2–3 | Low |
| 4–7 | Medium |
| 8–10 | High |
| 11–15 | Very High |

---

## 🤖 AI Features

Powered by **OpenRouter** using **Llama 3.1 8B Instruct** (primary) with **Mixtral 8x7B** (fallback).

| Feature | Trigger | Description |
|---|---|---|
| **Real-time coaching** | On risk events | Actionable safety tips (e.g., "bend your knees, not your back") — cooldown-gated, cached |
| **Incident analysis** | On CRITICAL alerts | Root cause analysis with structured output (what happened, why, corrective action) |
| **Shift reports** | On-demand button | AI summary of the entire session: trends, key incidents, recommendations |

All AI features **gracefully degrade** — the system works fully without an API key.

**Rate limiting:** Configurable max calls per minute (default: 10). Responses are cached to avoid duplicate queries.

---

## 🖥️ Dashboard

Single-page **NiceGUI** application with a dark industrial theme, designed to fit a single viewport.

### Layout

| Left Side (55%) | Right Side (45%) |
|---|---|
| Live camera feed with per-person overlays (skeleton, bounding box, HR label, collision lines) | Recent alerts feed |
| Incident gallery (mini strip / full gallery toggle) | AI Coach tips + Incident analysis panels |
| Zone safety trend chart (ECharts — safety + fatigue + HR over time) | Scrollable worker list with per-person cards |

### Zone Overview Bar (Top)

| Zone Safety Index | People Count | Avg Fatigue | Avg Stress | Avg REBA | Avg HR | Alert Count |
|---|---|---|---|---|---|---|
| EChart gauge (0–100) | Live count | Cross-worker avg | Cross-worker avg | Cross-worker avg | Cross-worker avg | Session total |

### Per-Worker Cards

Each worker card shows: safety score, risk badge, back angle, fatigue, PERCLOS, HR, REBA, stress — plus active flags (UNSAFE LIFT, MICROSLEEP, DROWSY, YAWN). Click "Full Details" for a dedicated `/worker/{id}` page with all vitals and body metrics.

### Incident Gallery

- Auto-captures screenshots when safety score < 35 or PERCLOS > 50%
- Saved to `captures/` with 30s cooldown between captures
- Mini horizontal strip view (camera on) or full grid view (camera off)

---

## 📡 IoT & Machine Integration

### MQTT Machine Control

On critical safety events, AImSafe publishes MQTT commands to control factory equipment:

| Topic | Example Payload | Trigger |
|---|---|---|
| `factory/conveyor/zone{n}` | `{"action": "stop"}` | DANGER-level risk |
| `factory/lighting/zone{n}` | `{"action": "set", "brightness": 100}` | WARNING-level risk |
| `factory/alert/zone{n}` | `{"action": "alarm", "level": "critical"}` | Any critical alert |

Uses **paho-mqtt**. MQTT is disabled by default (`MQTT_ENABLED=false`) and works without a broker.

### Wearable Simulation

Per-person heart rate (50–180 BPM) and HRV (10–100ms) simulation via stress-driven random walk. Each worker gets unique baselines. Designed as an integration-ready interface for real wearable sensors.

### Voice Alerts

Thread-safe offline text-to-speech via **pyttsx3** with priority queue. Critical messages can interrupt lower-priority announcements.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Computer Vision** | [MediaPipe Tasks API](https://developers.google.com/mediapipe) | Pose (33 landmarks) + Face (468 landmarks + 52 blendshapes) — model loaded via buffer for Windows compatibility |
| **Image Processing** | [OpenCV](https://opencv.org/) | Camera capture, frame processing, MJPEG encoding |
| **ML Math** | [NumPy](https://numpy.org/) | Angle computation, EMA smoothing, statistical operations |
| **Web Dashboard** | [NiceGUI](https://nicegui.io/) | Python-native web UI framework (built on FastAPI + Vue.js) — dark theme, responsive |
| **Backend** | [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) | ASGI server, MJPEG video streaming endpoint, WebSocket state push |
| **Charts** | [ECharts](https://echarts.apache.org/) | Gauge widgets, trend line charts (safety/fatigue/HR over time) |
| **LLM Intelligence** | [OpenAI SDK](https://github.com/openai/openai-python) → [OpenRouter](https://openrouter.ai/) | Async client targeting Llama 3.1 8B / Mixtral 8x7B for coaching, analysis, reports |
| **Machine Control** | [paho-mqtt](https://pypi.org/project/paho-mqtt/) | Publish conveyor stop / lighting / alarm commands to MQTT broker |
| **Audio / TTS** | [pyttsx3](https://pypi.org/project/pyttsx3/) | Offline text-to-speech for critical voice alerts |
| **HTTP Client** | [httpx](https://www.python-httpx.org/) | Async HTTP for LLM API calls |
| **Configuration** | [python-dotenv](https://pypi.org/project/python-dotenv/) | Load settings from `.env` file |
| **Testing** | [pytest](https://pytest.org/) + [pytest-asyncio](https://pypi.org/project/pytest-asyncio/) | Unit and async test suite |

### Python Libraries (requirements.txt)

```
nicegui>=1.4       # Web UI framework
opencv-python>=4.8 # Camera + image processing
mediapipe>=0.10    # Body + face landmark detection
numpy              # Numerical computation
fastapi            # Backend API + WebSocket
uvicorn            # ASGI server
plotly             # (Available) Plotting library
paho-mqtt          # MQTT publish/subscribe
plyer              # Desktop notifications
openai             # OpenRouter-compatible LLM client
httpx              # Async HTTP
python-dotenv      # .env config loading
pytest             # Testing framework
pytest-asyncio     # Async test support
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+**
- **Webcam** (optional — runs in demo mode with 3 synthetic workers if no camera found)
- **OpenRouter API key** (optional — only needed for AI coaching features)

### Install

```bash
cd safeguard-ai
pip install -r requirements.txt
```

### Configure

Create a `.env` file (or copy from `.env.example`):

```env
OPENROUTER_API_KEY=sk-or-v1-your-key-here   # optional — AI coaching
CAMERA_INDEX=0                                # webcam index
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8080
DARK_THEME=true
TTS_ENABLED=false                             # voice alerts
MQTT_ENABLED=false                            # machine control
```

See `config.py` for all available settings and their defaults.

### Launch

```bash
python main.py
```

Open **http://localhost:8080**. If no webcam is detected, the system automatically enters **demo mode** with animated synthetic data for 3 workers.

### Run Tests

```bash
python -m pytest tests/ -v
```

---

## 📁 Project Structure

```
safeguard-ai/
├── main.py                          # Entry point — NiceGUI app (routes: / and /worker/{id})
├── config.py                        # All settings loaded from .env with defaults
├── requirements.txt                 # Python dependencies
├── pytest.ini                       # Test configuration
│
├── vision/                          # 👁️ Computer Vision
│   ├── camera.py                    #   Threaded OpenCV capture + automatic demo mode
│   ├── pose_analyzer.py             #   Multi-person pose (33 landmarks × 5 people)
│   ├── face_analyzer.py             #   Face mesh (468 landmarks + 52 blendshapes)
│   │                                #   → adaptive EAR, soft PERCLOS, microsleep,
│   │                                #     stress, gaze, yawn detection
│   └── hand_analyzer.py             #   Hand landmarks (available, not actively used)
│
├── engine/                          # ⚙️ Risk Assessment
│   ├── worker_state.py              #   WorkerState, Alert, RiskLevel, ZoneStatus dataclasses
│   ├── risk_engine.py               #   Per-person rule evaluation + weighted score fusion
│   ├── ergonomic_scorer.py          #   REBA scoring (1–15)
│   ├── fatigue_tracker.py           #   Rolling-window fatigue scoring (0–100)
│   ├── repetition_tracker.py        #   Cyclical motion counting (5-min window)
│   └── zone_manager.py              #   Danger zone polygons + ray-cast proximity
│
├── llm/                             # 🧠 LLM Intelligence
│   ├── openrouter_client.py         #   Async OpenRouter client (rate-limited, cached)
│   ├── safety_coach.py              #   Real-time coaching tips + incident analysis
│   ├── shift_reporter.py            #   On-demand AI shift summary reports
│   └── prompts.py                   #   All prompt templates (coaching, analysis, reports)
│
├── alerts/
│   └── alert_manager.py             #   Alert history + dispatch to listeners
│
├── integration/
│   ├── mqtt_bridge.py               #   MQTT machine control publisher
│   └── wearable_sim.py              #   Per-person HR/HRV simulation
│
├── audio/
│   └── tts_engine.py                #   Offline TTS via pyttsx3 (priority queue)
│
├── ui/
│   └── dashboard.py                 #   NiceGUI dashboard — zone overview, worker list,
│                                    #   camera feed, incident gallery, worker detail pages
│
├── models/                          #   MediaPipe model files (.task) — gitignored
├── data/
│   └── zones.json                   #   Danger zone polygon definitions
├── captures/                        #   Auto-captured incident screenshots — gitignored
└── tests/                           #   Test suite (16 test files, pytest)
```

---

## 🏗️ Architecture

```
╔═══════════════════════════════════════════════════════════════╗
║                    AImSafe — Data Pipeline                   ║
╚═══════════════════════════════════════════════════════════════╝

  Camera (30 FPS)
       │
       ▼
  ┌─────────────────────────────────────────────┐
  │          VISION LAYER (~33ms/frame)          │
  │                                             │
  │  MediaPipe Pose     MediaPipe Face Mesh     │
  │  (33 landmarks      (468 landmarks          │
  │   × up to 5         + 52 blendshapes)       │
  │   people)                                   │
  │       │                    │                │
  │  back angle,          EAR, PERCLOS,         │
  │  shoulder tilt,       microsleep,           │
  │  knee angles,         stress, yawn,         │
  │  arm extension,       gaze, head pose       │
  │  body centroid                              │
  └──────────┬─────────────────┬────────────────┘
             │                 │
             ▼                 ▼
  ┌─────────────────────────────────────────────┐
  │           RISK ENGINE (per frame)            │
  │                                             │
  │  Safety Rules          Temporal Trackers    │
  │  • Unsafe lift         • Fatigue (5-min)    │
  │  • Overreach           • Repetition (5-min) │
  │  • Zone proximity      • REBA scoring       │
  │  • Collision                                │
  │       │                      │              │
  │       ▼                      ▼              │
  │  ┌─────────────────────────────────────┐    │
  │  │  WORKER STATE FUSION                │    │
  │  │  Pose(35%) + Fatigue(30%)           │    │
  │  │  + Proximity(25%) + Repetition(10%) │    │
  │  │  = Safety Score (0–100)             │    │
  │  └─────────────────────────────────────┘    │
  └──────────────────────┬──────────────────────┘
                         │
        ┌────────────────┼────────────────┬──────────┐
        ▼                ▼                ▼          ▼
  ┌──────────┐   ┌──────────────┐  ┌─────────┐  ┌───────┐
  │Dashboard │   │ Alert System │  │LLM Coach│  │ MQTT  │
  │(NiceGUI) │   │ + TTS Audio  │  │(Llama)  │  │Bridge │
  │+ Gallery │   │ + Captures   │  │+ Reports│  │       │
  └──────────┘   └──────────────┘  └─────────┘  └───────┘
```

### Threading Model

| Thread | Responsibility |
|---|---|
| **Main thread** | NiceGUI / Uvicorn async event loop |
| **Camera thread** | Dedicated thread for OpenCV capture (blocking I/O) |
| **Vision processing** | Runs on camera thread, pushes results to async queue |
| **Risk engine** | Async task, consumes from vision queue |
| **LLM calls** | Async (httpx), non-blocking, fire-and-forget with dashboard callback |
| **MQTT publishing** | Sync (fast, immediate publish on risk events) |
| **Wearable sim** | Async task, generates data at 1 Hz |

---

## 🔐 Privacy & Design Principles

- **Privacy by design** — no face recognition, no identity storage, no biometrics saved. Only anonymized geometric metrics are processed.
- **Offline-first** — TTS and all core safety features work without internet
- **Graceful degradation** — no camera = demo mode, no API key = no AI coaching, no MQTT = log-only
- **No custom ML models** — everything builds on pre-trained MediaPipe outputs using rules, angles, and statistics
- **Windows compatible** — MediaPipe model files are loaded via byte buffer to avoid Windows path resolution issues
- **Non-invasive** — the system exists for one purpose: **keeping people safe and comfortable while they work**

---

## 📄 License

Hackathon project — **MERIThON 2026**.
