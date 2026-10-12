import os
import sqlite3
import time
from datetime import datetime
from functools import wraps
from io import BytesIO

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory, session
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.getenv("DATABASE_PATH", os.path.join(BASE_DIR, "ecotrack.db"))
app = Flask(__name__, static_folder=None)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-this-before-deployment")
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", "").strip()  # e.g. sustainability-issues/3
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "").strip()
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))
ADMIN_SIGNUP_CODE = os.getenv("ADMIN_SIGNUP_CODE", "change-admin-code")
ALLOWED_STATUSES = {"Open", "In Progress", "Resolved"}
PRIVATE_AREAS = {"washroom", "restroom", "toilet", "changing room", "private area"}

def now_text():
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with db() as con:
        con.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL, role TEXT NOT NULL, created_at TEXT NOT NULL)""")
        con.execute("""CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT, area TEXT NOT NULL, problem TEXT NOT NULL,
            severity TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'Open', confidence REAL,
            source TEXT NOT NULL, description TEXT DEFAULT '', time TEXT NOT NULL)""")
        con.execute("""CREATE TABLE IF NOT EXISTS readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT, resource TEXT NOT NULL, value REAL NOT NULL,
            unit TEXT NOT NULL, time TEXT NOT NULL)""")

def login_required(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            return jsonify({"error": "Please sign up or log in first."}), 401
        return fn(*args, **kwargs)
    return wrapped

def private_area(area):
    return (area or "").strip().lower() in PRIVATE_AREAS

def advice_for(label):
    s = label.lower()
    if any(x in s for x in ("leak", "water", "overflow", "tap", "spill")):
        return "Ask maintenance to inspect the source; shut off water only if safe."
    if any(x in s for x in ("litter", "trash", "waste", "bin", "garbage")):
        return "Arrange waste clearance and check whether waste segregation is needed."
    if any(x in s for x in ("light", "fan", "electric", "energy")):
        return "Verify the room is unoccupied and follow campus electrical safety procedures."
    return "Ask campus maintenance staff to inspect and verify this potential issue."

def severity_for(label, confidence):
    s = label.lower()
    if any(x in s for x in ("leak", "overflow", "water")):
        return "High" if confidence >= 0.75 else "Medium"
    if any(x in s for x in ("litter", "trash", "waste", "bin")):
        return "Medium"
    return "Low" if confidence < 0.75 else "Medium"

def call_roboflow(image_bytes, filename, mimetype):
    if not MODEL_ID or not ROBOFLOW_API_KEY:
        raise RuntimeError("AI model is not configured. Set ROBOFLOW_MODEL_ID and ROBOFLOW_API_KEY in .env.")
    endpoint = f"https://detect.roboflow.com/{MODEL_ID}"
    response = requests.post(
        endpoint,
        params={"api_key": ROBOFLOW_API_KEY, "confidence": int(CONFIDENCE_THRESHOLD * 100), "overlap": 30},
        files={"file": (secure_filename(filename) or "frame.jpg", BytesIO(image_bytes), mimetype or "image/jpeg")},
        timeout=35,
    )
    if response.status_code != 200:
        raise RuntimeError(f"AI provider returned HTTP {response.status_code}: {response.text[:250]}")
    data = response.json()
    predictions = data.get("predictions", [])
    if not predictions:
        return None, 0.0, data
    # Use the highest-confidence detection; custom model labels define which problems can be found.
    best = max(predictions, key=lambda p: float(p.get("confidence", 0)))
    return str(best.get("class", "sustainability issue")), float(best.get("confidence", 0)), data

@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")

@app.get("/api/health")
def health():
    return jsonify({"ok": True, "model_configured": bool(MODEL_ID and ROBOFLOW_API_KEY),
                    "provider": "Roboflow Hosted Inference" if MODEL_ID and ROBOFLOW_API_KEY else None})

@app.post("/api/auth")
def auth():
    data = request.get_json(silent=True) or {}
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))
    role = str(data.get("role", "User"))
    mode = str(data.get("mode", "login"))
    name = str(data.get("name", "")).strip() or email.split("@")[0]
    if not email or len(password) < 8:
        return jsonify({"error": "Enter an email and a password of at least 8 characters."}), 400
    if role not in {"User", "Maintenance", "Admin"}:
        return jsonify({"error": "Invalid role."}), 400
    if mode == "signup":
        if role == "Admin" and data.get("admin_code") != ADMIN_SIGNUP_CODE:
            return jsonify({"error": "Invalid admin signup code."}), 403
        try:
            with db() as con:
                cur = con.execute("INSERT INTO users(name,email,password_hash,role,created_at) VALUES(?,?,?,?,?)",
                    (name, email, generate_password_hash(password), role, now_text()))
                user_id = cur.lastrowid
        except sqlite3.IntegrityError:
            return jsonify({"error": "This email already has an account. Please log in."}), 409
    elif mode == "login":
        with db() as con:
            user = con.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        if not user or not check_password_hash(user["password_hash"], password):
            return jsonify({"error": "Incorrect email or password. If new, use Sign Up first."}), 401
        if user["role"] != role:
            return jsonify({"error": f"This account uses the {user['role']} role. Select that role to log in."}), 403
        user_id = user["id"]
    else:
        return jsonify({"error": "Unknown authentication mode."}), 400
    session.clear()
    session["user_id"] = user_id
    session["role"] = role
    session["email"] = email
    return jsonify({"ok": True, "role": role, "email": email})

@app.post("/api/logout")
def api_logout():
    session.clear()
    return jsonify({"ok": True})

@app.get("/api/reports")
@login_required
def get_reports():
    with db() as con:
        rows = con.execute("SELECT * FROM reports ORDER BY id ASC").fetchall()
    return jsonify([dict(r) for r in rows])

@app.patch("/api/reports/<int:report_id>")
@login_required
def update_report(report_id):
    data = request.get_json(silent=True) or {}
    status = data.get("status")
    if status not in ALLOWED_STATUSES:
        return jsonify({"error": "Invalid status."}), 400
    if session.get("role") not in {"Maintenance", "Admin"}:
        return jsonify({"error": "Only Maintenance or Admin accounts can update report status."}), 403
    with db() as con:
        cur = con.execute("UPDATE reports SET status=? WHERE id=?", (status, report_id))
        if cur.rowcount == 0:
            return jsonify({"error": "Report not found."}), 404
    return jsonify({"ok": True})

@app.patch("/api/reports/<int:report_id>/description")
@login_required
def update_description(report_id):
    data = request.get_json(silent=True) or {}
    with db() as con:
        con.execute("UPDATE reports SET description=? WHERE id=?", (str(data.get("description", ""))[:1000], report_id))
    return jsonify({"ok": True})

@app.post("/api/detect")
@login_required
def detect():
    uploaded = request.files.get("image")
    area = str(request.form.get("area", "Other")).strip()[:100]
    source = str(request.form.get("source", "upload")).strip()
    if private_area(area) and source == "camera":
        return jsonify({"error": "Camera detection is disabled for private areas. You may submit a non-camera photo report if campus policy permits."}), 400
    if not uploaded:
        return jsonify({"error": "Upload an image or capture a camera frame."}), 400
    if uploaded.mimetype not in {"image/jpeg", "image/png", "image/webp"}:
        return jsonify({"error": "Only JPG, PNG, or WEBP images are supported."}), 415
    image_bytes = uploaded.read()
    if not image_bytes:
        return jsonify({"error": "The uploaded image is empty."}), 400
    if len(image_bytes) > 8 * 1024 * 1024:
        return jsonify({"error": "Image exceeds 8 MB."}), 413
    try:
        label, confidence, raw = call_roboflow(image_bytes, uploaded.filename or "image.jpg", uploaded.mimetype)
    except (requests.RequestException, RuntimeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 502
    detected = bool(label and confidence >= CONFIDENCE_THRESHOLD)
    report_id = None
    problem = label if detected else None
    if detected:
        # Camera detections are rate-limited per area/label to reduce repeated alerts from each frame.
        if source == "camera":
            with db() as con:
                recent = con.execute("""SELECT id FROM reports WHERE area=? AND problem=? AND source='camera'
                    AND time >= datetime('now','-1 minute') ORDER BY id DESC LIMIT 1""", (area, label)).fetchone()
                if recent:
                    report_id = recent["id"]
        if report_id is None:
            with db() as con:
                cur = con.execute("""INSERT INTO reports(area,problem,severity,status,confidence,source,time)
                    VALUES(?,?,?,?,?,?,?)""",
                    (area, label, severity_for(label, confidence), "Open", confidence, source, now_text()))
                report_id = cur.lastrowid
    return jsonify({"detected": detected, "model_configured": True, "area": area,
                    "problem": problem, "confidence": confidence if detected else 0,
                    "advice": advice_for(problem) if detected else "No matching issue was detected by the configured model. This does not prove the area is problem-free.",
                    "report_id": report_id, "predictions": raw.get("predictions", [])})

@app.post("/api/readings")
@login_required
def save_reading():
    data = request.get_json(silent=True) or {}
    resource = str(data.get("resource", ""))
    try:
        value = float(data.get("value"))
    except (TypeError, ValueError):
        return jsonify({"error": "Enter a numeric meter reading."}), 400
    if value < 0:
        return jsonify({"error": "Reading cannot be negative."}), 400
    unit = "litres" if resource.lower().startswith("water") else "kWh" if resource.lower().startswith("electricity") else None
    if not unit:
        return jsonify({"error": "Unsupported resource."}), 400
    with db() as con:
        con.execute("INSERT INTO readings(resource,value,unit,time) VALUES(?,?,?,?)", (resource, value, unit, now_text()))
    return jsonify({"ok": True, "resource": resource, "value": value, "unit": unit, "time": now_text()})

@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "Request too large. Maximum upload size is 8 MB."}), 413

if __name__ == "__main__":
    init_db()
    # Local development only. Use a production WSGI server and HTTPS for deployment.
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=False)
