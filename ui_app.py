import sys, time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import pandas as pd

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QSpinBox, QCheckBox, QGroupBox, QFrame,
    QTableWidget, QTableWidgetItem, QFileDialog, QMessageBox,
    QComboBox, QDialog, QFormLayout, QDialogButtonBox
)

# ==========================
# CONFIG
# ==========================
APP_TITLE = "AI Smart Face Attendance System"
FACE_SIZE = (200, 200)
DEFAULT_THRESHOLD = 60
COOLDOWN_SECONDS = 2.0
UNKNOWN_SAVE_COOLDOWN = 4.0
REG_SAMPLES_DEFAULT = 50
REG_BLUR_MIN = 35

PREVIEW_MAX_W = 860
PREVIEW_MAX_H = 500
PREVIEW_MIN_W = 640
PREVIEW_MIN_H = 360

ATT_COLS = ["record_id","date","time","person_id","name","event","confidence","operator","snapshot_path"]

# ==========================
# PATHS
# ==========================
DATASET_DIR = Path("dataset")
TRAINER_DIR = Path("trainer")
TRAINER_PATH = TRAINER_DIR / "trainer.yml"

LOG_DIR = Path("attendance_logs")
SNAP_MARKED = LOG_DIR / "snapshots" / "marked"
SNAP_UNKNOWN = LOG_DIR / "snapshots" / "unknown"

for d in [DATASET_DIR, TRAINER_DIR, LOG_DIR, SNAP_MARKED, SNAP_UNKNOWN]:
    d.mkdir(parents=True, exist_ok=True)

# ==========================
# OpenCV
# ==========================
FACE_CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
EYE_CASCADE = cv2.data.haarcascades + "haarcascade_eye.xml"
face_cascade = cv2.CascadeClassifier(FACE_CASCADE)
eye_cascade = cv2.CascadeClassifier(EYE_CASCADE)

# ==========================
# Helpers
# ==========================
def today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")

def now_time_str() -> str:
    return datetime.now().strftime("%H:%M:%S")

def now_ts() -> float:
    return time.time()

def attendance_path_for(date_str: str) -> Path:
    return LOG_DIR / f"attendance_{date_str}.csv"

def ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=ATT_COLS)
    for c in ATT_COLS:
        if c not in df.columns:
            df[c] = ""
    return df[ATT_COLS]

def load_day_df(date_str: str) -> pd.DataFrame:
    p = attendance_path_for(date_str)
    if not p.exists():
        return ensure_cols(pd.DataFrame())
    try:
        return ensure_cols(pd.read_csv(p))
    except Exception:
        return ensure_cols(pd.DataFrame())

def save_day_df(date_str: str, df: pd.DataFrame):
    ensure_cols(df).to_csv(attendance_path_for(date_str), index=False)

def export_all_combined_df() -> pd.DataFrame:
    frames = []
    for p in sorted(LOG_DIR.glob("attendance_*.csv")):
        try:
            frames.append(pd.read_csv(p))
        except Exception:
            pass
    if not frames:
        return ensure_cols(pd.DataFrame())
    return ensure_cols(pd.concat(frames, ignore_index=True))

def load_registered_map() -> dict[int, str]:
    mp = {}
    for p in DATASET_DIR.iterdir():
        if p.is_dir():
            parts = p.name.split("_", 1)
            if len(parts) == 2 and parts[0].isdigit():
                mp[int(parts[0])] = parts[1].replace("_", " ")
    return mp

def registered_df() -> pd.DataFrame:
    rows = []
    for p in DATASET_DIR.iterdir():
        if p.is_dir():
            parts = p.name.split("_", 1)
            if len(parts) == 2 and parts[0].isdigit():
                pid = int(parts[0])
                rows.append({"person_id": pid, "name": parts[1].replace("_"," "), "samples": len(list(p.glob("*.jpg")))})
    df = pd.DataFrame(rows)
    return df.sort_values("person_id") if not df.empty else df

def blur_score(gray_face: np.ndarray) -> float:
    return float(cv2.Laplacian(gray_face, cv2.CV_64F).var())

def frame_to_pixmap(frame_bgr: np.ndarray, w: int, h: int) -> QPixmap:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    hh, ww, ch = rgb.shape
    img = QImage(rgb.data, ww, hh, ch * ww, QImage.Format_RGB888)
    return QPixmap.fromImage(img).scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

def ai_popup(parent, title: str, text: str, icon=QMessageBox.Information):
    m = QMessageBox(parent)
    m.setWindowTitle(title)
    m.setText(text)
    m.setIcon(icon)
    m.setStyleSheet("""
        QMessageBox { background: #050816; }
        QLabel { color: #ffffff; font-size: 14px; font-weight: 900; }
        QPushButton {
            background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #00e5ff, stop:1 #7c3aed);
            color: #050816;
            padding: 8px 14px;
            border-radius: 10px;
            font-weight: 1000;
        }
    """)
    m.exec()

class ClickableCard(QFrame):
    clicked = Signal()
    def mousePressEvent(self, e):
        self.clicked.emit()
        super().mousePressEvent(e)

class BlinkGate:
    def __init__(self):
        self.had_eyes = False
        self.last_blink_ts = 0.0
    def update(self, gray_face, now_ts_: float) -> float:
        eyes = eye_cascade.detectMultiScale(gray_face, 1.2, 6, minSize=(20, 20))
        has_eyes = len(eyes) >= 1
        if has_eyes:
            self.had_eyes = True
        if self.had_eyes and not has_eyes:
            self.last_blink_ts = now_ts_
            self.had_eyes = False
        return self.last_blink_ts

class EditDialog(QDialog):
    def __init__(self, parent, row: dict):
        super().__init__(parent)
        self.setWindowTitle("Edit Record (All Fields)")
        self.resize(560, 380)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.in_record_id = QLineEdit(row.get("record_id","")); self.in_record_id.setReadOnly(True)
        self.in_date = QLineEdit(row.get("date","")); self.in_date.setReadOnly(True)
        self.in_time = QLineEdit(row.get("time","")); self.in_time.setReadOnly(True)

        self.in_person_id = QLineEdit(row.get("person_id",""))
        self.in_name = QLineEdit(row.get("name",""))

        self.in_event = QComboBox()
        self.in_event.addItems(["IN","OUT"])
        ev = str(row.get("event","IN")).upper()
        self.in_event.setCurrentText(ev if ev in ("IN","OUT") else "IN")

        self.in_conf = QLineEdit(str(row.get("confidence","")))
        self.in_operator = QLineEdit(row.get("operator",""))
        self.in_snap = QLineEdit(row.get("snapshot_path",""))

        form.addRow("Record ID", self.in_record_id)
        form.addRow("Date", self.in_date)
        form.addRow("Time", self.in_time)
        form.addRow("Person ID", self.in_person_id)
        form.addRow("Name", self.in_name)
        form.addRow("Event", self.in_event)
        form.addRow("Confidence", self.in_conf)
        form.addRow("Operator", self.in_operator)
        form.addRow("Snapshot Path", self.in_snap)

        layout.addLayout(form)
        btns = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def get_fields(self):
        pid = self.in_person_id.text().strip()
        if not pid.isdigit():
            raise ValueError("Person ID must be numeric.")
        try:
            conf = float(self.in_conf.text().strip())
        except Exception:
            conf = 0.0
        return {
            "person_id": int(pid),
            "name": self.in_name.text().strip(),
            "event": self.in_event.currentText(),
            "confidence": round(conf, 2),
            "operator": self.in_operator.text().strip(),
            "snapshot_path": self.in_snap.text().strip(),
        }

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1400, 860)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.on_frame)
        self.mode = None  # register/attendance

        self.reg_folder = None
        self.reg_needed = REG_SAMPLES_DEFAULT
        self.reg_taken = 0

        self.recognizer = None
        self.registered_map = load_registered_map()

        self.cooldown = {}
        self.last_unknown_ts = 0.0
        self.blink_gate = BlinkGate()

        # OUT gap hours (easy operation)
        self.last_in_time_ts = {}
        self.out_gap_hours = 0  # UI default 0 hours

        self.build_ui()
        self.apply_theme()
        self.load_model()
        self.refresh_all()

    def apply_theme(self):
        self.setStyleSheet("""
            QMainWindow { background:#050816; color:#ffffff; }
            QLabel { color:#ffffff; }
            QLabel#H1 { font-size:22px; font-weight:1000; color:#ffffff; }
            QLabel#H2 { font-size:18px; font-weight:1000; color:#ffffff; }
            QLabel#Lbl { font-size:13px; font-weight:900; color:#cfe7ff; }

            QTabBar::tab { background:#0b1026; color:#ffffff; padding:10px 14px; margin-right:6px;
                          border-top-left-radius:12px; border-top-right-radius:12px; font-weight:900; }
            QTabBar::tab:selected { background:#111a3a; border:1px solid #2b3b7a; }

            QLineEdit, QSpinBox, QComboBox {
                background:#0b1026; color:#ffffff;
                border:1px solid #2b3b7a; border-radius:12px; padding:9px;
                font-weight:800;
            }

            QPushButton {
                background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #00e5ff, stop:1 #7c3aed);
                color:#050816; border-radius:12px; padding:10px 12px; font-weight:1000;
            }
            QPushButton#danger {
                background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #ff3d71, stop:1 #ffb703);
                color:#050816;
            }
            QPushButton#secondary {
                background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #22c55e, stop:1 #00e5ff);
                color:#050816;
            }

            QFrame#Card { background:#0b1026; border:1px solid #2b3b7a; border-radius:16px; }
            QLabel#CardTitle { color:#9adfff; font-weight:900; }
            QLabel#CardValue { font-size:26px; font-weight:1000; color:white; }

            QLabel#Preview { background:#030615; border:1px solid #2b3b7a; border-radius:14px; color:#9adfff; font-weight:900; }

            QLabel#Banner {
                padding:10px; border-radius:12px;
                background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #00e5ff, stop:1 #7c3aed);
                color:#050816; font-weight:1000;
            }
            QLabel#StatusLine { padding:6px; color:#cfe7ff; font-weight:900; }

            QGroupBox { border:1px solid #2b3b7a; border-radius:14px; margin-top:10px; padding:10px; background:#0b1026; font-weight:900; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding:0 8px; }

            QTableWidget { background:#030615; border:1px solid #2b3b7a; border-radius:14px; gridline-color:#2b3b7a;
                           color:#ffffff; selection-background-color:#7c3aed; }
            QHeaderView::section { background:#0b1026; padding:8px; border:0; font-weight:900; color:#ffffff; }
        """)

    def build_ui(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Dashboard
        dash = QWidget()
        dl = QVBoxLayout(dash)

        head = QLabel("AI DASHBOARD • SMART FACE ATTENDANCE")
        head.setObjectName("H1")
        head.setStyleSheet("""
            padding:14px; border-radius:16px;
            background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #00e5ff, stop:0.5 #7c3aed, stop:1 #ffb703);
            color:#050816;
        """)
        dl.addWidget(head)

        cards = QHBoxLayout()
        self.card_reg = self.card("REGISTERED", "0", lambda: self.tabs.setCurrentIndex(1))
        self.card_in = self.card("IN TODAY", "0", lambda: self.tabs.setCurrentIndex(3))
        self.card_out = self.card("OUT TODAY", "0", lambda: self.tabs.setCurrentIndex(3))
        self.card_unknown = self.card("UNKNOWN TODAY", "0", lambda: self.tabs.setCurrentIndex(2))
        for c in [self.card_reg, self.card_in, self.card_out, self.card_unknown]:
            cards.addWidget(c)
        dl.addLayout(cards)

        dl.addWidget(QLabel("<b>Recent Activity (Today)</b>"))
        self.recent = QTableWidget(0, 5)
        self.recent.setHorizontalHeaderLabels(["Time","ID","Name","Event","Confidence"])
        self.recent.horizontalHeader().setStretchLastSection(True)
        dl.addWidget(self.recent)

        btn = QPushButton("⟳ REFRESH")
        btn.clicked.connect(self.refresh_all)
        dl.addWidget(btn)
        dl.addStretch(1)

        # Register
        reg = QWidget()
        rl = QHBoxLayout(reg)

        left = QVBoxLayout()
        t = QLabel("Register")
        t.setObjectName("H2")
        left.addWidget(t)

        self.reg_id = QLineEdit(); self.reg_id.setPlaceholderText("Numeric ID")
        self.reg_name = QLineEdit(); self.reg_name.setPlaceholderText("Full name")
        self.reg_samples = QSpinBox(); self.reg_samples.setRange(10, 200); self.reg_samples.setValue(REG_SAMPLES_DEFAULT)

        self.btn_reg_start = QPushButton("▶ START CAMERA")
        self.btn_reg_start.clicked.connect(lambda: self.start_camera("register"))
        self.btn_reg_capture = QPushButton("● START CAPTURE")
        self.btn_reg_capture.clicked.connect(self.start_capture)
        self.btn_train = QPushButton("⚙ TRAIN MODEL"); self.btn_train.setObjectName("secondary")
        self.btn_train.clicked.connect(self.train_model)
        self.btn_reg_stop = QPushButton("■ STOP"); self.btn_reg_stop.setObjectName("danger")
        self.btn_reg_stop.clicked.connect(self.stop_camera)

        lab = QLabel("ID"); lab.setObjectName("Lbl")
        lab2 = QLabel("Name"); lab2.setObjectName("Lbl")
        lab3 = QLabel("Samples"); lab3.setObjectName("Lbl")

        left.addWidget(lab); left.addWidget(self.reg_id)
        left.addWidget(lab2); left.addWidget(self.reg_name)
        left.addWidget(lab3); left.addWidget(self.reg_samples)
        left.addWidget(self.btn_reg_start)
        left.addWidget(self.btn_reg_capture)
        left.addWidget(self.btn_train)
        left.addWidget(self.btn_reg_stop)
        left.addStretch(1)

        right = QVBoxLayout()
        self.reg_preview = QLabel("REGISTRATION PREVIEW"); self.reg_preview.setObjectName("Preview")
        self.reg_preview.setMinimumSize(PREVIEW_MIN_W, PREVIEW_MIN_H)
        self.reg_preview.setMaximumSize(PREVIEW_MAX_W, PREVIEW_MAX_H)
        self.reg_preview.setAlignment(Qt.AlignCenter)

        self.reg_banner = QLabel("READY"); self.reg_banner.setObjectName("Banner"); self.reg_banner.setAlignment(Qt.AlignCenter)
        self.reg_status = QLabel("Status: -"); self.reg_status.setObjectName("StatusLine")

        right.addWidget(self.reg_preview, alignment=Qt.AlignLeft)
        right.addWidget(self.reg_banner)
        right.addWidget(self.reg_status)
        right.addStretch(1)

        rl.addLayout(left, 1)
        rl.addLayout(right, 2)

        # Attendance
        att = QWidget()
        al = QHBoxLayout(att)

        aleft = QVBoxLayout()
        t2 = QLabel("Attendance (AUTO IN/OUT)")
        t2.setObjectName("H2")
        aleft.addWidget(t2)

        self.operator = QLineEdit("Admin")
        self.threshold = QSpinBox(); self.threshold.setRange(30, 120); self.threshold.setValue(DEFAULT_THRESHOLD)
        self.chk_multi = QCheckBox("Multi-face mode"); self.chk_multi.setChecked(True)
        self.chk_liveness = QCheckBox("Require blink (optional)"); self.chk_liveness.setChecked(False)
        self.chk_unknown = QCheckBox("Save unknown snapshots"); self.chk_unknown.setChecked(True)

        cfg = QGroupBox("OUT Rule (Hours)")
        cl = QVBoxLayout(cfg)
        lh = QLabel("Allow OUT only after N hour(s) from IN"); lh.setObjectName("Lbl")
        self.out_gap_hours_spin = QSpinBox(); self.out_gap_hours_spin.setRange(0, 24); self.out_gap_hours_spin.setValue(0)
        cl.addWidget(lh)
        cl.addWidget(self.out_gap_hours_spin)

        self.btn_att_start = QPushButton("▶ START CAMERA")
        self.btn_att_start.clicked.connect(lambda: self.start_camera("attendance"))
        self.btn_att_stop = QPushButton("■ STOP"); self.btn_att_stop.setObjectName("danger")
        self.btn_att_stop.clicked.connect(self.stop_camera)

        self.banner = QLabel("READY"); self.banner.setObjectName("Banner"); self.banner.setAlignment(Qt.AlignCenter)

        lo = QLabel("Operator"); lo.setObjectName("Lbl")
        lt = QLabel("Threshold"); lt.setObjectName("Lbl")

        aleft.addWidget(lo); aleft.addWidget(self.operator)
        aleft.addWidget(cfg)
        aleft.addWidget(lt); aleft.addWidget(self.threshold)
        aleft.addWidget(self.chk_multi); aleft.addWidget(self.chk_liveness); aleft.addWidget(self.chk_unknown)
        aleft.addWidget(self.btn_att_start); aleft.addWidget(self.btn_att_stop)
        aleft.addWidget(self.banner)
        aleft.addStretch(1)

        aright = QVBoxLayout()
        self.att_preview = QLabel("ATTENDANCE PREVIEW"); self.att_preview.setObjectName("Preview")
        self.att_preview.setMinimumSize(PREVIEW_MIN_W, PREVIEW_MIN_H)
        self.att_preview.setMaximumSize(PREVIEW_MAX_W, PREVIEW_MAX_H)
        self.att_preview.setAlignment(Qt.AlignCenter)

        self.att_info = QLabel("Live: -"); self.att_info.setObjectName("StatusLine")
        aright.addWidget(self.att_preview, alignment=Qt.AlignLeft)
        aright.addWidget(self.att_info)
        aright.addStretch(1)

        al.addLayout(aleft, 1)
        al.addLayout(aright, 2)

        # Records
        rec = QWidget()
        rcl = QVBoxLayout(rec)

        top = QHBoxLayout()
        self.btn_ref = QPushButton("⟳ REFRESH"); self.btn_ref.clicked.connect(self.refresh_all)
        self.btn_exp_today = QPushButton("⬇ EXPORT TODAY"); self.btn_exp_today.clicked.connect(self.export_today)
        self.btn_exp_all = QPushButton("⬇ EXPORT ALL"); self.btn_exp_all.clicked.connect(self.export_all)
        self.btn_exp_reg = QPushButton("⬇ EXPORT REGISTERED"); self.btn_exp_reg.clicked.connect(self.export_registered)
        top.addWidget(self.btn_ref); top.addWidget(self.btn_exp_today); top.addWidget(self.btn_exp_all); top.addWidget(self.btn_exp_reg)
        top.addStretch(1)
        rcl.addLayout(top)

        self.table_today = QTableWidget(0, len(ATT_COLS))
        self.table_today.setHorizontalHeaderLabels([c.upper() for c in ATT_COLS])
        self.table_today.setColumnHidden(0, True)
        self.table_today.horizontalHeader().setStretchLastSection(True)

        self.table_all = QTableWidget(0, len(ATT_COLS))
        self.table_all.setHorizontalHeaderLabels([c.upper() for c in ATT_COLS])
        self.table_all.setColumnHidden(0, True)
        self.table_all.horizontalHeader().setStretchLastSection(True)

        actions = QHBoxLayout()
        self.btn_edit = QPushButton("✏ EDIT SELECTED"); self.btn_edit.setObjectName("secondary")
        self.btn_del = QPushButton("🗑 DELETE SELECTED"); self.btn_del.setObjectName("danger")
        self.btn_edit.clicked.connect(self.edit_selected)
        self.btn_del.clicked.connect(self.delete_selected)
        actions.addWidget(self.btn_edit); actions.addWidget(self.btn_del); actions.addStretch(1)

        rcl.addWidget(QLabel("<b>TODAY</b>"))
        rcl.addWidget(self.table_today)
        rcl.addLayout(actions)
        rcl.addWidget(QLabel("<b>ALL LOGS</b>"))
        rcl.addWidget(self.table_all)

        self.tabs.addTab(dash, "Dashboard")
        self.tabs.addTab(reg, "Register")
        self.tabs.addTab(att, "Attendance")
        self.tabs.addTab(rec, "Records")

    def card(self, title, value, on_click):
        c = ClickableCard()
        c.setObjectName("Card")
        lay = QVBoxLayout(c)
        t = QLabel(title); t.setObjectName("CardTitle")
        v = QLabel(value); v.setObjectName("CardValue")
        c.value = v
        lay.addWidget(t); lay.addWidget(v)
        c.clicked.connect(on_click)
        return c

    # ---------- Model
    def load_model(self):
        self.registered_map = load_registered_map()
        if TRAINER_PATH.exists():
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.recognizer.read(str(TRAINER_PATH))
        else:
            self.recognizer = None

    def train_model(self):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        faces, labels = [], []
        for person_folder in DATASET_DIR.iterdir():
            if not person_folder.is_dir():
                continue
            parts = person_folder.name.split("_", 1)
            if len(parts) != 2 or not parts[0].isdigit():
                continue
            label = int(parts[0])
            for img_path in person_folder.glob("*.jpg"):
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    faces.append(img); labels.append(label)
        if not faces:
            ai_popup(self, "TRAINING FAILED", "No face images found. Register first.", QMessageBox.Warning)
            return
        recognizer.train(faces, np.array(labels))
        recognizer.save(str(TRAINER_PATH))
        self.load_model()
        ai_popup(self, "MODEL READY", "Model trained successfully.")
        self.refresh_all()

    # ---------- Camera
    def start_camera(self, mode):
        self.mode = mode
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            ai_popup(self, "CAMERA ERROR", "Could not open camera.", QMessageBox.Critical)
            self.cap = None
            return
        if not self.timer.isActive():
            self.timer.start(30)
        if mode == "register":
            self.reg_banner.setText("CAMERA STARTED")
            self.reg_status.setText("Status: Camera started")
        else:
            self.banner.setText("SCANNING...")

    def stop_camera(self):
        if self.timer.isActive():
            self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.banner.setText("STOPPED")
        self.reg_banner.setText("STOPPED")
        self.reg_status.setText("Status: Stopped")

    # ---------- Register capture
    def start_capture(self):
        sid = self.reg_id.text().strip()
        name = self.reg_name.text().strip()
        if not sid.isdigit() or len(name) < 2:
            ai_popup(self, "VALIDATION", "Enter numeric ID and valid Name.", QMessageBox.Warning)
            return
        self.reg_needed = int(self.reg_samples.value())
        self.reg_taken = 0
        folder = DATASET_DIR / f"{int(sid)}_{name.replace(' ', '_')}"
        folder.mkdir(exist_ok=True)
        self.reg_folder = folder
        self.reg_banner.setText("CAPTURING...")
        self.reg_status.setText("Status: Capturing samples...")

    # ---------- Auto IN/OUT with hour gap
    def get_today_state(self, pid: int):
        df = load_day_df(today_str())
        if df.empty:
            return False, False
        has_in = ((df["person_id"].astype(int) == pid) & (df["event"].astype(str) == "IN")).any()
        has_out = ((df["person_id"].astype(int) == pid) & (df["event"].astype(str) == "OUT")).any()
        return bool(has_in), bool(has_out)

    def decide_event(self, pid: int, now_ts_: float):
        has_in, has_out = self.get_today_state(pid)
        if not has_in and not has_out:
            return "IN", "First record today"
        if has_in and not has_out:
            gap_hours = int(self.out_gap_hours_spin.value())
            min_gap = gap_hours * 3600
            in_ts = self.last_in_time_ts.get(pid)
            if in_ts is not None and (now_ts_ - in_ts) < min_gap:
                return None, f"OUT after {gap_hours} hour(s) from IN"
            return "OUT", "Second record today"
        return None, "Already IN & OUT today"

    def append_mark(self, pid: int, name: str, conf: float, event: str, operator: str, snap: str):
        df = load_day_df(today_str())
        if not df.empty and ((df["person_id"].astype(int) == pid) & (df["event"].astype(str) == event)).any():
            return False
        row = {
            "record_id": str(uuid4()),
            "date": today_str(),
            "time": now_time_str(),
            "person_id": pid,
            "name": name,
            "event": event,
            "confidence": round(float(conf), 2),
            "operator": operator,
            "snapshot_path": snap
        }
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        save_day_df(today_str(), df)
        if event == "IN":
            self.last_in_time_ts[pid] = now_ts()
        return True

    # ---------- Edit/Delete
    def get_selected_row(self):
        table = self.table_today if self.table_today.currentRow() >= 0 else self.table_all
        r = table.currentRow()
        if r < 0:
            return None
        row = {}
        for i, col in enumerate(ATT_COLS):
            item = table.item(r, i)
            row[col] = item.text() if item else ""
        return row

    def edit_selected(self):
        row = self.get_selected_row()
        if not row:
            ai_popup(self, "EDIT", "Select a record first.", QMessageBox.Warning)
            return
        dlg = EditDialog(self, row)
        if dlg.exec() != QDialog.Accepted:
            return
        try:
            fields = dlg.get_fields()
        except Exception as e:
            ai_popup(self, "EDIT FAILED", str(e), QMessageBox.Warning)
            return
        date_ = row["date"]
        df = load_day_df(date_)
        mask = df["record_id"].astype(str) == str(row["record_id"])
        if not mask.any():
            ai_popup(self, "EDIT", "Record not found in file.", QMessageBox.Warning)
            return
        for k, v in fields.items():
            df.loc[mask, k] = v
        save_day_df(date_, df)
        ai_popup(self, "UPDATED", "Record updated successfully.")
        self.refresh_all()

    def delete_selected(self):
        row = self.get_selected_row()
        if not row:
            ai_popup(self, "DELETE", "Select a record first.", QMessageBox.Warning)
            return
        ans = QMessageBox.question(self, "Confirm Delete", "Delete selected record?", QMessageBox.Yes | QMessageBox.No)
        if ans != QMessageBox.Yes:
            return
        date_ = row["date"]
        df = load_day_df(date_)
        df = df[df["record_id"].astype(str) != str(row["record_id"])].reset_index(drop=True)
        save_day_df(date_, df)
        ai_popup(self, "DELETED", "Record deleted successfully.")
        self.refresh_all()

    # ---------- Exports
    def export_today(self):
        df = load_day_df(today_str())
        if df.empty:
            ai_popup(self, "EXPORT", "No records today.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", f"attendance_{today_str()}.csv", "CSV Files (*.csv)")
        if not path:
            return
        df.to_csv(path, index=False)
        ai_popup(self, "EXPORT", f"Saved:\n{path}")

    def export_all(self):
        df = export_all_combined_df()
        if df.empty:
            ai_popup(self, "EXPORT", "No logs found.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "attendance_all.csv", "CSV Files (*.csv)")
        if not path:
            return
        df.to_csv(path, index=False)
        ai_popup(self, "EXPORT", f"Saved:\n{path}")

    def export_registered(self):
        df = registered_df()
        if df.empty:
            ai_popup(self, "EXPORT", "No registered candidates.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "registered_candidates.csv", "CSV Files (*.csv)")
        if not path:
            return
        df.to_csv(path, index=False)
        ai_popup(self, "EXPORT", f"Saved:\n{path}")

    # ---------- Refresh
    def refresh_all(self):
        reg = registered_df()
        df_today = load_day_df(today_str())

        in_today = 0 if df_today.empty else int((df_today["event"].astype(str) == "IN").sum())
        out_today = 0 if df_today.empty else int((df_today["event"].astype(str) == "OUT").sum())
        unknown_today = len(list(SNAP_UNKNOWN.glob(f"{today_str()}_*.jpg")))

        self.card_reg.value.setText(str(len(reg)))
        self.card_in.value.setText(str(in_today))
        self.card_out.value.setText(str(out_today))
        self.card_unknown.value.setText(str(unknown_today))

        self.fill_table(self.table_today, df_today)
        self.fill_table(self.table_all, export_all_combined_df())

        # recent activity
        self.recent.setRowCount(0)
        if not df_today.empty:
            df2 = df_today.sort_values(by="time", ascending=False).head(10)
            for _, r in df2.iterrows():
                i = self.recent.rowCount()
                self.recent.insertRow(i)
                vals = [r["time"], r["person_id"], r["name"], r["event"], r["confidence"]]
                for c, v in enumerate(vals):
                    self.recent.setItem(i, c, QTableWidgetItem(str(v)))

    def fill_table(self, table, df):
        table.setRowCount(0)
        if df is None or df.empty:
            return
        df = ensure_cols(df).sort_values(by=["date","time"], ascending=False)
        for _, r in df.iterrows():
            i = table.rowCount()
            table.insertRow(i)
            for c, col in enumerate(ATT_COLS):
                table.setItem(i, c, QTableWidgetItem(str(r[col])))

    # ---------- Frame loop
    def on_frame(self):
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            return

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(90, 90))
        t = now_ts()

        if self.mode == "register":
            if self.reg_folder is not None and self.reg_taken < self.reg_needed:
                if len(faces) == 0:
                    self.reg_status.setText("Status: No face detected...")
                else:
                    x, y, w, h = max(faces, key=lambda b: b[2]*b[3])
                    face_img = cv2.resize(gray[y:y+h, x:x+w], FACE_SIZE)
                    if blur_score(face_img) >= REG_BLUR_MIN:
                        self.reg_taken += 1
                        cv2.imwrite(str(self.reg_folder / f"{self.reg_taken}.jpg"), face_img)
                        self.reg_status.setText(f"Captured successfully: {self.reg_taken}/{self.reg_needed}")
                    else:
                        self.reg_status.setText("Too blurry, hold still...")

                    if self.reg_taken >= self.reg_needed:
                        self.reg_banner.setText("CAPTURE COMPLETED ✅")
                        ai_popup(self, "Capture Completed", "Captured successfully. Now click TRAIN MODEL.")
                        self.refresh_all()

            for (x, y, w, h) in faces:
                cv2.rectangle(display, (x, y), (x+w, y+h), (0,255,0), 2)

            self.reg_preview.setPixmap(frame_to_pixmap(display, self.reg_preview.width(), self.reg_preview.height()))
            return

        if self.mode == "attendance":
            if self.recognizer is None:
                self.banner.setText("MODEL NOT TRAINED")
                self.att_info.setText("Register + Train model first.")
                self.att_preview.setPixmap(frame_to_pixmap(display, self.att_preview.width(), self.att_preview.height()))
                return

            threshold = int(self.threshold.value())
            operator = self.operator.text().strip() or "Operator"
            live = []
            marked_any = False

            for (x, y, w, h) in faces:
                face_img = cv2.resize(gray[y:y+h, x:x+w], FACE_SIZE)
                label, dist = self.recognizer.predict(face_img)
                conf = max(0.0, min(100.0, 100.0 - float(dist)))

                if dist < threshold:
                    pid = int(label)
                    name = self.registered_map.get(pid, f"ID {pid}")
                    live.append(f"{name}({pid})")

                    event, reason = self.decide_event(pid, t)
                    if event is None:
                        cv2.rectangle(display, (x, y), (x+w, y+h), (0,165,255), 2)
                        cv2.putText(display, reason, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,165,255), 2)
                        continue

                    if self.chk_liveness.isChecked():
                        blink_ts = self.blink_gate.update(face_img, t)
                        if t - blink_ts > 3.0:
                            self.banner.setText("BLINK REQUIRED")
                            continue

                    key = (pid, event)
                    if t - self.cooldown.get(key, 0.0) < COOLDOWN_SECONDS:
                        continue

                    snap = str(SNAP_MARKED / f"{today_str()}_{now_time_str().replace(':','')}_ID{pid}_{event}.jpg")
                    cv2.imwrite(snap, frame)

                    if not self.append_mark(pid, name, conf, event, operator, snap):
                        continue

                    self.cooldown[key] = t
                    marked_any = True

                    cv2.rectangle(display, (x, y), (x+w, y+h), (0,255,0), 2)
                    cv2.putText(display, f"{name} {event} conf:{conf:.1f}", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2)

                    self.banner.setText("ATTENDANCE TAKEN SUCCESSFULLY ✅")
                    self.att_info.setText(f"{event} marked: {name} (ID {pid})")

                    ai_popup(self, "Attendance Taken Successfully",
                             f"{event} marked for:\n\nName: {name}\nID: {pid}\nTime: {now_time_str()}\nDate: {today_str()}",
                             QMessageBox.Information)
                    QApplication.beep()
                    self.refresh_all()

                    if not self.chk_multi.isChecked():
                        break
                else:
                    cv2.rectangle(display, (x, y), (x+w, y+h), (255,80,80), 2)
                    cv2.putText(display, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,80,80), 2)

            if len(faces) == 0:
                self.banner.setText("NO FACE DETECTED")
                self.att_info.setText("Live: -")
            else:
                if not marked_any:
                    self.banner.setText("SCANNING...")
                if live:
                    self.att_info.setText("Live: " + ", ".join(live[:6]))

                if self.chk_unknown.isChecked() and not marked_any and (t - self.last_unknown_ts) > UNKNOWN_SAVE_COOLDOWN:
                    unknown_path = SNAP_UNKNOWN / f"{today_str()}_{now_time_str().replace(':','')}_unknown.jpg"
                    cv2.imwrite(str(unknown_path), frame)
                    self.last_unknown_ts = t
                    self.refresh_all()

            self.att_preview.setPixmap(frame_to_pixmap(display, self.att_preview.width(), self.att_preview.height()))

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()