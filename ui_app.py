import sys, time
from dataclasses import dataclass
from datetime import datetime, time as dtime
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
    QTableWidget, QTableWidgetItem, QFileDialog,
    QDialog, QFormLayout, QDialogButtonBox, QComboBox, QMessageBox
)

# ==========================
# CONFIG
# ==========================
APP_TITLE = "Smart Face Attendance System"
DEFAULT_THRESHOLD = 60
COOLDOWN_SECONDS = 2.0
UNKNOWN_SAVE_COOLDOWN = 4.0
REG_BLUR_MIN = 35
FACE_SIZE = (200, 200)

# Preview fixed size (prevents growth)
PREVIEW_W = 840
PREVIEW_H = 480

# ==========================
# PATHS (same as your project)
# ==========================
DATASET_DIR = Path("dataset")
TRAINER_DIR = Path("trainer")
TRAINER_PATH = TRAINER_DIR / "trainer.yml"

LOG_DIR = Path("attendance_logs")
SNAP_DIR = LOG_DIR / "snapshots"
SNAP_MARKED = SNAP_DIR / "marked"
SNAP_UNKNOWN = SNAP_DIR / "unknown"

for d in [DATASET_DIR, TRAINER_DIR, LOG_DIR, SNAP_DIR, SNAP_MARKED, SNAP_UNKNOWN]:
    d.mkdir(parents=True, exist_ok=True)

# ==========================
# OpenCV cascades
# ==========================
FACE_CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
EYE_CASCADE = cv2.data.haarcascades + "haarcascade_eye.xml"
face_cascade = cv2.CascadeClassifier(FACE_CASCADE)
eye_cascade = cv2.CascadeClassifier(EYE_CASCADE)

ATT_COLS = [
    "record_id", "date", "time", "person_id", "name",
    "event", "status", "confidence", "operator", "snapshot_path"
]

def today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")

def now_time_str() -> str:
    return datetime.now().strftime("%H:%M:%S")

def attendance_path_for(date_str: str) -> Path:
    return LOG_DIR / f"attendance_{date_str}.csv"

def ensure_att_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=ATT_COLS)
    for c in ATT_COLS:
        if c not in df.columns:
            df[c] = ""
    return df[ATT_COLS]

def load_day_df(date_str: str) -> pd.DataFrame:
    p = attendance_path_for(date_str)
    if not p.exists():
        return ensure_att_cols(pd.DataFrame())
    try:
        return ensure_att_cols(pd.read_csv(p))
    except Exception:
        return ensure_att_cols(pd.DataFrame())

def save_day_df(date_str: str, df: pd.DataFrame):
    df = ensure_att_cols(df)
    df.to_csv(attendance_path_for(date_str), index=False)

def export_all_combined_df() -> pd.DataFrame:
    frames = []
    for p in sorted(LOG_DIR.glob("attendance_*.csv")):
        try:
            frames.append(pd.read_csv(p))
        except Exception:
            pass
    if not frames:
        return ensure_att_cols(pd.DataFrame())
    return ensure_att_cols(pd.concat(frames, ignore_index=True))

def load_registered_map() -> dict[int, str]:
    mp: dict[int, str] = {}
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
                name = parts[1].replace("_", " ")
                samples = len(list(p.glob("*.jpg")))
                rows.append({"person_id": pid, "name": name, "samples": samples})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by="person_id")
    return df

def blur_score(gray_face: np.ndarray) -> float:
    return float(cv2.Laplacian(gray_face, cv2.CV_64F).var())

def frame_to_pixmap(frame_bgr: np.ndarray, target_w: int, target_h: int) -> QPixmap:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(img).scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

def parse_hhmm(hhmm: str) -> dtime:
    # hhmm like "16:00"
    hh, mm = hhmm.split(":")
    return dtime(int(hh), int(mm), 0)

def time_to_hhmm(t: dtime) -> str:
    return f"{t.hour:02d}:{t.minute:02d}"

class BlinkGate:
    def __init__(self):
        self.had_eyes = False
        self.last_blink_ts = 0.0

    def update(self, gray_face: np.ndarray, now_ts: float) -> float:
        eyes = eye_cascade.detectMultiScale(gray_face, scaleFactor=1.2, minNeighbors=6, minSize=(20, 20))
        has_eyes = len(eyes) >= 1
        if has_eyes:
            self.had_eyes = True
        if self.had_eyes and not has_eyes:
            self.last_blink_ts = now_ts
            self.had_eyes = False
        return self.last_blink_ts

class ClickableCard(QFrame):
    clicked = Signal()
    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)

def show_popup(parent, title: str, text: str, icon=QMessageBox.Information):
    m = QMessageBox(parent)
    m.setIcon(icon)
    m.setWindowTitle(title)
    m.setText(text)
    # Force readable popup text (fix)
    m.setStyleSheet("""
        QMessageBox { background: white; }
        QLabel { color: #0f172a; font-size: 14px; font-weight: 700; }
        QPushButton { background:#2563eb; color:white; padding:8px 14px; border-radius:10px; font-weight:800; }
        QPushButton:hover { background:#1d4ed8; }
    """)
    m.exec()

class EditDialog(QDialog):
    def __init__(self, parent, row: dict):
        super().__init__(parent)
        self.setWindowTitle("Edit Attendance Record")
        self.resize(540, 360)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.in_record_id = QLineEdit(str(row.get("record_id", ""))); self.in_record_id.setReadOnly(True)
        self.in_date = QLineEdit(str(row.get("date", ""))); self.in_date.setReadOnly(True)
        self.in_time = QLineEdit(str(row.get("time", ""))); self.in_time.setReadOnly(True)

        self.in_person_id = QLineEdit(str(row.get("person_id", "")))
        self.in_name = QLineEdit(str(row.get("name", "")))

        self.in_event = QComboBox()
        self.in_event.addItems(["IN", "OUT"])
        cur_ev = str(row.get("event", "IN")).upper()
        self.in_event.setCurrentText(cur_ev if cur_ev in ("IN", "OUT") else "IN")

        self.in_status = QLineEdit(str(row.get("status", "")))
        self.in_conf = QLineEdit(str(row.get("confidence", "")))
        self.in_operator = QLineEdit(str(row.get("operator", "")))
        self.in_snap = QLineEdit(str(row.get("snapshot_path", "")))

        form.addRow("Record ID", self.in_record_id)
        form.addRow("Date", self.in_date)
        form.addRow("Time", self.in_time)
        form.addRow("Person ID", self.in_person_id)
        form.addRow("Name", self.in_name)
        form.addRow("Event", self.in_event)
        form.addRow("Status", self.in_status)
        form.addRow("Confidence", self.in_conf)
        form.addRow("Operator", self.in_operator)
        form.addRow("Snapshot", self.in_snap)

        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def fields(self) -> dict:
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
            "status": self.in_status.text().strip(),
            "confidence": round(conf, 2),
            "operator": self.in_operator.text().strip(),
            "snapshot_path": self.in_snap.text().strip(),
        }

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1300, 820)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.on_frame)

        self.mode = None  # register/attendance

        self.reg_folder = None
        self.reg_needed = 50
        self.reg_taken = 0

        self.recognizer = None
        self.registered_map = load_registered_map()

        self.cooldown = {}  # (pid,event)->last_ts
        self.last_unknown_ts = 0.0

        self.blink_gate = BlinkGate()

        # Configurable time rules (UI)
        self.late_after = dtime(9, 10, 0)
        self.in_allowed_until = dtime(11, 0, 0)
        self.out_allowed_after = dtime(16, 0, 0)

        self.build_ui()
        self.apply_theme()
        self.load_model()
        self.refresh_all()

    def apply_theme(self):
        self.setStyleSheet("""
            QMainWindow { background:#070b18; color:#e5e7eb; }
            QLabel { color:#e5e7eb; }
            QTabBar::tab {
                background:#0f172a; color:#e5e7eb;
                padding:10px 14px; margin-right:6px;
                border-top-left-radius:12px; border-top-right-radius:12px;
                font-weight:900;
            }
            QTabBar::tab:selected { background:#111827; border:1px solid #334155; }

            QLineEdit, QSpinBox, QComboBox {
                background:#0b1220;
                color:#e5e7eb;
                border:1px solid #334155;
                border-radius:12px;
                padding:9px;
                font-weight:700;
            }

            QPushButton {
                background:#2563eb;
                color:white;
                border-radius:12px;
                padding:10px 12px;
                font-weight:900;
            }
            QPushButton:hover { background:#1d4ed8; }

            QPushButton#danger { background:#ef4444; }
            QPushButton#danger:hover { background:#dc2626; }

            QPushButton#secondary { background:#10b981; color:#052e24; }
            QPushButton#secondary:hover { background:#059669; color:#ecfdf5; }

            QFrame#Card {
                background:qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #0f172a, stop:1 #111827);
                border:1px solid #334155;
                border-radius:16px;
            }
            QLabel#CardTitle { color:#cbd5e1; font-weight:900; }
            QLabel#CardValue { font-size:26px; font-weight:1000; color:white; }

            QLabel#Preview {
                background:#020617;
                border:1px solid #334155;
                border-radius:14px;
                color:#cbd5e1;
                font-weight:900;
            }

            QLabel#BannerInfo {
                padding:10px;
                border-radius:12px;
                background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #0b3b74, stop:1 #2563eb);
                border:1px solid #1d4ed8;
                font-weight:1000;
                color:white;
            }

            QGroupBox {
                border:1px solid #334155;
                border-radius:14px;
                margin-top:10px;
                padding:10px;
                background:#0b1220;
                font-weight:900;
            }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding:0 8px; }

            QTableWidget {
                background:#061023;
                border:1px solid #334155;
                border-radius:14px;
                gridline-color:#334155;
                color:#e5e7eb;
                selection-background-color:#2563eb;
            }
            QHeaderView::section {
                background:#0f172a;
                padding:8px;
                border:0;
                font-weight:900;
                color:#e5e7eb;
            }
            QTableWidget::item { padding:6px; }
        """)

    def build_ui(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # DASHBOARD
        dash = QWidget()
        dl = QVBoxLayout(dash)

        header = QFrame()
        header.setObjectName("HeaderBar")
        header.setStyleSheet("""
            QFrame {
                background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #2563eb, stop:0.5 #7c3aed, stop:1 #f59e0b);
                border-radius:16px;
                padding:12px;
            }
        """)
        hl = QHBoxLayout(header)
        title = QLabel(APP_TITLE)
        title.setStyleSheet("font-size:22px; font-weight:1000; color:white;")
        hl.addWidget(title)
        hl.addStretch(1)
        self.btn_dash_refresh = QPushButton("Refresh")
        self.btn_dash_refresh.clicked.connect(self.refresh_all)
        hl.addWidget(self.btn_dash_refresh)
        dl.addWidget(header)

        cards = QHBoxLayout()
        self.card_reg = self.make_click_card("Registered", "0", "#22c55e", target_tab=1)
        self.card_in = self.make_click_card("IN Today", "0", "#3b82f6", target_tab=3)
        self.card_out = self.make_click_card("OUT Today", "0", "#f59e0b", target_tab=3)
        self.card_unknown = self.make_click_card("Unknown Today", "0", "#ef4444", target_tab=2)
        for c in [self.card_reg, self.card_in, self.card_out, self.card_unknown]:
            cards.addWidget(c)
        dl.addLayout(cards)

        tip = QLabel("Click the cards to navigate. Configure IN/OUT time rules in Attendance tab.")
        tip.setStyleSheet("color:#cbd5e1; font-weight:700; padding:8px;")
        dl.addWidget(tip)
        dl.addStretch(1)

        # REGISTER TAB
        reg = QWidget()
        rl = QHBoxLayout(reg)

        left = QVBoxLayout()
        left.addWidget(QLabel("<h2>Register</h2>"))

        self.reg_id = QLineEdit(); self.reg_id.setPlaceholderText("Numeric ID (e.g., 1)")
        self.reg_name = QLineEdit(); self.reg_name.setPlaceholderText("Full name")
        self.reg_samples = QSpinBox(); self.reg_samples.setRange(10, 200); self.reg_samples.setValue(50)

        self.btn_reg_start = QPushButton("Start Camera")
        self.btn_reg_start.clicked.connect(lambda: self.start_camera("register"))
        self.btn_reg_capture = QPushButton("Start Capture")
        self.btn_reg_capture.clicked.connect(self.start_capture)
        self.btn_train = QPushButton("Train Model"); self.btn_train.setObjectName("secondary")
        self.btn_train.clicked.connect(self.train_model)
        self.btn_reg_stop = QPushButton("Stop Camera"); self.btn_reg_stop.setObjectName("danger")
        self.btn_reg_stop.clicked.connect(self.stop_camera)

        left.addWidget(QLabel("ID")); left.addWidget(self.reg_id)
        left.addWidget(QLabel("Name")); left.addWidget(self.reg_name)
        left.addWidget(QLabel("Samples")); left.addWidget(self.reg_samples)
        left.addWidget(self.btn_reg_start)
        left.addWidget(self.btn_reg_capture)
        left.addWidget(self.btn_train)
        left.addWidget(self.btn_reg_stop)
        left.addStretch(1)

        right = QVBoxLayout()
        self.reg_preview = QLabel("Registration Preview")
        self.reg_preview.setObjectName("Preview")
        self.reg_preview.setFixedSize(PREVIEW_W, PREVIEW_H)
        self.reg_preview.setAlignment(Qt.AlignCenter)
        self.reg_status = QLabel("Status: -")
        self.reg_status.setStyleSheet("color:#cbd5e1; font-weight:700;")
        right.addWidget(self.reg_preview, alignment=Qt.AlignLeft)
        right.addWidget(self.reg_status)
        right.addStretch(1)

        rl.addLayout(left, 1)
        rl.addLayout(right, 2)

        # ATTENDANCE TAB
        att = QWidget()
        al = QHBoxLayout(att)

        aleft = QVBoxLayout()
        aleft.addWidget(QLabel("<h2>Attendance (Auto IN/OUT)</h2>"))

        self.operator = QLineEdit("Admin")
        self.threshold = QSpinBox(); self.threshold.setRange(30, 120); self.threshold.setValue(DEFAULT_THRESHOLD)
        self.chk_multi = QCheckBox("Multi-face mode (mark many)"); self.chk_multi.setChecked(True)
        self.chk_liveness = QCheckBox("Require blink (anti-proxy demo)"); self.chk_liveness.setChecked(False)
        self.chk_unknown = QCheckBox("Save unknown snapshots"); self.chk_unknown.setChecked(True)

        # Configurable time rules UI
        self.in_until = QLineEdit("11:00")
        self.out_after = QLineEdit("16:00")
        self.late_after_in = QLineEdit("09:10")

        self.btn_apply_rules = QPushButton("Apply Time Rules"); self.btn_apply_rules.setObjectName("secondary")
        self.btn_apply_rules.clicked.connect(self.apply_time_rules)

        self.btn_att_start = QPushButton("Start Camera")
        self.btn_att_start.clicked.connect(lambda: self.start_camera("attendance"))
        self.btn_att_stop = QPushButton("Stop Camera"); self.btn_att_stop.setObjectName("danger")
        self.btn_att_stop.clicked.connect(self.stop_camera)

        self.banner = QLabel("Ready")
        self.banner.setObjectName("BannerInfo")
        self.banner.setAlignment(Qt.AlignCenter)

        aleft.addWidget(QLabel("Operator")); aleft.addWidget(self.operator)

        rules = QGroupBox("Time Rules (Configurable)")
        rlay = QVBoxLayout(rules)
        rlay.addWidget(QLabel("IN allowed until (HH:MM)"))
        rlay.addWidget(self.in_until)
        rlay.addWidget(QLabel("OUT allowed after (HH:MM)"))
        rlay.addWidget(self.out_after)
        rlay.addWidget(QLabel("Late after (HH:MM)"))
        rlay.addWidget(self.late_after_in)
        rlay.addWidget(self.btn_apply_rules)
        aleft.addWidget(rules)

        aleft.addWidget(QLabel("Threshold")); aleft.addWidget(self.threshold)
        aleft.addWidget(self.chk_multi)
        aleft.addWidget(self.chk_liveness)
        aleft.addWidget(self.chk_unknown)
        aleft.addWidget(self.btn_att_start)
        aleft.addWidget(self.btn_att_stop)
        aleft.addWidget(self.banner)
        aleft.addStretch(1)

        aright = QVBoxLayout()
        self.att_preview = QLabel("Attendance Preview")
        self.att_preview.setObjectName("Preview")
        self.att_preview.setFixedSize(PREVIEW_W, PREVIEW_H)
        self.att_preview.setAlignment(Qt.AlignCenter)
        self.att_info = QLabel("Live: -")
        self.att_info.setStyleSheet("color:#cbd5e1; font-weight:800; padding:6px;")
        aright.addWidget(self.att_preview, alignment=Qt.AlignLeft)
        aright.addWidget(self.att_info)
        aright.addStretch(1)

        al.addLayout(aleft, 1)
        al.addLayout(aright, 2)

        # RECORDS TAB
        rec = QWidget()
        rcl = QVBoxLayout(rec)

        top = QHBoxLayout()
        self.btn_refresh_rec = QPushButton("Refresh")
        self.btn_refresh_rec.clicked.connect(self.refresh_all)
        self.btn_export_today = QPushButton("Export Today CSV")
        self.btn_export_today.clicked.connect(self.export_today)
        self.btn_export_all = QPushButton("Export All CSV")
        self.btn_export_all.clicked.connect(self.export_all)
        self.btn_export_reg = QPushButton("Export Registered CSV"); self.btn_export_reg.setObjectName("secondary")
        self.btn_export_reg.clicked.connect(self.export_registered)
        for b in [self.btn_refresh_rec, self.btn_export_today, self.btn_export_all, self.btn_export_reg]:
            top.addWidget(b)
        top.addStretch(1)
        rcl.addLayout(top)

        self.table_today = QTableWidget(0, 10)
        self.table_today.setHorizontalHeaderLabels(["RecordID","Date","Time","ID","Name","Event","Status","Conf","Operator","Snapshot"])
        self.table_today.setColumnHidden(0, True)
        self.table_today.horizontalHeader().setStretchLastSection(True)

        self.table_all = QTableWidget(0, 10)
        self.table_all.setHorizontalHeaderLabels(["RecordID","Date","Time","ID","Name","Event","Status","Conf","Operator","Snapshot"])
        self.table_all.setColumnHidden(0, True)
        self.table_all.horizontalHeader().setStretchLastSection(True)

        actions = QHBoxLayout()
        self.btn_edit = QPushButton("✏ Edit Selected"); self.btn_edit.setObjectName("secondary")
        self.btn_del = QPushButton("🗑 Delete Selected"); self.btn_del.setObjectName("danger")
        self.btn_edit.clicked.connect(self.edit_selected)
        self.btn_del.clicked.connect(self.delete_selected)
        actions.addWidget(self.btn_edit)
        actions.addWidget(self.btn_del)
        actions.addStretch(1)

        rcl.addWidget(QLabel("<b>Today</b>"))
        rcl.addWidget(self.table_today)
        rcl.addLayout(actions)
        rcl.addWidget(QLabel("<b>All Logs</b>"))
        rcl.addWidget(self.table_all)

        self.tabs.addTab(dash, "Dashboard")
        self.tabs.addTab(reg, "Register")
        self.tabs.addTab(att, "Attendance")
        self.tabs.addTab(rec, "Records")

    def make_click_card(self, title: str, value: str, accent: str, target_tab: int) -> ClickableCard:
        card = ClickableCard()
        card.setObjectName("Card")
        card.setStyleSheet(f"QFrame{{border-left:6px solid {accent};}}")
        l = QVBoxLayout(card)
        t = QLabel(title); t.setObjectName("CardTitle")
        v = QLabel(value); v.setObjectName("CardValue")
        card.value = v
        l.addWidget(t)
        l.addWidget(v)
        card.clicked.connect(lambda: self.tabs.setCurrentIndex(target_tab))
        return card

    # ---------- Validation
    def validate_register_inputs(self):
        sid = self.reg_id.text().strip()
        name = self.reg_name.text().strip()
        if not sid.isdigit():
            return False, "ID must be numeric."
        if len(name) < 2:
            return False, "Name must be at least 2 characters."
        if self.reg_samples.value() < 10:
            return False, "Samples must be at least 10."
        return True, "OK"

    def validate_model_ready(self):
        if self.recognizer is None:
            return False, "Model not trained. Register and Train first."
        if not self.registered_map:
            return False, "No registered persons found."
        return True, "OK"

    # ---------- Time rules apply
    def apply_time_rules(self):
        try:
            self.in_allowed_until = parse_hhmm(self.in_until.text().strip())
            self.out_allowed_after = parse_hhmm(self.out_after.text().strip())
            self.late_after = parse_hhmm(self.late_after_in.text().strip())
        except Exception:
            show_popup(self, "Invalid Time", "Please enter time in HH:MM format (e.g., 16:00).", QMessageBox.Warning)
            return
        show_popup(self, "Rules Updated", f"IN until: {time_to_hhmm(self.in_allowed_until)}\nOUT after: {time_to_hhmm(self.out_allowed_after)}\nLate after: {time_to_hhmm(self.late_after)}")

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
                if img is None:
                    continue
                faces.append(img)
                labels.append(label)
        if not faces:
            show_popup(self, "Training Failed", "No face images found. Register first.", QMessageBox.Warning)
            return
        recognizer.train(faces, np.array(labels))
        recognizer.save(str(TRAINER_PATH))
        self.load_model()
        show_popup(self, "Training", "Model trained successfully.")
        self.refresh_all()

    # ---------- Camera
    def start_camera(self, mode: str):
        if mode == "register":
            ok, msg = self.validate_register_inputs()
            if not ok:
                show_popup(self, "Validation", msg, QMessageBox.Warning)
                return

        self.mode = mode
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            show_popup(self, "Camera Error", "Could not open camera.", QMessageBox.Critical)
            self.cap = None
            return
        if not self.timer.isActive():
            self.timer.start(30)
        if mode == "register":
            self.reg_status.setText("Status: Camera started")
        else:
            ok, msg = self.validate_model_ready()
            if not ok:
                show_popup(self, "Model", msg, QMessageBox.Warning)
            self.banner.setText("Camera started")

    def stop_camera(self):
        if self.timer.isActive():
            self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.banner.setText("Stopped")
        self.reg_status.setText("Status: Stopped")

    # ---------- Register capture
    def start_capture(self):
        ok, msg = self.validate_register_inputs()
        if not ok:
            show_popup(self, "Validation", msg, QMessageBox.Warning)
            return
        pid = int(self.reg_id.text().strip())
        name = self.reg_name.text().strip()
        self.reg_needed = int(self.reg_samples.value())
        self.reg_taken = 0
        folder = DATASET_DIR / f"{pid}_{name.replace(' ', '_')}"
        folder.mkdir(exist_ok=True)
        self.reg_folder = folder
        self.reg_status.setText("Status: Capturing samples...")

    # ---------- Attendance: auto IN/OUT decision
    def get_person_state_today(self, pid: int) -> tuple[bool, bool]:
        df = load_day_df(today_str())
        if df.empty:
            return False, False
        has_in = ((df["person_id"].astype(int) == pid) & (df["event"].astype(str) == "IN")).any()
        has_out = ((df["person_id"].astype(int) == pid) & (df["event"].astype(str) == "OUT")).any()
        return bool(has_in), bool(has_out)

    def decide_event_auto(self, pid: int) -> tuple[str | None, str]:
        """
        Auto rules:
        - If no record today => IN (only if before IN allowed until)
        - If IN exists and OUT not => OUT (only if after OUT allowed after)
        - Else => None
        """
        now_t = datetime.now().time()
        has_in, has_out = self.get_person_state_today(pid)

        if not has_in and not has_out:
            if now_t <= self.in_allowed_until:
                return "IN", "First time today => IN"
            return None, "IN time window closed"

        if has_in and not has_out:
            if now_t >= self.out_allowed_after:
                return "OUT", "Already IN, allowed OUT now"
            return None, "OUT allowed only after configured time"

        return None, "Already completed IN & OUT today"

    def append_mark(self, pid: int, name: str, conf: float, event: str, operator: str, snap: str):
        df = load_day_df(today_str())
        if event == "IN":
            status = "Late" if datetime.now().time() > self.late_after else "OnTime"
        else:
            status = "-"
        row = {
            "record_id": str(uuid4()),
            "date": today_str(),
            "time": now_time_str(),
            "person_id": int(pid),
            "name": name,
            "event": event,
            "status": status,
            "confidence": round(float(conf), 2),
            "operator": operator,
            "snapshot_path": snap
        }
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        save_day_df(today_str(), df)

    # ---------- Records
    def get_selected_row(self):
        table = self.table_today if self.table_today.currentRow() >= 0 else self.table_all
        row = table.currentRow()
        if row < 0:
            return None
        return {ATT_COLS[i]: (table.item(row, i).text() if table.item(row, i) else "") for i in range(len(ATT_COLS))}

    def edit_selected(self):
        row = self.get_selected_row()
        if not row:
            show_popup(self, "Edit", "Select a record first.", QMessageBox.Warning)
            return
        dlg = EditDialog(self, row)
        if dlg.exec() != QDialog.Accepted:
            return
        try:
            fields = dlg.fields()
        except Exception as e:
            show_popup(self, "Edit Failed", str(e), QMessageBox.Warning)
            return
        date_ = row["date"]
        df = load_day_df(date_)
        mask = df["record_id"].astype(str) == str(row["record_id"])
        if not mask.any():
            show_popup(self, "Edit", "Record not found.", QMessageBox.Warning)
            return
        for k, v in fields.items():
            df.loc[mask, k] = v
        save_day_df(date_, df)
        self.refresh_all()

    def delete_selected(self):
        row = self.get_selected_row()
        if not row:
            show_popup(self, "Delete", "Select a record first.", QMessageBox.Warning)
            return
        ans = QMessageBox.question(self, "Confirm Delete", "Delete selected record?", QMessageBox.Yes | QMessageBox.No)
        if ans != QMessageBox.Yes:
            return
        date_ = row["date"]
        df = load_day_df(date_)
        df = df[df["record_id"].astype(str) != str(row["record_id"])].reset_index(drop=True)
        save_day_df(date_, df)
        self.refresh_all()

    # ---------- Exports
    def export_today(self):
        df = load_day_df(today_str())
        if df.empty:
            show_popup(self, "Export", "No records today.", QMessageBox.Information)
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", f"attendance_{today_str()}.csv", "CSV Files (*.csv)")
        if not path:
            return
        df.to_csv(path, index=False)
        show_popup(self, "Export", f"Saved:\n{path}", QMessageBox.Information)

    def export_all(self):
        df = export_all_combined_df()
        if df.empty:
            show_popup(self, "Export", "No logs found.", QMessageBox.Information)
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "attendance_all.csv", "CSV Files (*.csv)")
        if not path:
            return
        df.to_csv(path, index=False)
        show_popup(self, "Export", f"Saved:\n{path}", QMessageBox.Information)

    def export_registered(self):
        df = registered_df()
        if df.empty:
            show_popup(self, "Export", "No registered candidates.", QMessageBox.Information)
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "registered_candidates.csv", "CSV Files (*.csv)")
        if not path:
            return
        df.to_csv(path, index=False)
        show_popup(self, "Export", f"Saved:\n{path}", QMessageBox.Information)

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

    def fill_table(self, table, df):
        table.setRowCount(0)
        if df is None or df.empty:
            return
        df = ensure_att_cols(df).sort_values(by=["date", "time"], ascending=False)
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
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(90, 90))
        now_ts = time.time()

        if self.mode == "register":
            # capture samples
            if self.reg_folder is not None and self.reg_taken < self.reg_needed and len(faces) > 0:
                x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                face_img = cv2.resize(gray[y:y+h, x:x+w], FACE_SIZE)
                if blur_score(face_img) >= REG_BLUR_MIN:
                    self.reg_taken += 1
                    cv2.imwrite(str(self.reg_folder / f"{self.reg_taken}.jpg"), face_img)
                    self.reg_status.setText(f"Status: Captured {self.reg_taken}/{self.reg_needed}")
                else:
                    self.reg_status.setText("Status: Too blurry. Hold still...")
                if self.reg_taken >= self.reg_needed:
                    self.reg_status.setText("Status: Capture complete. Click Train Model.")
                    self.refresh_all()

            for (x, y, w, h) in faces:
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)

            self.reg_preview.setPixmap(frame_to_pixmap(display, PREVIEW_W, PREVIEW_H))
            return

        if self.mode == "attendance":
            ok_model, msg = self.validate_model_ready()
            if not ok_model:
                self.banner.setText("Model not trained")
                self.att_info.setText(msg)
                self.att_preview.setPixmap(frame_to_pixmap(display, PREVIEW_W, PREVIEW_H))
                return

            threshold = int(self.threshold.value())
            operator = self.operator.text().strip() or "Operator"

            live_names = []
            marked_any = False

            for (x, y, w, h) in faces:
                face_img = cv2.resize(gray[y:y+h, x:x+w], FACE_SIZE)
                label, dist = self.recognizer.predict(face_img)
                conf = max(0.0, min(100.0, 100.0 - float(dist)))

                if dist < threshold:
                    pid = int(label)
                    name = self.registered_map.get(pid, f"ID {pid}")
                    live_names.append(f"{name}({pid})")

                    # AUTO decide IN/OUT for this person
                    event, reason = self.decide_event_auto(pid)
                    if event is None:
                        cv2.rectangle(display, (x, y), (x+w, y+h), (0, 165, 255), 2)
                        cv2.putText(display, reason, (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
                        continue

                    # liveness
                    if self.chk_liveness.isChecked():
                        blink_ts = self.blink_gate.update(face_img, now_ts)
                        if now_ts - blink_ts > 3.0:
                            self.banner.setText("Blink required (anti-proxy)")
                            cv2.putText(display, "Blink required", (x, y-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
                            continue

                    # cooldown per pid+event
                    key = (pid, event)
                    last = self.cooldown.get(key, 0.0)
                    if now_ts - last < COOLDOWN_SECONDS:
                        continue

                    # mark
                    snap_path = str(SNAP_MARKED / f"{today_str()}_{now_time_str().replace(':','')}_ID{pid}_{event}.jpg")
                    cv2.imwrite(snap_path, frame)
                    self.append_mark(pid, name, conf, event, operator, snap_path)
                    self.cooldown[key] = now_ts
                    marked_any = True

                    cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(display, f"{name} {event} conf:{conf:.1f}", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

                    self.banner.setText("Attendance Taken Successfully")
                    self.att_info.setText(f"{event} marked: {name} (ID {pid})")

                    show_popup(
                        self,
                        "Attendance Taken Successfully",
                        f"{event} marked for:\n\nName: {name}\nID: {pid}\nTime: {now_time_str()}\nDate: {today_str()}",
                        QMessageBox.Information
                    )
                    QApplication.beep()
                    self.refresh_all()

                    if not self.chk_multi.isChecked():
                        break
                else:
                    cv2.rectangle(display, (x, y), (x+w, y+h), (255, 80, 80), 2)
                    cv2.putText(display, "Unknown", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 80, 80), 2)

            if len(faces) == 0:
                self.banner.setText("No face detected")
                self.att_info.setText("Live: -")
            else:
                if not marked_any:
                    self.banner.setText("Scanning...")
                if live_names:
                    self.att_info.setText("Live: " + ", ".join(live_names[:6]))

                if self.chk_unknown.isChecked() and not marked_any and (now_ts - self.last_unknown_ts) > UNKNOWN_SAVE_COOLDOWN:
                    unknown_path = SNAP_UNKNOWN / f"{today_str()}_{now_time_str().replace(':','')}_unknown.jpg"
                    cv2.imwrite(str(unknown_path), frame)
                    self.last_unknown_ts = now_ts
                    self.refresh_all()

            self.att_preview.setPixmap(frame_to_pixmap(display, PREVIEW_W, PREVIEW_H))

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()