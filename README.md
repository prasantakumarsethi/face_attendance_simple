Here’s a **production-quality `README.md`** tailored exactly to your provided code (PySide6 + OpenCV + Pandas + CSV logging + GUI system):

---

```markdown
# 🎯 AI Smart Face Attendance System

A **production-ready desktop application** for face-based attendance using **Python, OpenCV, and PySide6 (Qt UI)**.  
This system provides **real-time face recognition, automated IN/OUT marking, dataset management, and analytics dashboard**.

---

## 🚀 Key Highlights

- 🎥 Real-time face detection & recognition (OpenCV LBPH)
- 🧠 Smart **AUTO IN / OUT logic**
- 👁️ Optional **Blink Detection (Liveness Check)**
- 🧾 CSV-based attendance logging (daily + combined)
- 📊 Interactive **Dashboard with analytics**
- 🖥️ Modern **PySide6 UI (dark theme)**
- 📸 Snapshot storage (recognized & unknown faces)
- 📤 Export reports (Today / All / Registered)
- ✏️ Edit & Delete attendance records
- 🧑‍💼 Multi-user dataset management

---

## 🛠️ Tech Stack

| Technology | Purpose |
|----------|--------|
| Python 3.x | Core programming |
| OpenCV | Face detection & recognition |
| PySide6 (Qt) | GUI application |
| NumPy | Image processing |
| Pandas | Data handling & CSV operations |

---

## 📁 Project Structure

```

project/
│
├── dataset/                 # Registered user face datasets
│   └── <id_name>/
│       └── images...
│
├── trainer/
│   └── trainer.yml         # Trained face model
│
├── attendance_logs/
│   ├── attendance_YYYY-MM-DD.csv
│   └── snapshots/
│       ├── marked/         # Recognized faces
│       └── unknown/        # Unknown detections
│
└── main.py                 # Your main application file

````

---

## ⚙️ Installation & Setup

### 1️⃣ Clone / Copy Project

```bash
git clone <your-repo-url>
cd project
````

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate:

* Windows:

```bash
venv\Scripts\activate
```

* Mac/Linux:

```bash
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install opencv-python opencv-contrib-python PySide6 numpy pandas
```

> ⚠️ `opencv-contrib-python` is REQUIRED for LBPH recognizer

---

### 4️⃣ Run Application

```bash
python ui_app.py
```

---

## 🎯 How It Works

### 🔹 1. Registration Phase

* Go to **Register Tab**
* Enter:

  * Person ID (numeric)
  * Name
  * Number of samples
* Start camera → Capture face samples
* Train model

📌 Data stored in:

```
dataset/<id_name>/
```

---

### 🔹 2. Training

* Click **Train Model**
* Generates:

```
trainer/trainer.yml
```

---

### 🔹 3. Attendance System

* Start camera in **Attendance Tab**
* System:

  * Detects face
  * Recognizes user
  * Automatically decides:

    * ✅ IN (first entry)
    * ✅ OUT (based on time gap rule)

---

## 🧠 Smart Features

### ✅ Auto IN/OUT Logic

| Condition         | Action               |
| ----------------- | -------------------- |
| No record today   | IN                   |
| IN exists, no OUT | OUT (after time gap) |
| IN & OUT done     | Ignore               |

---

### ⏱️ OUT Rule Configuration

You can define:

```
Allow OUT only after N hours from IN
```

---

### 👁️ Liveness Detection (Optional)

* Uses **eye detection (blink)**
* Prevents spoofing (photo attacks)

---

### 🔁 Cooldown System

* Prevents duplicate marking
* Configurable delay between entries

---

## 📊 Dashboard Features

* 👥 Total Registered Users
* 🟢 IN Today
* 🔴 OUT Today
* ❓ Unknown Faces
* 📋 Recent Activity Table

---

## 📁 Attendance Logs

Stored as:

```
attendance_logs/attendance_YYYY-MM-DD.csv
```

### Columns:

| Field         | Description            |
| ------------- | ---------------------- |
| record_id     | Unique UUID            |
| date          | Date                   |
| time          | Time                   |
| person_id     | User ID                |
| name          | User name              |
| event         | IN / OUT               |
| confidence    | Recognition confidence |
| operator      | System operator        |
| snapshot_path | Image path             |

---

## 📸 Snapshots

### Recognized:

```
attendance_logs/snapshots/marked/
```

### Unknown:

```
attendance_logs/snapshots/unknown/
```

---

## 📤 Export Options

* Export Today’s Records
* Export All Records
* Export Registered Users

---

## ✏️ Record Management

* Edit any field (ID, name, event, etc.)
* Delete records
* Real-time UI update

---

## ⚠️ Requirements & Notes

* Webcam required
* Good lighting improves accuracy
* Face should be clearly visible during registration
* Minimum blur threshold enforced

---

## 🔥 Future Enhancements

* 🔐 Face Recognition using Deep Learning (FaceNet / Dlib)
* 🌐 REST API integration (Spring Boot / FastAPI)
* 📱 Mobile App integration
* ☁️ Cloud storage (AWS / Firebase)
* 📊 Advanced analytics dashboard
* 🧾 Excel/PDF report export
* 👤 User authentication system

---

## 🐞 Troubleshooting

### Camera not opening?

* Check if camera is used by another app

### Model not working?

* Ensure:

  * Dataset exists
  * Model is trained

### Low accuracy?

* Improve:

  * Lighting
  * Face angle
  * Number of samples

---

## 👨‍💻 Author

**Prasanta Kumar Sethi**
Software Engineer | Java Full Stack Developer

---

## 📜 License

This project is open-source and free to use.

---

## ⭐ Support

If you like this project:

* ⭐ Star the repo
* 🍴 Fork it
* 🧠 Contribute improvements

---
