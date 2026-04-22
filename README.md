```markdown
# 🎯 Face Attendance System (SQLite + OpenCV)

A simple face-based attendance system built using **Python, OpenCV, and SQLite**.  
This project detects faces via webcam, marks attendance in a database, and can send email notifications.

---

## 📁 Project Structure

```

face_attendance_sqlite/
│  app.py
│  requirements.txt
│
├─ core/
│   ├─ db.py
│   ├─ vision.py
│   ├─ mailer.py
│   └─ reports.py
│
└─ data/
├─ snapshots/
│   ├─ marked/
│   └─ unknown/
└─ app.db

````

---

## 🚀 Features

- 🎥 Real-time face detection using OpenCV
- 🗂️ SQLite database for storing attendance
- 🧾 Attendance logs with timestamps
- 📧 Email notification after marking attendance
- 📊 Basic attendance report generation
- 📁 Snapshot storage for marked/unknown faces (extendable)

---

## 🛠️ Tech Stack

- Python 3.x
- OpenCV
- SQLite3
- SMTP (for email alerts)

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository
```bash
git clone <your-repo-url>
cd face_attendance_sqlite
````

---

### 2️⃣ Create Virtual Environment

```cmd
py -m venv .venv
.venv\Scripts\activate
```

---

### 3️⃣ Install Dependencies

```cmd
py -m pip install -r requirements.txt
```

---

### 4️⃣ Run the Application

```cmd
python app.py
```

---

## 🗄️ Database Details

The SQLite database (`app.db`) is auto-created inside the `data/` folder.

### Tables:

* **users**

  * id (Primary Key)
  * name

* **attendance**

  * id (Primary Key)
  * user_id (Foreign Key)
  * timestamp
  * status

---

## 📧 Email Configuration

Update credentials in:

```
core/mailer.py
```

```python
sender = "your_email@gmail.com"
password = "your_app_password"
```

> ⚠️ Use **App Password** (not your real Gmail password)

---

## 📊 Sample Output

```
Starting camera...
Face detected!
Attendance marked!
Attendance Report:
('John Doe', '2026-04-22T10:15:30', 'Present')
```

---

## ⚠️ Limitations (Current Version)

* Uses basic Haarcascade (not highly accurate)
* No real face recognition (only detection)
* Hardcoded user_id
* No duplicate attendance prevention

---

## 🔥 Future Enhancements

* ✅ Face Recognition using `face_recognition` / Deep Learning
* ✅ User Registration Module
* ✅ GUI using PyQt
* ✅ Export reports to Excel
* ✅ REST API (Spring Boot / FastAPI)
* ✅ Cloud deployment

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork and improve the project.

---

## 📜 License

This project is open-source and free to use.

---

## 👨‍💻 Author

**Prasanta Kumar Sethi**
Software Engineer | Java Full Stack Developer

```

---

If you want, I can also:
✔ :contentReference[oaicite:0]{index=0}  
✔ :contentReference[oaicite:1]{index=1}  
✔ :contentReference[oaicite:2]{index=2}  

Just tell me 👍
```


Here’s a polished, **GitHub-ready `README.md`** tailored specifically for your project:

---

# 🎯 Face Attendance System (SQLite + OpenCV)

A lightweight and practical **Face Attendance System** built using **Python, OpenCV, and SQLite**.
It detects faces via webcam, logs attendance into a database, and supports email notifications and reporting.

---

## 📁 Project Structure

```
face_attendance_sqlite/
│  app.py
│  requirements.txt
│
├─ core/
│   ├─ db.py
│   ├─ vision.py
│   ├─ mailer.py
│   └─ reports.py
│
└─ data/
   ├─ snapshots/
   │   ├─ marked/
   │   └─ unknown/
   └─ app.db
```

---

## 🚀 Features

* 🎥 Real-time face detection using webcam
* 🗄️ SQLite database for attendance storage
* 🧾 Timestamp-based attendance logging
* 📧 Email notification after marking attendance
* 📊 Attendance report generation
* 📁 Structured data storage for future scalability

---

## 🛠️ Tech Stack

* **Python 3.x**
* **OpenCV**
* **SQLite3**
* **SMTP (Email Integration)**

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the Repository

```bash
git clone <your-repo-url>
cd face_attendance_sqlite
```

---

### 2️⃣ Create Virtual Environment

```cmd
py -m venv .venv
.venv\Scripts\activate
```

---

### 3️⃣ Install Dependencies

```cmd
py -m pip install -r requirements.txt
```

---

### 4️⃣ Run the Application

```cmd
python app.py
```

---

## 🗄️ Database Schema

The database (`app.db`) is automatically created inside the `data/` folder.

### Tables:

#### `users`

| Column | Type         |
| ------ | ------------ |
| id     | INTEGER (PK) |
| name   | TEXT         |

#### `attendance`

| Column    | Type         |
| --------- | ------------ |
| id        | INTEGER (PK) |
| user_id   | INTEGER (FK) |
| timestamp | TEXT         |
| status    | TEXT         |

---

## 📧 Email Configuration

Update your email credentials in:

```
core/mailer.py
```

```python
sender = "your_email@gmail.com"
password = "your_app_password"
```

⚠️ Use an **App Password**, not your actual Gmail password.

---

## 📊 Example Output

```
Starting camera...
Face detected!
Attendance marked!
Attendance Report:
('John Doe', '2026-04-22T10:15:30', 'Present')
```

---

## ⚠️ Current Limitations

* Uses Haarcascade (basic detection, not recognition)
* No real face identity matching yet
* Hardcoded user handling
* No duplicate attendance prevention

---

## 🔥 Future Enhancements

* ✅ Face Recognition (Deep Learning / embeddings)
* ✅ User Registration System
* ✅ PyQt GUI Dashboard
* ✅ Excel/CSV Report Export
* ✅ REST API integration
* ✅ Cloud deployment support

---

## 🤝 Contributing

Feel free to fork, improve, and submit pull requests.

---

## 📜 License

This project is open-source and free to use.

---

## 👨‍💻 Author

**Prasanta Kumar Sethi**
Software Engineer | Java Full Stack Developer
* Add screenshots + demo GIF
* Convert it into a **full UI app (PyQt)**
* Add real face recognition (not just detection)

Just tell me 👍

