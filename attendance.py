import cv2
import os
import pandas as pd
from datetime import datetime
from pathlib import Path

DATASET_DIR = Path("dataset")
TRAINER_PATH = Path("trainer") / "trainer.yml"
LOG_DIR = Path("attendance_logs")
LOG_DIR.mkdir(exist_ok=True)

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

def load_label_map():
    """
    Reads dataset folders like:
    dataset/1_John
    dataset/2_Ali
    returns {1: "John", 2: "Ali"}
    """
    label_map = {}
    if not DATASET_DIR.exists():
        return label_map

    for person_folder in DATASET_DIR.iterdir():
        if person_folder.is_dir():
            parts = person_folder.name.split("_", 1)
            if len(parts) == 2 and parts[0].isdigit():
                label_map[int(parts[0])] = parts[1]
    return label_map

def mark_attendance(person_id, name, confidence):
    today = datetime.now().strftime("%Y-%m-%d")
    filename = LOG_DIR / f"attendance_{today}.csv"

    row = {
        "date": today,
        "time": datetime.now().strftime("%H:%M:%S"),
        "person_id": person_id,
        "name": name,
        "confidence": round(confidence, 2)
    }

    if filename.exists():
        df = pd.read_csv(filename)
        # prevent duplicate for same person in same day
        if ((df["person_id"] == person_id)).any():
            return False
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(filename, index=False)
    return True

def main():
    if not TRAINER_PATH.exists():
        print("trainer/trainer.yml not found. Run register.py first.")
        return

    label_map = load_label_map()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(TRAINER_PATH))

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Camera not found.")
        return

    print("=== Attendance Mode ===")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200, 200))

            label, dist = recognizer.predict(face_img)
            # For LBPH: smaller dist = better. We'll map it to a simple confidence.
            confidence = max(0.0, min(100.0, 100.0 - dist))

            name = label_map.get(label, "Unknown")

            # Threshold: tune this (lower dist means more accurate)
            if dist < 60:
                marked = mark_attendance(label, name, confidence)
                status = "MARKED" if marked else "ALREADY"
                color = (0, 255, 0)
                text = f"{name} (ID {label}) {status} conf:{confidence:.1f}"
            else:
                color = (0, 0, 255)
                text = "Unknown"

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Attendance - LBPH", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()