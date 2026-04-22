import cv2
import os
from pathlib import Path

DATASET_DIR = Path("dataset")
TRAINER_DIR = Path("trainer")
DATASET_DIR.mkdir(exist_ok=True)
TRAINER_DIR.mkdir(exist_ok=True)

# Haar cascade (comes with opencv)
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces = []
    labels = []

    for person_folder in DATASET_DIR.iterdir():
        if not person_folder.is_dir():
            continue
        label = int(person_folder.name.split("_")[0])  # folder like "1_John"
        for img_path in person_folder.glob("*.jpg"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(img)
            labels.append(label)

    if len(faces) == 0:
        print("No faces found to train.")
        return

    recognizer.train(faces, labels)
    recognizer.save(str(TRAINER_DIR / "trainer.yml"))
    print("Training complete. Saved: trainer/trainer.yml")

def main():
    print("=== Face Registration (LBPH) ===")
    person_id = input("Enter numeric ID (e.g., 1,2,3): ").strip()
    name = input("Enter name (e.g., John): ").strip()

    if not person_id.isdigit():
        print("ID must be numeric for LBPH.")
        return

    label = int(person_id)
    folder = DATASET_DIR / f"{label}_{name}"
    folder.mkdir(exist_ok=True)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Camera not found.")
        return

    count = 0
    max_samples = 40

    print("Look at camera. Press 'q' to quit early.")
    while True:
        ret, frame = cam.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200, 200))

            count += 1
            cv2.imwrite(str(folder / f"{count}.jpg"), face_img)

            cv2.putText(frame, f"Captured: {count}/{max_samples}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Register - Face Capture", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if count >= max_samples:
            break

    cam.release()
    cv2.destroyAllWindows()

    print(f"Saved {count} images in {folder}")
    train_model()

if __name__ == "__main__":
    main()