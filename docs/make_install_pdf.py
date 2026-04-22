from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib import colors

OUT_DIR = Path(__file__).resolve().parent
PDF_PATH = OUT_DIR / "Installation_Guide.pdf"

TITLE = "Smart Face Attendance System (IN/OUT)\nInstallation & Configuration Guide"
SUBTITLE = "CSV + LBPH + PySide6 (Windows)"

CONTENT = [
("1. Copy Project Folder",
"""Copy the project folder to the new system:
  face_attendance_simple/

Recommended to copy:
  • ui_app.py
  • dataset/ (optional but needed for existing registrations)
  • trainer/trainer.yml (optional but needed for recognition without retraining)
  • attendance_logs/ (optional - past logs)
"""),

("2. Install Python",
"""Install Python 3.10 or 3.11 (64-bit) from https://python.org
During installation, select:
  • Add Python to PATH

Verify:
  python --version
"""),

("3. Create & Activate Virtual Environment",
"""Open Command Prompt in the project folder:

  cd D:\\face_attendance_simple
  python -m venv .venv
  .venv\\Scripts\\activate

Upgrade pip:
  python -m pip install --upgrade pip
"""),

("4. Install Required Libraries",
"""Install dependencies:

  pip install PySide6 opencv-contrib-python numpy pandas

Verify OpenCV face module exists:
  python -c "import cv2; print(hasattr(cv2,'face'))"

If False, fix:
  pip uninstall opencv-python -y
  pip install opencv-contrib-python
"""),

("5. Run the Application",
"""Run:
  python ui_app.py

If trainer/trainer.yml is missing:
  • Go to Register → capture samples → Train Model → then Attendance.
"""),

("6. First-Time Setup Checklist",
"""• Ensure good lighting and face centered during registration.
• Register 50–80 samples per person.
• Train model after adding new persons.
• Close other apps using the camera (Zoom/Teams/Browser) if camera fails.
"""),

("7. Troubleshooting",
"""A) ModuleNotFoundError (cv2/PySide6/pandas):
   Activate venv and install missing package with pip.

B) Wrong python/venv:
   Check:
     where python
     python -m pip -V
   Ensure paths point inside:
     ...\\face_attendance_simple\\.venv\\

C) Recognition inaccurate:
   • Increase samples, improve light, retrain model
   • Adjust threshold (55–65)
"""),
]

def header(c, w, h):
    c.setFillColor(colors.HexColor("#0B3B74"))
    c.rect(0, h-2.2*cm, w, 2.2*cm, stroke=0, fill=1)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(1.5*cm, h-1.45*cm, "Installation Guide")

def footer(c, w, h, page_no):
    c.setFillColor(colors.HexColor("#111827"))
    c.rect(0, 0, w, 1.0*cm, stroke=0, fill=1)
    c.setFillColor(colors.white)
    c.setFont("Helvetica", 9)
    c.drawString(1.5*cm, 0.35*cm, "Smart Face Attendance System (IN/OUT)")
    c.drawRightString(w-1.5*cm, 0.35*cm, f"Page {page_no}")

def wrap_text(c, text, font="Helvetica", size=10.5, max_width=17.5*cm):
    c.setFont(font, size)
    lines = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        line = ""
        for w in words:
            test = (line + " " + w).strip()
            if c.stringWidth(test, font, size) <= max_width:
                line = test
            else:
                lines.append(line)
                line = w
        if line:
            lines.append(line)
    return lines

def main():
    w, h = A4
    c = canvas.Canvas(str(PDF_PATH), pagesize=A4)

    # Cover
    c.setFillColor(colors.HexColor("#0B1220"))
    c.rect(0, 0, w, h, stroke=0, fill=1)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 20)
    y = h - 4*cm
    for line in TITLE.split("\n"):
        c.drawString(2*cm, y, line)
        y -= 0.9*cm
    c.setFillColor(colors.HexColor("#CFE0FF"))
    c.setFont("Helvetica", 12)
    c.drawString(2*cm, y-0.5*cm, SUBTITLE)
    c.showPage()

    page_no = 1
    for sec_title, sec_body in CONTENT:
        header(c, w, h)
        footer(c, w, h, page_no)
        page_no += 1

        y = h - 3.2*cm
        c.setFillColor(colors.HexColor("#0F172A"))
        c.setFont("Helvetica-Bold", 13.5)
        c.drawString(1.5*cm, y, sec_title)
        y -= 0.8*cm

        c.setFillColor(colors.HexColor("#111827"))
        lines = wrap_text(c, sec_body, font="Helvetica", size=10.5, max_width=18.0*cm)

        for ln in lines:
            if y < 2.0*cm:
                c.showPage()
                header(c, w, h)
                footer(c, w, h, page_no)
                page_no += 1
                y = h - 3.2*cm
            c.drawString(1.5*cm, y, ln)
            y -= 0.55*cm

        c.showPage()

    c.save()
    print("Saved:", PDF_PATH)

if __name__ == "__main__":
    main()