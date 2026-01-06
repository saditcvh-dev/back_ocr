"""
Proceso profesional de PDF:
1. PDF escaneado → OCR (texto invisible)
2. PDF OCR → añadir texto encima (overlay)
No se convierte a Word.
No se altera el texto OCR.
"""

import subprocess
from pathlib import Path
import fitz  # PyMuPDF


# -----------------------------
# CONFIGURACIÓN
# -----------------------------

INPUT_PDF = Path("uploads/FOV_Inscripcion_e7355f8a36d7ef_e7355f8a36d7efb1.pdf")
OCR_PDF = Path("outputs/FOV_Inscripcion_ocr.pdf")
FINAL_PDF = Path("outputs/FOV_Inscripcion_final.pdf")

OCR_LANGUAGE = "spa"


# -----------------------------
# PASO 1: OCR CON OCRmyPDF
# -----------------------------

def apply_ocr(input_pdf: Path, output_pdf: Path):
    """
    Aplica OCR y mantiene el PDF visual intacto.
    El texto queda invisible pero seleccionable.
    """
    cmd = [
        "ocrmypdf",
        "-l", OCR_LANGUAGE,
        "--force-ocr",
        "--deskew",
        str(input_pdf),
        str(output_pdf)
    ]
    subprocess.run(cmd, check=True)


# -----------------------------
# PASO 2: AÑADIR TEXTO ENCIMA
# -----------------------------

def add_overlay_text(input_pdf: Path, output_pdf: Path):
    """
    Añade texto encima del PDF como si fuera escritura digital.
    No toca el OCR.
    """
    doc = fitz.open(input_pdf)

    page = doc[0]  # primera página

    page.insert_text(
        (72, 150),  # posición (x, y)
        "Texto añadido encima del PDF",
        fontsize=12,
        fontname="helv",
        color=(0, 0, 0)  # negro
    )

    page.insert_text(
        (72, 180),
        "Firma: _______________________",
        fontsize=12
    )

    doc.save(output_pdf)
    doc.close()


# -----------------------------
# MAIN
# -----------------------------

if __name__ == "__main__":
    OCR_PDF.parent.mkdir(exist_ok=True)

    print("▶ Aplicando OCR...")
    apply_ocr(INPUT_PDF, OCR_PDF)

    print("▶ Añadiendo texto encima...")
    add_overlay_text(OCR_PDF, FINAL_PDF)

    print(" PDF final listo:", FINAL_PDF)
