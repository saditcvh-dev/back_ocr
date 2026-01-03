import requests
import json
import time
from pathlib import Path

BASE_URL = "http://localhost:8000/api/pdf"

print("=" * 60)
print("PRUEBA DEL SISTEMA DE COLAS DE PDFs")
print("=" * 60)

# Buscar un PDF de prueba en la carpeta uploads
pdf_files = list(Path("uploads").glob("*.pdf"))
print(f"\n📁 PDFs disponibles en uploads/: {len(pdf_files)}")
for pdf in pdf_files[:3]:
    print(f"   - {pdf.name}")

if not pdf_files:
    print("\n⚠️  No hay PDFs en la carpeta 'uploads/'")
    print("   Descarga un PDF de prueba y colócalo en uploads/")
    exit(1)

# Usar el primer PDF
test_pdf = pdf_files[0]
print(f"\n✅ Usando PDF de prueba: {test_pdf.name}")

# PASO 1: Subir el PDF
print("\n" + "=" * 60)
print("PASO 1: Subiendo PDF a la COLA")
print("=" * 60)

with open(test_pdf, 'rb') as f:
    files = {'file': (test_pdf.name, f, 'application/pdf')}
    params = {'use_ocr': True}
    
    response = requests.post(f"{BASE_URL}/upload", files=files, params=params)
    
    if response.status_code == 200:
        data = response.json()
        pdf_id = data['id']
        task_id = data['task_id']
        
        print(f"✅ PDF encolado exitosamente!")
        print(f"   PDF ID: {pdf_id}")
        print(f"   Task ID: {task_id}")
        print(f"   Status: {data['status']}")
        print(f"   Mensaje: {data['message']}")
    else:
        print(f"❌ Error al subir PDF: {response.status_code}")
        print(response.text)
        exit(1)

# PASO 2: Consultar estado varias veces
print("\n" + "=" * 60)
print("PASO 2: Consultando estado del procesamiento")
print("=" * 60)

for i in range(10):
    time.sleep(2)  # Esperar 2 segundos entre consultas
    
    response = requests.get(f"{BASE_URL}/upload-status/{pdf_id}")
    
    if response.status_code == 200:
        data = response.json()
        status = data['status']
        progress = data.get('progress', 0)
        
        print(f"\n[{i+1}/10] Status: {status.upper()}")
        print(f"   Progress: {progress}%")
        
        if status == 'completed':
            print(f"   ✅ ¡COMPLETADO!")
            print(f"   Páginas: {data.get('pages')}")
            print(f"   Texto extraído: {data.get('extracted_text_path')}")
            print(f"   Se usó OCR: {data.get('used_ocr')}")
            break
        elif status == 'failed':
            print(f"   ❌ Error: {data.get('error')}")
            break
    else:
        print(f"❌ Error consultando estado: {response.status_code}")
        print(response.text)

print("\n" + "=" * 60)
print("PRUEBA COMPLETADA")
print("=" * 60)
