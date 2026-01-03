"""
Script para probar si Celery está procesando tareas
"""
import time
from app.core.celery_app import celery_app
from app.tasks.pdf_tasks import process_pdf_task

print("\n" + "="*60)
print("VERIFICANDO CELERY")
print("="*60)

# Inspeccionar worker disponibles
inspector = celery_app.control.inspect()

print("\n1️⃣  Verificando workers conectados...")
stats = inspector.stats()
if stats:
    print(f"✅ Workers encontrados: {len(stats)}")
    for worker_name in stats.keys():
        print(f"   - {worker_name}")
else:
    print("❌ NO HAY WORKERS CONECTADOS!")
    print("   Asegúrate de que el Celery Worker está corriendo")

print("\n2️⃣  Verificando tareas registradas...")
registered = inspector.registered()
if registered:
    for worker_name, tasks in registered.items():
        print(f"   Worker: {worker_name}")
        for task in tasks:
            if 'process_pdf' in task:
                print(f"   ✅ {task}")
else:
    print("❌ NO HAY TAREAS REGISTRADAS")

print("\n3️⃣  Probando envío de tarea de prueba...")
print("   (Esta tarea debería aparecer en el worker)")

# Enviar tarea de prueba simple
test_task = celery_app.send_task('process_pdf_task', args=['test_id', 'test_path.pdf'], kwargs={'use_ocr': False})
print(f"   Task ID: {test_task.id}")
print(f"   State: {test_task.state}")

print("\n" + "="*60)
print("GUÍA DE SOLUCIÓN:")
print("="*60)
print("Si los workers no aparecen:")
print("1. Asegúrate de que Redis está corriendo: docker start redis-pdf-api")
print("2. Inicia el Celery Worker: celery -A app.core.celery_app worker --loglevel=debug")
print("3. El worker debe mostrar '[tasks] process_pdf_task' al iniciar")
print("\n" + "="*60 + "\n")
