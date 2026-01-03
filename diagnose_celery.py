#!/usr/bin/env python3
"""Diagnóstico completo del estado de Celery y Redis"""

import redis
import json
from app.core.celery_app import celery_app
from app.core.config import settings
import time

print("=" * 60)
print("DIAGNÓSTICO CELERY + REDIS")
print("=" * 60)

try:
    # 1. Revisar conexión Redis
    print("\n[1] REDIS CONNECTION")
    r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    r.ping()
    print("✓ Redis DB 0 (Broker) conectado")
    
    r_results = redis.Redis(host='localhost', port=6379, db=1, decode_responses=True)
    r_results.ping()
    print("✓ Redis DB 1 (Results) conectado")
    
    # 2. Revisar tareas pendientes
    print("\n[2] TAREAS PENDIENTES EN COLA")
    # Obtener todas las claves en el broker
    keys = r.keys('*')
    print(f"Total de claves en Redis DB 0: {len(keys)}")
    
    # Revisar la cola celery
    queue_len = r.llen('celery')
    print(f"Tareas en cola 'celery': {queue_len}")
    
    if queue_len > 0:
        print("Primeras tareas en cola:")
        for i in range(min(3, queue_len)):
            task_data = r.lindex('celery', i)
            if task_data:
                try:
                    task_json = json.loads(task_data)
                    print(f"  [{i}] ID: {task_json.get('headers', {}).get('id', 'N/A')}")
                    print(f"       Task: {task_json.get('headers', {}).get('task', 'N/A')}")
                except:
                    print(f"  [{i}] {task_data[:80]}")
    
    # 3. Revisar resultado de tareas
    print("\n[3] RESULTADOS DE TAREAS")
    result_keys = r_results.keys('*')
    print(f"Resultados guardados: {len(result_keys)}")
    if result_keys:
        print(f"Claves de resultados: {result_keys[:5]}")
    
    # 4. Revisar estado de Celery
    print("\n[4] ESTADO DE CELERY")
    celery_status = celery_app.control.inspect()
    
    # Revisar workers activos
    active_workers = celery_status.active()
    if active_workers:
        print(f"✓ Workers activos: {list(active_workers.keys())}")
        for worker, tasks in active_workers.items():
            print(f"  {worker}: {len(tasks)} tareas activas")
    else:
        print("✗ NO HAY WORKERS ACTIVOS")
    
    # Revisar tareas registradas
    registered_tasks = celery_status.registered()
    if registered_tasks:
        print(f"\n✓ Tareas registradas:")
        for worker, tasks in registered_tasks.items():
            if 'process_pdf_task' in tasks:
                print(f"  {worker}: ✓ process_pdf_task disponible")
            else:
                print(f"  {worker}: ✗ process_pdf_task NO DISPONIBLE")
    else:
        print("✗ NO HAY TAREAS REGISTRADAS")
    
    # Revisar tareas pendientes
    reserved_tasks = celery_status.reserved()
    if reserved_tasks:
        print(f"\n✓ Tareas reservadas (siendo procesadas):")
        for worker, tasks in reserved_tasks.items():
            print(f"  {worker}: {len(tasks)} tareas")
    else:
        print("\n  No hay tareas siendo procesadas actualmente")
    
    # 5. Revisar stats
    print("\n[5] ESTADÍSTICAS")
    stats = celery_status.stats()
    if stats:
        for worker, stat in stats.items():
            print(f"  {worker}:")
            print(f"    Pool: {stat.get('pool', {})}")
    else:
        print("✗ No se puede obtener estadísticas")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("FIN DEL DIAGNÓSTICO")
print("=" * 60)
