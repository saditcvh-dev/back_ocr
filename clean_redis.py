#!/usr/bin/env python
"""
Limpia la cola de tareas de Celery/Redis
"""
import redis

# Conectar a Redis
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Limpiar todas las claves de Celery
print("Limpiando queues de Celery...")
redis_client.flushdb()
print("✅ Cola de Celery limpiada")

# Conectar a la DB de resultados
results_redis = redis.Redis(host='localhost', port=6379, db=1)
print("Limpiando resultados...")
results_redis.flushdb()
print("✅ Resultados limpiados")

print("\n🎉 Redis completamente limpio - Listo para nuevas tareas")
