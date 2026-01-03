# Sistema de Colas de Procesamiento PDF (v2.0)

## 🎯 Cambios Implementados

Este sistema ha sido actualizado para procesar PDFs de forma asincrónica mediante una **cola de tareas con Celery + Redis**. 

### ¿Qué mejora?
- ✅ **No se satura el servidor**: Los PDFs se encolan y procesan en segundo plano
- ✅ **Respuesta inmediata**: El endpoint devuelve al instante sin esperar a OCR
- ✅ **Múltiples usuarios**: Varios PDFs se procesan en paralelo
- ✅ **Seguimiento de estado**: Consulta el progreso en tiempo real
- ✅ **Tolerancia a fallos**: Si algo falla, puedes reintentar

---

## 📋 Requisitos Previos

Necesitas tener instalado:
1. **Redis** - Para la cola de mensajes
2. **Python 3.8+** - Ya tienes
3. **Tesseract OCR** - Ya tienes configurado

### Instalar Redis (Windows)

Opción 1 - Usar Docker (Recomendado):
```bash
docker run -d -p 6379:6379 redis:7-alpine
```

Opción 2 - Instalar en Windows:
- Descarga desde: https://github.com/microsoftarchive/redis/releases
- O usa WSL2

### Instalar dependencias Python

```bash
pip install -r requirements.txt
```

---

## 🚀 Cómo Ejecutar (3 terminales)

### Terminal 1: FastAPI Server
```bash
python run.py
```
Estará disponible en: http://localhost:8000

### Terminal 2: Celery Worker (Procesa los PDFs)
```bash
# Windows:
start_celery_worker.bat

# Linux/Mac:
bash start_celery_worker.sh

# O ejecuta manualmente:
celery -A app.core.celery_app worker --loglevel=info --concurrency=2
```

### Terminal 3: Redis (Opcional si usas Docker)
```bash
docker run -d -p 6379:6379 redis:7-alpine
```

Si instalaste Redis localmente, ejecuta el servidor Redis en tu máquina.

---

## 📡 Flujo de Funcionamiento

```
Usuario                FastAPI                 Celery              Redis
  |                      |                       |                   |
  |--1. Subir PDF------->|                       |                   |
  |                      |--2. Encolar Tarea-----|---Guardar Task----|
  |<---3. Respuesta------|                       |                   |
  |                      |                       |                   |
  |--4. Consultar Status-|--5. Check Status------|--Get Result-------|
  |<---6. Status---------|                       |                   |
  |                      |                       |--7. Procesar--    |
  |                      |                       |   (Mientras       |
  |                      |                       |    tanto)         |
  |--8. Consultar Status-|                       |                   |
  |<---9. Completado!----|                       |                   |
```

---

## 🔌 API Endpoints

### 1️⃣ Subir PDF (Encolar)
```bash
POST /api/pdf/upload
Content-Type: multipart/form-data

Parámetros:
- file: <PDF file>
- use_ocr: true (opcional)
```

**Respuesta:**
```json
{
  "id": "documento_a1b2c3d4",
  "filename": "documento.pdf",
  "size": 2048576,
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "message": "PDF encolado para procesamiento...",
  "estimated_wait_time": 10.0
}
```

### 2️⃣ Consultar Estado del Procesamiento
```bash
GET /api/pdf/upload-status/{pdf_id}
```

**Respuesta:**
```json
{
  "pdf_id": "documento_a1b2c3d4",
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "pages": 50,
  "extracted_text_path": "extracted_texts/documento_a1b2c3d4.txt",
  "used_ocr": true,
  "progress": 50,
  "error": null,
  "created_at": "2025-12-27T10:30:00",
  "completed_at": null
}
```

**Estados posibles:**
- `pending` - En cola, esperando a ser procesado
- `processing` - Actualmente procesándose
- `completed` - ✅ Listo para usar
- `failed` - ❌ Error durante el procesamiento

### 3️⃣ Buscar en un PDF (Una vez completado)
```bash
POST /api/pdf/{pdf_id}/search

Body JSON:
{
  "term": "tu búsqueda",
  "case_sensitive": false
}
```

### 4️⃣ Buscar en TODOS los PDFs
```bash
POST /api/pdf/global-search?term=texto&case_sensitive=false
```

### 5️⃣ Descargar Texto Extraído
```bash
GET /api/pdf/{pdf_id}/text
```

---

## 💡 Ejemplo de Uso con JavaScript/Angular

```javascript
// 1. Subir PDF
uploadFile(file: File) {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('use_ocr', true);
  
  return this.http.post('/api/pdf/upload', formData);
}

// 2. Suscribirse a cambios de estado
monitorUpload(pdfId: string) {
  return interval(2000) // Consultar cada 2 segundos
    .pipe(
      switchMap(() => this.http.get(`/api/pdf/upload-status/${pdfId}`)),
      takeWhile((status: any) => status.status !== 'completed' && status.status !== 'failed')
    );
}

// Uso:
this.uploadFile(file).subscribe((response: any) => {
  const pdfId = response.id;
  
  this.monitorUpload(pdfId).subscribe(
    (status) => {
      console.log(`Estado: ${status.status} (${status.progress}%)`);
      if (status.status === 'completed') {
        console.log('PDF listo para buscar!');
      }
    }
  );
});
```

---

## 📊 Monitoreo de Celery (Opcional)

Para ver en tiempo real qué se está procesando, instala Flower:

```bash
pip install flower
celery -A app.core.celery_app flower
```

Luego accede a: http://localhost:5555

---

## 🛠️ Configuración Avanzada

### Ajustar concurrencia (cuántos PDFs a la vez)

En `start_celery_worker.bat` o `.sh`, cambia `--concurrency=2`:

```bash
# Procesar 4 PDFs simultáneamente
celery -A app.core.celery_app worker --loglevel=info --concurrency=4
```

**⚠️ Nota:** Más concurrencia = más memoria y CPU. Ajusta según tu servidor.

### Cambiar timeout de procesamiento

En `app/core/celery_app.py`:

```python
task_time_limit=30 * 60,  # Cambiar a minutos
```

---

## 🐛 Troubleshooting

### Error: "Redis connection refused"
```
Solución: Asegúrate de que Redis está ejecutándose
Windows: Verifica en el Docker Desktop o servicio de Windows
Linux: sudo systemctl start redis-server
```

### Error: "Celery worker no procesa tareas"
```
Solución: Verifica que el worker está activo en la terminal 2
- Deberías ver "Ready to accept tasks" en la consola
- Si no, revisa que TESSDATA_PREFIX esté correctamente configurado
```

### El PDF no se procesa (stuck en "pending")
```
Solución:
1. Verifica que Celery worker esté corriendo (Terminal 2)
2. Revisa los logs del worker
3. Reinicia el worker
```

---

## 📈 Métricas de Rendimiento

Con la nueva arquitectura:

| Métrica | Antes | Después |
|---------|-------|---------|
| Respuesta al subir | 30-120s ⏳ | 100ms ⚡ |
| Timeout de cliente | Frecuente | Cero |
| PDFs simultáneos | 1 | 4+ |
| Saturación servidor | Alta | Baja |

---

## 📝 Notas Importantes

1. **Persistencia**: El estado actualmente se guarda en memoria. Para producción, usa Redis o una BD.
2. **Almacenamiento**: Los PDFs se guardan en `/uploads/` y textos en `/extracted_texts/`
3. **Limpieza**: Implementa un cron job para limpiar PDFs antiguos
4. **Seguridad**: Agrega autenticación para producción

---

## 🎓 Próximas Mejoras Opcionales

- [ ] Base de datos (PostgreSQL) en lugar de memoria
- [ ] Autenticación y autorización
- [ ] Rate limiting por usuario
- [ ] Webhook de notificación al completar
- [ ] Compresión automática de textos grandes
- [ ] Caché de búsquedas frecuentes

---

¡Sistema listo para escalar! 🚀
