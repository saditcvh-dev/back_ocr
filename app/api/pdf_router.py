from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Body
from fastapi.responses import FileResponse, JSONResponse
import os
import time
from datetime import datetime
from typing import List, Dict, Optional
from app.services.pdf_service import PDFService
from app.models.schemas import (
    PDFUploadResponse, 
    PDFUploadStatus,
    SearchRequest, 
    SearchResponse,
    PDFInfo,
    PDFListItem,
    PDFListResponse
)
from app.core.config import settings
from app.tasks.pdf_tasks import process_pdf_task, generate_ocr_pdf_task, full_process_pdf_task
from app.core.celery_app import celery_app
import logging
import threading

# Logger local
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/pdf", tags=["pdf"])
pdf_service = PDFService()

# Almacenamiento de metadatos en memoria (en producción usar Redis/DB)
pdf_storage = {}
pdf_task_status = {}  # {pdf_id: {task_id, status, created_at, etc.}}

@router.post("/upload", response_model=PDFUploadResponse)
async def upload_pdf(
    file: UploadFile = File(...), 
    use_ocr: bool = Query(True),
    use_ocrmypdf: bool = Query(True, description="Usar OCRmyPDF para mejor calidad OCR")
):
    """Sube un PDF a la cola de procesamiento sin esperar a que se procese"""
    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Solo se permiten archivos PDF")
        
        file_bytes = await file.read()
        pdf_id = pdf_service.generate_pdf_id(file.filename, file_bytes)
        
        # 1. Guardar archivo PDF
        pdf_path = os.path.join(settings.UPLOAD_FOLDER, f"{pdf_id}.pdf")
        with open(pdf_path, "wb") as f:
            f.write(file_bytes)
        
        # 2. Encolar tarea de procesamiento en Celery con nuevo parámetro use_ocrmypdf
        task = process_pdf_task.delay(
            pdf_id=pdf_id,
            pdf_path=pdf_path,
            use_ocr=use_ocr,
            use_ocrmypdf=use_ocrmypdf  # NUEVO PARÁMETRO
        )
        
        # 3. Guardar metadatos en memoria
        now = datetime.now()
        pdf_storage[pdf_id] = {
            'filename': file.filename,
            'pdf_path': pdf_path,
            'size': len(file_bytes),
            'upload_time': time.time(),
            'task_id': task.id,
            'use_ocr': use_ocr,
            'use_ocrmypdf': use_ocrmypdf  # NUEVO: guardar si se usó OCRmyPDF
        }
        
        pdf_task_status[pdf_id] = {
            'task_id': task.id,
            'status': 'pending',
            'created_at': now,
            'completed_at': None,
            'pages': None,
            'extracted_text_path': None,
            'used_ocr': use_ocr,
            'used_ocrmypdf': use_ocrmypdf,  # NUEVO
            'error': None
        }
        
        # 4. Devolver respuesta inmediata
        # Si no hay workers conectados, procesar sincrónicamente (mantener comportamiento previo)
        try:
            inspector = celery_app.control.inspect()
            registered = inspector.registered() if inspector else None
        except Exception:
            registered = None

        if not registered:
            # No hay workers -> procesar en background local (no bloquear la petición)
            logger.warning(f"No Celery workers detectados — procesando en hilo local {pdf_id}")

            def _local_process(pid: str, ppath: str, puse_ocr: bool, puse_ocrmypdf: bool, t_id: str):
                try:
                    pdf_task_status[pid].update({'status': 'processing'})
                    text, pages, used_ocr = pdf_service.extract_text_from_pdf(
                        ppath, 
                        use_ocr=puse_ocr,
                        use_ocrmypdf=puse_ocrmypdf  # NUEVO: pasar parámetro
                    )
                    text_path = pdf_service.save_extracted_text(text, pid)

                    pdf_task_status[pid].update({
                        'status': 'completed',
                        'pages': pages,
                        'extracted_text_path': text_path,
                        'used_ocr': used_ocr,
                        'used_ocrmypdf': used_ocr,  # Si usó OCR, probablemente usó OCRmyPDF si estaba disponible
                        'completed_at': datetime.now()
                    })

                    pdf_storage[pid].update({
                        'pages': pages,
                        'text_path': text_path,
                        'text': ''
                    })
                except Exception as e:
                    logger.exception(f"Error en procesamiento local {pid}: {e}")
                    pdf_task_status[pid].update({'status': 'failed', 'error': str(e)})

            thread = threading.Thread(
                target=_local_process, 
                args=(pdf_id, pdf_path, use_ocr, use_ocrmypdf, task.id),  # NUEVO: agregar use_ocrmypdf
                daemon=True
            )
            thread.start()

            return PDFUploadResponse(
                id=pdf_id,
                filename=file.filename,
                size=len(file_bytes),
                task_id=task.id,
                status="pending",
                message="PDF aceptado y procesando localmente (no hay workers).",
                estimated_wait_time=0.0
            )

        # Si hay workers, devolvemos response pendiente como antes
        return PDFUploadResponse(
            id=pdf_id,
            filename=file.filename,
            size=len(file_bytes),
            task_id=task.id,
            status="pending",
            message="PDF encolado para procesamiento. Usa el endpoint /upload-status/{pdf_id} para consultar el progreso",
            estimated_wait_time=10.0  # Estimado en segundos
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al encolar PDF: {str(e)}")

# ============================================================================
# NUEVOS ENDPOINTS PARA OCRmyPDF (NO MODIFICAN LOS EXISTENTES)
# ============================================================================

@router.post("/generate-ocr-pdf")
async def generate_ocr_pdf_endpoint(
    file: UploadFile = File(...),
    language: str = Query('spa', description="Idioma para OCR"),
    output_filename: Optional[str] = Query(None, description="Nombre del archivo de salida (opcional)")
):
    """
    GENERAR PDF CON OCR INCUBIERTO (searchable PDF)
    
    Diferencia con /upload:
    - /upload: Extrae texto y lo guarda en .txt
    - /generate-ocr-pdf: Genera un NUEVO PDF con texto OCR incrustado (searchable)
    
    Útil para: documentos escaneados, imágenes PDF, recibos, contratos, etc.
    """
    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Solo se permiten archivos PDF")
        
        file_bytes = await file.read()
        
        # Guardar archivo temporalmente
        temp_id = f"ocr_gen_{int(time.time())}"
        temp_path = os.path.join(settings.UPLOAD_FOLDER, f"{temp_id}.pdf")
        with open(temp_path, "wb") as f:
            f.write(file_bytes)
        
        # Definir ruta de salida
        if output_filename:
            if not output_filename.endswith('.pdf'):
                output_filename += '.pdf'
            output_path = os.path.join(settings.UPLOAD_FOLDER, output_filename)
        else:
            output_path = os.path.join(
                settings.UPLOAD_FOLDER, 
                f"{os.path.splitext(file.filename)[0]}_ocr.pdf"
            )
        
        # Encolar tarea para generar PDF con OCR
        task = generate_ocr_pdf_task.delay(
            input_pdf_path=temp_path,
            output_pdf_path=output_path,
            language=language
        )
        
        # Guardar información de la tarea
        ocr_task_id = f"ocr_pdf_{temp_id}"
        pdf_task_status[ocr_task_id] = {
            'task_id': task.id,
            'status': 'pending',
            'created_at': datetime.now(),
            'completed_at': None,
            'input_file': file.filename,
            'output_file': output_path,
            'language': language,
            'error': None
        }
        
        return {
            "message": "PDF en cola para generación con OCR",
            "task_id": task.id,
            "ocr_task_id": ocr_task_id,
            "input_file": file.filename,
            "output_file": os.path.basename(output_path),
            "status": "pending"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar PDF con OCR: {str(e)}")

@router.post("/full-process")
async def full_process_pdf_endpoint(
    file: UploadFile = File(...),
    use_ocr: bool = Query(True),
    generate_ocr_pdf: bool = Query(False, description="Generar PDF con OCR incrustado además de extraer texto"),
    language: str = Query('spa', description="Idioma para OCR")
):
    """
    PROCESO COMPLETO: Extrae texto Y opcionalmente genera PDF con OCR
    
    Combina ambas funcionalidades en una sola llamada:
    1. Extrae texto del PDF (guarda en .txt)
    2. Opcionalmente genera PDF con OCR incrustado (searchable PDF)
    
    Ideal para: Procesamiento completo de documentos escaneados
    """
    try:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Solo se permiten archivos PDF")
        
        file_bytes = await file.read()
        pdf_id = pdf_service.generate_pdf_id(file.filename, file_bytes)
        
        # Guardar archivo
        pdf_path = os.path.join(settings.UPLOAD_FOLDER, f"{pdf_id}.pdf")
        with open(pdf_path, "wb") as f:
            f.write(file_bytes)
        
        # Encolar tarea completa
        task = full_process_pdf_task.delay(
            pdf_id=pdf_id,
            pdf_path=pdf_path,
            use_ocr=use_ocr,
            generate_ocr_pdf=generate_ocr_pdf,
            language=language
        )
        
        # Guardar metadatos
        now = datetime.now()
        pdf_storage[pdf_id] = {
            'filename': file.filename,
            'pdf_path': pdf_path,
            'size': len(file_bytes),
            'upload_time': time.time(),
            'task_id': task.id,
            'use_ocr': use_ocr,
            'full_process': True,
            'generate_ocr_pdf': generate_ocr_pdf
        }
        
        pdf_task_status[pdf_id] = {
            'task_id': task.id,
            'status': 'pending',
            'created_at': now,
            'completed_at': None,
            'pages': None,
            'extracted_text_path': None,
            'ocr_pdf_path': None,
            'used_ocr': use_ocr,
            'generate_ocr_pdf': generate_ocr_pdf,
            'error': None
        }
        
        return {
            "message": "PDF en cola para procesamiento completo",
            "pdf_id": pdf_id,
            "task_id": task.id,
            "extract_text": True,
            "generate_ocr_pdf": generate_ocr_pdf,
            "status": "pending"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en procesamiento completo: {str(e)}")

@router.get("/ocr-capabilities")
async def get_ocr_capabilities():
    """
    OBTENER INFORMACIÓN SOBRE CAPACIDADES OCR DISPONIBLES
    
    Verifica si OCRmyPDF está instalado y disponible
    Útil para diagnóstico del sistema
    """
    try:
        capabilities = pdf_service.get_ocr_capabilities()
        
        return {
            "system": "PDF OCR Processing System",
            "timestamp": datetime.now().isoformat(),
            "capabilities": capabilities,
            "recommendation": "Usar OCRmyPDF para mejor calidad" if capabilities.get('ocrmypdf', {}).get('available') else "Usar Tesseract (OCRmyPDF no disponible)",
            "installation_help": "Instalar OCRmyPDF: pip install ocrmypdf" if not capabilities.get('ocrmypdf', {}).get('available') else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo capacidades OCR: {str(e)}")

@router.get("/download-ocr-pdf/{task_id}")
async def download_ocr_pdf(task_id: str):
    """
    DESCARGAR PDF CON OCR GENERADO
    
    Descarga el PDF con OCR incrustado generado por una tarea
    """
    # Buscar la tarea por ID
    task_info = None
    for tid, info in pdf_task_status.items():
        if info.get('task_id') == task_id:
            task_info = info
            break
    
    if not task_info:
        raise HTTPException(status_code=404, detail="Tarea no encontrada")
    
    # Verificar estado de la tarea
    task = celery_app.AsyncResult(task_id)
    
    if task.state == 'PENDING' or task.state == 'PROCESSING':
        return JSONResponse(
            status_code=202,
            content={
                "status": "processing",
                "message": "El PDF con OCR aún se está generando",
                "task_state": task.state
            }
        )
    
    if task.state == 'FAILURE':
        raise HTTPException(
            status_code=400,
            detail=f"Error generando PDF con OCR: {task.info}"
        )
    
    # Si la tarea completó, obtener ruta del archivo
    output_file = task_info.get('output_file')
    if not output_file or not os.path.exists(output_file):
        # Intentar obtener del resultado de la tarea
        if task.state == 'SUCCESS' and task.result:
            output_file = task.result.get('output_file')
        
        if not output_file or not os.path.exists(output_file):
            raise HTTPException(status_code=404, detail="Archivo PDF con OCR no encontrado")
    
    # Servir el archivo
    return FileResponse(
        output_file,
        media_type='application/pdf',
        filename=os.path.basename(output_file)
    )

# ============================================================================
# LOS ENDPOINTS EXISTENTES SIGUEN IGUAL (SIN CAMBIOS)
# ============================================================================

@router.get("/upload-status/{pdf_id}", response_model=PDFUploadStatus)
async def get_upload_status(pdf_id: str):
    """Obtiene el estado del procesamiento de un PDF"""
    if pdf_id not in pdf_task_status:
        raise HTTPException(status_code=404, detail="PDF no encontrado")
    
    task_status = pdf_task_status[pdf_id]
    task_id = task_status['task_id']
    
    # Consultar estado de la tarea en Celery
    task = celery_app.AsyncResult(task_id)
    
    # Mapear estados de Celery a nuestro esquema
    if task.state == 'PENDING':
        status = 'pending'
    elif task.state == 'PROCESSING':
        status = 'processing'
    elif task.state == 'SUCCESS':
        status = 'completed'
    elif task.state == 'FAILURE':
        status = 'failed'
    else:
        status = task.state.lower()
    
    # Actualizar estado en memoria
    pdf_task_status[pdf_id]['status'] = status
    
    # Si se completó, guardar la información adicional
    if task.state == 'SUCCESS' and task.result:
        result = task.result
        pdf_task_status[pdf_id].update({
            'pages': result.get('pages'),
            'extracted_text_path': result.get('text_path'),
            'used_ocr': result.get('used_ocr'),
            'completed_at': datetime.now()
        })
        
        # También actualizar pdf_storage con la información
        if pdf_id in pdf_storage:
            pdf_storage[pdf_id].update({
                'pages': result.get('pages'),
                'text_path': result.get('text_path'),
                'text': '',  # No cargamos todo el texto en memoria
            })
    
    # Si falló, guardar el error
    if task.state == 'FAILURE':
        pdf_task_status[pdf_id]['error'] = str(task.info)
    
    return PDFUploadStatus(
        pdf_id=pdf_id,
        task_id=task_id,
        status=status,
        pages=pdf_task_status[pdf_id].get('pages'),
        extracted_text_path=pdf_task_status[pdf_id].get('extracted_text_path'),
        used_ocr=pdf_task_status[pdf_id].get('used_ocr'),
        progress=100 if status == 'completed' else (0 if status == 'pending' else 50),
        error=pdf_task_status[pdf_id].get('error'),
        created_at=pdf_task_status[pdf_id].get('created_at'),
        completed_at=pdf_task_status[pdf_id].get('completed_at')
    )

@router.post("/{pdf_id}/search", response_model=SearchResponse)
async def search_pdf(pdf_id: str, search_request: SearchRequest):
    if pdf_id not in pdf_storage:
        raise HTTPException(status_code=404, detail="PDF no encontrado")
    
    start_time = time.time()
    pdf_data = pdf_storage[pdf_id]
    text = pdf_data['text']
    
    results = pdf_service.search_in_text(
        text, 
        search_request.term, 
        search_request.case_sensitive
    )
    
    limited_results = results[:100]
    execution_time = time.time() - start_time
    
    return SearchResponse(
        term=search_request.term,
        total_matches=len(results),
        results=limited_results,
        pdf_id=pdf_id,
        execution_time=execution_time
    )

@router.get("/{pdf_id}/text")
async def get_pdf_text(pdf_id: str):
    if pdf_id not in pdf_storage:
        raise HTTPException(status_code=404, detail="PDF no encontrado")
    
    # Verificar estado del PDF — preferir la fuente `pdf_task_status` cuando exista
    pdf_data = pdf_storage.get(pdf_id, {})
    task_status = pdf_task_status.get(pdf_id, {})
    current_status = task_status.get('status') or pdf_data.get('status')

    # Si el PDF aún está pendiente, devolver estado
    if current_status == 'pending' or (not current_status and task_status):
        return JSONResponse(
            status_code=202,  # Accepted - aún procesando
            content={
                'status': 'pending',
                'message': 'PDF aún está siendo procesado',
                'task_id': task_status.get('task_id') or pdf_data.get('task_id'),
                'progress': task_status.get('progress', pdf_data.get('progress', 0))
            }
        )

    # Si hay error, devolverlo
    if current_status == 'failed':
        return JSONResponse(
            status_code=400,
            content={
                'status': 'failed',
                'message': 'Error al procesar PDF',
                'error': task_status.get('error') or pdf_data.get('error')
            }
        )

    # Si el PDF está completado, devolver el texto (buscar en pdf_task_status o pdf_storage)
    text_path = task_status.get('extracted_text_path') or pdf_data.get('text_path')
    if text_path and os.path.exists(text_path):
        return FileResponse(
            text_path, 
            media_type='text/plain',
            filename=f"{pdf_id}_texto.txt"
        )
    else:
        raise HTTPException(status_code=404, detail="Texto no encontrado")

@router.get("/{pdf_id}/info", response_model=PDFInfo)
async def get_pdf_info(pdf_id: str):
    if pdf_id not in pdf_storage:
        raise HTTPException(status_code=404, detail="PDF no encontrado")
    
    pdf_data = pdf_storage[pdf_id]
    text_file_size = None
    text_path = pdf_data.get('text_path')
    if text_path and os.path.exists(text_path):
        text_file_size = os.path.getsize(text_path)
    
    return PDFInfo(
        id=pdf_id,
        filename=pdf_data['filename'],
        upload_date=datetime.fromtimestamp(pdf_data['upload_time']),
        size=pdf_data['size'],
        pages=pdf_data.get('pages'),  # Puede ser None si aún se procesa
        has_text=bool(pdf_data.get('text', '').strip()),
        text_file_size=text_file_size
    )

@router.get("/list", response_model=PDFListResponse)
async def list_pdfs():
    """
    Devuelve la lista de todos los PDFs con su estado de procesamiento.
    Útil para ver qué PDFs están en cola, en proceso o completados.
    """
    pdfs_list = []
    
    for pdf_id, data in pdf_storage.items():
        # Obtener estado de procesamiento
        task_status_info = pdf_task_status.get(pdf_id, {})
        status = task_status_info.get('status', 'unknown')
        task_id = task_status_info.get('task_id')
        
        # Si el PDF aún se está procesando, consultar estado real en Celery
        if status == 'pending' and task_id:
            task = celery_app.AsyncResult(task_id)
            if task.state == 'STARTED':
                status = 'processing'
            elif task.state == 'SUCCESS':
                status = 'completed'
            elif task.state == 'FAILURE':
                status = 'failed'
        
        # Calcular progreso basado en estado
        if status == 'completed':
            progress = 100
        elif status == 'processing':
            progress = 50
        elif status == 'failed':
            progress = 0
        else:  # pending, unknown
            progress = 0
        
        # Calcular tamaño en MB
        size_mb = round(data['size'] / (1024 * 1024), 2)
        
        pdfs_list.append({
            "id": pdf_id,
            "filename": data['filename'],
            "size_bytes": data['size'],
            "size_mb": size_mb,
            "status": status,
            "progress": progress,
            "pages": task_status_info.get('pages'),
            "task_id": task_id,
            "upload_time": data.get('upload_time'),
            "created_at": task_status_info.get('created_at'),
            "completed_at": task_status_info.get('completed_at'),
            "extracted_text_path": task_status_info.get('extracted_text_path'),
            "used_ocr": task_status_info.get('used_ocr'),
            "error": task_status_info.get('error')
        })
    
    # Agrupar por estado para facilitar visualización
    by_status = {
        'completed': [p for p in pdfs_list if p['status'] == 'completed'],
        'processing': [p for p in pdfs_list if p['status'] == 'processing'],
        'pending': [p for p in pdfs_list if p['status'] == 'pending'],
        'failed': [p for p in pdfs_list if p['status'] == 'failed'],
    }
    
    return {
        "total": len(pdf_storage),
        "by_status": {
            "completed": len(by_status['completed']),
            "processing": len(by_status['processing']),
            "pending": len(by_status['pending']),
            "failed": len(by_status['failed'])
        },
        "pdfs": pdfs_list,
        "summary": {
            "completed": by_status['completed'],
            "processing": by_status['processing'],
            "pending": by_status['pending'],
            "failed": by_status['failed']
        }
    }

@router.get("/dashboard")
async def get_dashboard():
    """
    Dashboard visual amigable para ver el estado de los PDFs.
    Devuelve información legible para los usuarios.
    """
    pdfs_info = []
    
    for pdf_id, data in pdf_storage.items():
        task_status_info = pdf_task_status.get(pdf_id, {})
        status = task_status_info.get('status', 'unknown')
        task_id = task_status_info.get('task_id')
        
        # Consultar estado real en Celery
        if task_id:
            task = celery_app.AsyncResult(task_id)
            if task.state == 'STARTED':
                status = 'processing'
                status_display = "En procesamiento"
            elif task.state == 'SUCCESS':
                status = 'completed'
                status_display = "Completado"
            elif task.state == 'FAILURE':
                status = 'failed'
                status_display = "Error"
            else:
                status_display = "En cola"
        else:
            status_display = "En cola"
        
        # Calcular progreso
        if status == 'completed':
            progress = 100
            progress_bar = "████████████████████ 100%"
        elif status == 'processing':
            progress = 50
            progress_bar = "██████████░░░░░░░░░░ 50%"
        else:
            progress = 0
            progress_bar = "░░░░░░░░░░░░░░░░░░░░ 0%"
        
        size_mb = round(data['size'] / (1024 * 1024), 2)
        
        pdfs_info.append({
            "numero": len(pdfs_info) + 1,
            "nombre_archivo": data['filename'],
            "tamaño_mb": size_mb,
            "estado": status_display,
            "progreso": f"{progress}%",
            "barra_progreso": progress_bar,
            "paginas": task_status_info.get('pages') or "Procesando...",
            "fecha_subida": task_status_info.get('created_at'),
            "fecha_completado": task_status_info.get('completed_at') or "Pendiente",
            "id_interno": pdf_id,
            "ruta_texto": task_status_info.get('extracted_text_path') or "No disponible",
            "ocr_usado": "Sí" if task_status_info.get('used_ocr') else "No",
            "error": task_status_info.get('error') or "Ninguno"
        })
    
    # Contar estados
    estados = {
        "completados": len([p for p in pdfs_info if "Completado" in p['estado']]),
        "procesando": len([p for p in pdfs_info if "En procesamiento" in p['estado']]),
        "en_cola": len([p for p in pdfs_info if "En cola" in p['estado']]),
        "con_error": len([p for p in pdfs_info if "Error" in p['estado']])
    }
    
    return {
        "titulo": "Dashboard de Procesamiento de PDFs",
        "total_pdfs": len(pdf_storage),
        "estados": estados,
        "pdfs": pdfs_info,
        "endpoints_ultiles": {
            "consultar_estado": "/api/pdf/upload-status/{pdf_id}",
            "buscar_en_pdf": "/api/pdf/search/{pdf_id}",
            "descargar_texto": "/api/pdf/{pdf_id}/text",
            "lista_detallada": "/api/pdf/list",
            "generar_pdf_ocr": "/api/pdf/generate-ocr-pdf",
            "proceso_completo": "/api/pdf/full-process"
        }
    }

@router.delete("/{pdf_id}")
async def delete_pdf(pdf_id: str):
    if pdf_id not in pdf_storage:
        raise HTTPException(status_code=404, detail="PDF no encontrado")
    
    pdf_data = pdf_storage[pdf_id]
    
    try:
        if os.path.exists(pdf_data['pdf_path']):
            os.remove(pdf_data['pdf_path'])
        if os.path.exists(pdf_data['text_path']):
            os.remove(pdf_data['text_path'])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error eliminando archivos: {str(e)}")
    
    del pdf_storage[pdf_id]
    
    return {"message": f"PDF {pdf_id} eliminado exitosamente"}

@router.post("/quick-search")
async def quick_search(
    file: UploadFile = File(...),
    search_term: str = Query(...),
    use_ocr: bool = Query(True)
):
    try:
        start_time = time.time()
        file_bytes = await file.read()
        
        temp_id = f"temp_{hash(file_bytes) % 1000000}"
        temp_path = os.path.join(settings.UPLOAD_FOLDER, f"{temp_id}.pdf")
        
        with open(temp_path, "wb") as f:
            f.write(file_bytes)
        
        text, pages, _ = pdf_service.extract_text_from_pdf(temp_path, use_ocr=use_ocr)
        results = pdf_service.search_in_text(text, search_term)
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        execution_time = time.time() - start_time
        
        return SearchResponse(
            term=search_term,
            total_matches=len(results),
            results=results[:50],
            pdf_id="temp",
            execution_time=execution_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en búsqueda rápida: {str(e)}")

@router.post("/global-search")
async def global_search(
    term: str = Query(..., description="Término de búsqueda"),
    case_sensitive: bool = Query(False, description="Coincidir mayúsculas/minúsculas"),
    context_chars: int = Query(100, description="Caracteres de contexto alrededor del match"),
    max_documents: int = Query(100, description="Máximo número de documentos a procesar")
):
    """
    Busca un término en TODOS los PDFs procesados y almacenados.
    Devuelve lista de documentos con coincidencias, ordenados por relevancia.
    """
    start_time = time.time()

    # Usar el método del servicio
    document_results = pdf_service.search_across_documents(
        search_term=term,
        case_sensitive=case_sensitive,
        context_chars=context_chars,
        max_documents=max_documents
    )

    # Calcular estadísticas globales
    total_matches = sum(len(doc['results']) for doc in document_results)
    total_documents_with_matches = len(document_results)

    # Enriquecer respuesta con filename desde pdf_storage (opcional)
    enriched_results = []
    for doc in document_results:
        pdf_id = doc['pdf_id']
        filename = pdf_storage.get(pdf_id, {}).get('filename', pdf_id)
        
        enriched_results.append({
            "pdf_id": pdf_id,
            "filename": filename,
            "total_matches": len(doc['results']),
            "results": doc['results'][:20],  # Limitar resultados por documento
            "score": sum(r['score'] for r in doc['results'])
        })

    execution_time = time.time() - start_time

    return {
        "term": term,
        "total_documents_with_matches": total_documents_with_matches,
        "total_matches": total_matches,
        "execution_time": execution_time,
        "documents": enriched_results
    }