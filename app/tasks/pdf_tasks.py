from celery import shared_task
import logging
from app.services.pdf_service import PDFService
from app.core.config import settings

logger = logging.getLogger(__name__)

@shared_task(bind=True, name="process_pdf_task")
def process_pdf_task(self, pdf_id: str, pdf_path: str, use_ocr: bool = True, use_ocrmypdf: bool = True):
    """
    Tarea para procesar PDF en segundo plano.
    AGREGADO: Parámetro use_ocrmypdf para usar OCRmyPDF cuando esté disponible
    """
    try:
        logger.info(f"[INICIO] Procesando PDF: {pdf_id} - OCR={use_ocr}, OCRmyPDF={use_ocrmypdf}")
        
        # Actualizar estado a "procesando"
        self.update_state(
            state="PROCESSING",
            meta={'status': 'Extrayendo texto...', 'pdf_id': pdf_id}
        )
        
        # Crear instancia del servicio
        pdf_service = PDFService()
        
        # Procesar el PDF - USAR OCRmyPDF si está configurado
        logger.info(f"[OCR] Extrayendo texto con OCR={use_ocr}, OCRmyPDF={use_ocrmypdf}...")
        
        # Llamada actualizada para soportar OCRmyPDF
        text, pages, used_ocr = pdf_service.extract_text_from_pdf(
            pdf_path, 
            use_ocr=use_ocr,
            use_ocrmypdf=use_ocrmypdf  # NUEVO PARÁMETRO
        )
        
        logger.info(f"[ÉXITO] Texto extraído: {len(text)} caracteres de {pages} páginas")
        
        # Guardar el texto extraído
        text_path = pdf_service.save_extracted_text(text, pdf_id)
        logger.info(f"[GUARDADO] Texto guardado en: {text_path}")
        
        # Devolver resultado
        return {
            'pdf_id': pdf_id,
            'pages': pages,
            'text_path': text_path,
            'used_ocr': used_ocr,
            'text_length': len(text),
            'status': 'completed'
        }
        
    except Exception as e:
        logger.exception(f"[ERROR] Fallo en procesamiento de {pdf_id}: {str(e)}")
        raise

# ============================================================================
# NUEVA TAREA PARA GENERAR PDFs CON OCR INCUBIERTO
# ============================================================================

@shared_task(bind=True, name="generate_ocr_pdf_task")
def generate_ocr_pdf_task(self, input_pdf_path: str, output_pdf_path: str = None, 
                         language: str = 'spa'):
    """
    NUEVA TAREA: Generar PDF con OCR incrustado usando OCRmyPDF
    
    Diferencia con process_pdf_task:
    - process_pdf_task: Solo extrae texto y lo guarda en .txt
    - generate_ocr_pdf_task: Genera un NUEVO PDF con texto OCR incrustado (searchable PDF)
    
    Args:
        input_pdf_path: Ruta al PDF original
        output_pdf_path: Ruta para el PDF con OCR (opcional)
        language: Idioma para OCR (default: 'spa')
    """
    try:
        logger.info(f"[INICIO] Generando PDF OCR: {input_pdf_path}")
        
        # Actualizar estado
        self.update_state(
            state="PROCESSING",
            meta={'status': 'Generando PDF con OCR...', 'file': input_pdf_path}
        )
        
        # Crear instancia del servicio
        pdf_service = PDFService()
        
        # Generar PDF con OCR
        logger.info(f"[OCRmyPDF] Generando PDF searchable...")
        success, message = pdf_service.generate_ocr_pdf(
            input_pdf_path,
            output_pdf_path,
            language
        )
        
        if success:
            logger.info(f"[ÉXITO] {message}")
            return {
                'success': True,
                'message': message,
                'input_file': input_pdf_path,
                'output_file': output_pdf_path if output_pdf_path else f"{input_pdf_path}_ocr.pdf",
                'status': 'completed'
            }
        else:
            logger.error(f"[ERROR] {message}")
            return {
                'success': False,
                'message': message,
                'input_file': input_pdf_path,
                'status': 'failed'
            }
        
    except Exception as e:
        logger.exception(f"[ERROR] Fallo generando PDF OCR: {str(e)}")
        return {
            'success': False,
            'message': f"Error: {str(e)}",
            'status': 'failed'
        }

# ============================================================================
# TAREA PARA PROCESAMIENTO MASIVO CON OCRmyPDF
# ============================================================================

@shared_task(bind=True, name="batch_ocr_pdf_task")
def batch_ocr_pdf_task(self, input_dir: str, output_dir: str = None, 
                      language: str = 'spa', max_workers: int = 2):
    """
    TAREA MASIVA: Procesar múltiples PDFs en lote para generar PDFs con OCR
    
    Ideal para procesar carpetas completas de documentos escaneados
    
    Args:
        input_dir: Directorio con PDFs de entrada
        output_dir: Directorio para PDFs con OCR (opcional)
        language: Idioma para OCR
        max_workers: Número de procesos paralelos
    """
    try:
        logger.info(f"[INICIO] Procesamiento masivo: {input_dir}")
        
        # Actualizar estado
        self.update_state(
            state="PROCESSING",
            meta={'status': 'Procesando lote de PDFs...', 'input_dir': input_dir}
        )
        
        # Crear instancia del servicio
        pdf_service = PDFService()
        
        # Procesar en lote
        logger.info(f"[BATCH] Iniciando procesamiento masivo...")
        results = pdf_service.batch_generate_ocr_pdfs(
            input_dir,
            output_dir,
            language,
            max_workers
        )
        
        logger.info(f"[ÉXITO] Procesamiento masivo completado: {results}")
        return {
            'success': True,
            'results': results,
            'input_dir': input_dir,
            'output_dir': output_dir if output_dir else f"{input_dir}_ocr",
            'status': 'completed'
        }
        
    except Exception as e:
        logger.exception(f"[ERROR] Fallo en procesamiento masivo: {str(e)}")
        return {
            'success': False,
            'message': f"Error: {str(e)}",
            'status': 'failed'
        }

# ============================================================================
# TAREA DE DIAGNÓSTICO OCR
# ============================================================================

@shared_task(bind=True, name="check_ocr_capabilities_task")
def check_ocr_capabilities_task(self):
    """
    TAREA DE DIAGNÓSTICO: Verificar capacidades OCR disponibles
    
    Útil para verificar que OCRmyPDF está instalado y funcionando
    """
    try:
        logger.info("[DIAGNÓSTICO] Verificando capacidades OCR...")
        
        pdf_service = PDFService()
        capabilities = pdf_service.get_ocr_capabilities()
        
        logger.info(f"[DIAGNÓSTICO] Resultados: {capabilities}")
        return {
            'success': True,
            'capabilities': capabilities,
            'status': 'completed'
        }
        
    except Exception as e:
        logger.exception(f"[ERROR] Fallo en diagnóstico OCR: {str(e)}")
        return {
            'success': False,
            'message': f"Error: {str(e)}",
            'status': 'failed'
        }

# ============================================================================
# TAREA COMBINADA: EXTRACCIÓN + GENERACIÓN DE PDF OCR
# ============================================================================

@shared_task(bind=True, name="full_process_pdf_task")
def full_process_pdf_task(self, pdf_id: str, pdf_path: str, 
                         use_ocr: bool = True, 
                         generate_ocr_pdf: bool = False,
                         language: str = 'spa'):
    """
    TAREA COMBINADA: Extrae texto Y genera PDF con OCR si se solicita
    
    Combina la funcionalidad de ambas tareas:
    1. Extrae texto del PDF (a .txt)
    2. Opcionalmente genera PDF con OCR incrustado
    
    Args:
        pdf_id: ID único del PDF
        pdf_path: Ruta al PDF
        use_ocr: Usar OCR para extracción
        generate_ocr_pdf: Generar PDF con OCR incrustado
        language: Idioma para OCR
    """
    try:
        logger.info(f"[INICIO] Proceso completo para: {pdf_id}")
        
        # PASO 1: Extraer texto (tarea original)
        self.update_state(
            state="PROCESSING",
            meta={'status': 'Extrayendo texto...', 'pdf_id': pdf_id, 'step': 1}
        )
        
        pdf_service = PDFService()
        
        logger.info(f"[PASO 1] Extrayendo texto...")
        text, pages, used_ocr = pdf_service.extract_text_from_pdf(
            pdf_path, 
            use_ocr=use_ocr,
            use_ocrmypdf=True  # Usar OCRmyPDF para extracción
        )
        
        text_path = pdf_service.save_extracted_text(text, pdf_id)
        
        result = {
            'pdf_id': pdf_id,
            'pages': pages,
            'text_path': text_path,
            'used_ocr': used_ocr,
            'text_length': len(text),
            'ocr_pdf_generated': False,
            'ocr_pdf_path': None
        }
        
        # PASO 2: Generar PDF con OCR si se solicita
        if generate_ocr_pdf:
            self.update_state(
                state="PROCESSING",
                meta={'status': 'Generando PDF con OCR...', 'pdf_id': pdf_id, 'step': 2}
            )
            
            logger.info(f"[PASO 2] Generando PDF con OCR...")
            ocr_pdf_path = f"{pdf_path}_ocr.pdf"
            success, message = pdf_service.generate_ocr_pdf(
                pdf_path,
                ocr_pdf_path,
                language
            )
            
            if success:
                result['ocr_pdf_generated'] = True
                result['ocr_pdf_path'] = ocr_pdf_path
                result['ocr_pdf_message'] = message
                logger.info(f"[ÉXITO] PDF OCR generado: {ocr_pdf_path}")
            else:
                result['ocr_pdf_error'] = message
                logger.warning(f"[ADVERTENCIA] No se pudo generar PDF OCR: {message}")
        
        logger.info(f"[COMPLETADO] Proceso terminado para: {pdf_id}")
        result['status'] = 'completed'
        return result
        
    except Exception as e:
        logger.exception(f"[ERROR] Fallo en proceso completo: {str(e)}")
        raise