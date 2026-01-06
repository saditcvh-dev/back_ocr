from celery import shared_task
import logging
import os
from app.services.pdf_service import PDFService
from app.core.config import settings

logger = logging.getLogger(__name__)

@shared_task(bind=True, name="process_pdf_task")
def process_pdf_task(self, pdf_id: str, pdf_path: str, use_ocr: bool = True, generate_searchable_pdf: bool = False):
    """
    Tarea para procesar PDF en segundo plano.
    
    Modificado por: Nico
    Cambio: Agregado parámetro 'generate_searchable_pdf' y lógica de procesamiento condicional
    Motivo: Permitir generación de PDFs searchable con OCRmyPDF además de la extracción de texto normal
    """
    try:
        logger.info(f"[INICIO] Procesando PDF: {pdf_id} | Searchable: {generate_searchable_pdf}")
        
        # Actualizar estado a "procesando"
        self.update_state(
            state="PROCESSING",
            meta={'status': 'Extrayendo texto...', 'pdf_id': pdf_id}
        )
        
        # Crear instancia del servicio
        pdf_service = PDFService()
        
        # Procesar el PDF según el tipo solicitado
        searchable_pdf_path = None
        
        # Modificado por: Nico
        # Cambio: Lógica condicional para diferentes tipos de procesamiento
        # Motivo: Diferenciar entre extracción simple y generación de PDF con OCR incrustado
        if generate_searchable_pdf:
            logger.info(f"[OCRmyPDF] Generando PDF searchable...")
            
            # Verificar si OCRmyPDF está disponible
            if pdf_service._can_use_ocrmypdf():
                # Usar OCRmyPDF para generar PDF searchable
                text, pages, used_ocr, searchable_path = pdf_service.process_with_searchable_pdf(
                    pdf_path, 
                    pdf_id, 
                    use_ocr=use_ocr
                )
                
                if searchable_path:
                    searchable_pdf_path = searchable_path
                    logger.info(f"[OCRmyPDF] PDF searchable generado: {searchable_path}")
                else:
                    logger.warning("[OCRmyPDF] No se pudo generar PDF searchable, usando extracción normal")
                    
                    # Fallback a extracción normal
                    text, pages, used_ocr = pdf_service.extract_text_from_pdf(pdf_path, use_ocr=use_ocr)
            else:
                logger.warning("[OCRmyPDF] No disponible, usando extracción normal")
                
                # OCRmyPDF no disponible, usar extracción normal
                text, pages, used_ocr = pdf_service.extract_text_from_pdf(pdf_path, use_ocr=use_ocr)
        else:
            # Solo extraer texto normalmente (comportamiento original)
            logger.info(f"[OCR] Extrayendo texto con OCR={use_ocr}...")
            text, pages, used_ocr = pdf_service.extract_text_from_pdf(pdf_path, use_ocr=use_ocr)
        
        logger.info(f"[ÉXITO] Texto extraído: {len(text)} caracteres de {pages} páginas")
        
        # Guardar el texto extraído
        text_path = pdf_service.save_extracted_text(text, pdf_id)
        logger.info(f"[GUARDADO] Texto guardado en: {text_path}")
        
        # Devolver resultado con información adicional
        result = {
            'pdf_id': pdf_id,
            'pages': pages,
            'text_path': text_path,
            'used_ocr': used_ocr,
            'text_length': len(text),
            'status': 'completed',
            'generate_searchable_pdf': generate_searchable_pdf,
        }
        
        # Incluir ruta del PDF searchable si se generó
        if searchable_pdf_path:
            result['searchable_pdf_path'] = searchable_pdf_path
            
            # Verificar que el archivo existe
            if os.path.exists(searchable_pdf_path):
                result['searchable_file_size'] = os.path.getsize(searchable_pdf_path)
                logger.info(f"[PDF Searchable] Generado: {searchable_pdf_path} ({result['searchable_file_size']} bytes)")
            else:
                logger.warning(f"[PDF Searchable] Archivo no encontrado: {searchable_pdf_path}")
        
        logger.info(f"[COMPLETADO] PDF procesado exitosamente: {pdf_id}")
        return result
        
    except Exception as e:
        logger.exception(f"[ERROR] Fallo en procesamiento de {pdf_id}: {str(e)}")
        # Modificado por: Nico
        # Cambio: Incluir información del tipo de procesamiento en el error
        # Motivo: Facilitar debugging cuando falla la generación de PDF searchable
        error_info = {
            'pdf_id': pdf_id,
            'error': str(e),
            'generate_searchable_pdf': generate_searchable_pdf,
            'use_ocr': use_ocr
        }
        self.update_state(state="FAILURE", meta=error_info)
        raise