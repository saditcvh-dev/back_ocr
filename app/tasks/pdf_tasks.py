from celery import shared_task
import logging
from app.services.pdf_service import PDFService
from app.core.config import settings

logger = logging.getLogger(__name__)

@shared_task(bind=True, name="process_pdf_task")
def process_pdf_task(self, pdf_id: str, pdf_path: str, use_ocr: bool = True):
    """
    Tarea para procesar PDF en segundo plano.
    """
    try:
        logger.info(f"[INICIO] Procesando PDF: {pdf_id}")
        
        # Actualizar estado a "procesando"
        self.update_state(
            state="PROCESSING",
            meta={'status': 'Extrayendo texto...', 'pdf_id': pdf_id}
        )
        
        # Crear instancia del servicio
        pdf_service = PDFService()
        
        # Procesar el PDF
        logger.info(f"[OCR] Extrayendo texto con OCR={use_ocr}...")
        text, pages, used_ocr = pdf_service.extract_text_from_pdf(
            pdf_path, 
            use_ocr=use_ocr
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
