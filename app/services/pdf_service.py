# C:\Angular\OCR\pdf_api\app\services\pdf_service.py

import fitz  # PyMuPDF
import re
import os
import uuid
import hashlib
import logging
import concurrent.futures
from typing import List, Tuple, Dict
from datetime import datetime
from pathlib import Path
from bisect import bisect_left
from .ocr_service import OCRService
from app.core.config import settings
import gc
import gzip
import subprocess
import tempfile
import shutil

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFService:
    def __init__(self):
        self.ocr_service = OCRService(settings.TESSERACT_CMD)
        self.max_workers = min(4, os.cpu_count() or 2)  # Ajuste seguro de hilos
    
    def _extract_with_ocrmypdf(self, pdf_path: str, language: str = 'spa') -> Tuple[str, int, bool]:
        """Extrae texto usando ocrmypdf para OCR profesional"""
        temp_dir = None
        try:
            # Crear directorio temporal para procesamiento
            temp_dir = tempfile.mkdtemp(prefix="ocrmypdf_")
            original_path = Path(pdf_path)
            
            # Archivo de salida temporal
            ocr_output_path = Path(temp_dir) / f"{original_path.stem}_ocr.pdf"
            
            # Comando ocrmypdf optimizado
            cmd = [
                "ocrmypdf",
                "-l", language,
                "--force-ocr",
                "--deskew",
                "--clean",
                "--optimize", "1",
                "--skip-text",
                "--output-type", "pdf",
                "--jpeg-quality", "85",
                "--quiet",
                str(original_path),
                str(ocr_output_path)
            ]
            
            logger.info(f"Ejecutando ocrmypdf para: {original_path.name}")
            
            # Ejecutar ocrmypdf con timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutos timeout
            )
            
            # Manejo de resultados de ocrmypdf
            if result.returncode == 0:
                logger.info("ocrmypdf completado exitosamente")
            elif result.returncode in [1, 2, 3] and ocr_output_path.exists():
                logger.warning(f"ocrmypdf terminó con advertencias (código {result.returncode})")
                logger.info(f"Archivo generado a pesar de warnings: {ocr_output_path.name}")
            else:
                logger.warning(f"ocrmypdf falló ({result.returncode}): {result.stderr[:500]}")
                return self._extract_with_fallback_ocr(pdf_path, language)
            
            # Verificar que el archivo fue creado
            if not ocr_output_path.exists():
                logger.error("ocrmypdf no generó el archivo de salida")
                return self._extract_with_fallback_ocr(pdf_path, language)
            
            # Extraer texto del PDF con OCR aplicado
            doc = fitz.open(str(ocr_output_path))
            total_pages = len(doc)
            
            # Extraer texto de cada página
            text_parts = []
            for page_num in range(total_pages):
                page = doc.load_page(page_num)
                page_text = page.get_text("text") or ""
                text_parts.append(f"\n--- Página {page_num + 1} ---\n{page_text}")
                
                # Log de progreso
                if (page_num + 1) % 10 == 0:
                    logger.info(f"Extrayendo texto de página {page_num + 1}/{total_pages}")
            
            doc.close()
            text = "".join(text_parts)
            
            logger.info(f"Proceso completo. Páginas: {total_pages}, Texto extraído: {len(text)} caracteres")
            
            return text, total_pages, True  # used_ocr = True
            
        except subprocess.TimeoutExpired:
            logger.error("ocrmypdf timeout después de 5 minutos")
            return self._extract_with_fallback_ocr(pdf_path, language)
            
        except Exception as e:
            logger.error(f"Error en ocrmypdf: {str(e)}")
            return self._extract_with_fallback_ocr(pdf_path, language)
            
        finally:
            # Limpiar directorio temporal
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception as e:
                    logger.warning(f"No se pudo limpiar directorio temporal: {str(e)}")
    
    def _extract_with_fallback_ocr(self, pdf_path: str, language: str = 'spa') -> Tuple[str, int, bool]:
        """Fallback usando OCR normal cuando ocrmypdf falla"""
        logger.info(f"Usando OCR de fallback para: {pdf_path}")
        
        doc = None
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            text_parts = []
            used_ocr = False
            
            for page_num in range(total_pages):
                page = doc.load_page(page_num)
                page_text = page.get_text("text")
                
                if not page_text or len(page_text.strip()) < 10:
                    # Usar OCR para esta página
                    page_text = self._ocr_page(page, page_num, language)
                    if page_text.strip():
                        used_ocr = True
                
                text_parts.append(f"\n--- Página {page_num + 1} ---\n{page_text}")
                
                if (page_num + 1) % 10 == 0:
                    logger.info(f"Procesando página {page_num + 1}/{total_pages}")
            
            text = "".join(text_parts)
            return text, total_pages, used_ocr
            
        except Exception as e:
            logger.error(f"Error en OCR de fallback: {str(e)}")
            return "", 0, False
        finally:
            if doc:
                doc.close()
    
    def extract_text_from_pdf(self, pdf_path: str, use_ocr: bool = True,
                             language: str = 'spa', batch_size: int = 20,
                             use_ocrmypdf: bool = True) -> Tuple[str, int, bool]:
        """
        Extrae texto de PDF optimizado para velocidad y memoria
        Ahora soporta ocrmypdf como primera opción
        """
        # Primero intentar con ocrmypdf si está configurado
        if use_ocrmypdf and hasattr(self, '_extract_with_ocrmypdf'):
            try:
                logger.info(f"Intentando extracción con ocrmypdf para: {pdf_path}")
                return self._extract_with_ocrmypdf(pdf_path, language)
            except Exception as e:
                logger.warning(f"ocrmypdf no disponible o falló, usando método normal: {str(e)}")
        
        # Método normal (existente)
        doc = None
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            used_ocr = False

            logger.info(f"Iniciando extracción de {total_pages} páginas")

            if total_pages > 100:
                text, used_ocr = self._extract_large_pdf(doc, total_pages, use_ocr, language, batch_size)
            else:
                text, used_ocr = self._extract_small_pdf(doc, total_pages, use_ocr, language)

            return text, total_pages, used_ocr

        except Exception as e:
            logger.error(f"Error extrayendo texto: {str(e)}")
            return "", 0, False
        finally:
            if doc:
                doc.close()
            gc.collect()
    
    def extract_text_from_pdf_bytes(self, pdf_bytes: bytes, use_ocr: bool = True,
                                   language: str = 'spa', batch_size: int = 20,
                                   use_ocrmypdf: bool = True) -> Tuple[str, int, bool]:
        """Extracción directa desde bytes (sin tocar disco)"""
        # Para ocrmypdf necesitamos un archivo temporal
        if use_ocrmypdf and hasattr(self, '_extract_with_ocrmypdf'):
            temp_file = None
            try:
                # Crear archivo temporal
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
                    f.write(pdf_bytes)
                    temp_file = f.name
                
                logger.info(f"Procesando bytes con ocrmypdf usando archivo temporal")
                return self._extract_with_ocrmypdf(temp_file, language)
                
            except Exception as e:
                logger.warning(f"ocrmypdf falló para bytes, usando método normal: {str(e)}")
            finally:
                # Limpiar archivo temporal
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.unlink(temp_file)
                    except:
                        pass
        
        # Método normal para extracción desde bytes
        doc = None
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            total_pages = len(doc)

            if total_pages > 100:
                text, used_ocr = self._extract_large_pdf(doc, total_pages, use_ocr, language, batch_size)
            else:
                text, used_ocr = self._extract_small_pdf(doc, total_pages, use_ocr, language)

            return text, total_pages, used_ocr

        except Exception as e:
            logger.error(f"Error procesando PDF desde bytes: {str(e)}")
            return "", 0, False
        finally:
            if doc:
                doc.close()
            gc.collect()
    
    def _extract_small_pdf(self, doc: fitz.Document, total_pages: int,
                          use_ocr: bool, language: str) -> Tuple[str, bool]:
        """Extracción secuencial para PDFs pequeños"""
        text_parts = []
        used_ocr = False

        for page_num in range(total_pages):
            page = doc.load_page(page_num)
            page_text = page.get_text()

            if not page_text.strip() and use_ocr:
                page_text = self._ocr_page(page, page_num, language)
                if page_text.strip():
                    used_ocr = True

            text_parts.append(f"\n--- Página {page_num + 1} ---\n{page_text}")

            if (page_num + 1) % 10 == 0:
                logger.info(f"Procesadas {page_num + 1}/{total_pages} páginas")

        return "".join(text_parts), used_ocr

    def _extract_large_pdf(self, doc: fitz.Document, total_pages: int,
                          use_ocr: bool, language: str, batch_size: int) -> Tuple[str, bool]:
        """Extracción paralela por lotes para PDFs grandes"""
        text_parts = [""] * total_pages
        used_ocr = False

        def process_page(page_num: int):
            nonlocal used_ocr
            try:
                page = doc.load_page(page_num)
                page_text = page.get_text("text")

                if not page_text.strip() and use_ocr:
                    page_text = self._ocr_page(page, page_num, language)
                    if page_text.strip():
                        used_ocr = True

                return page_num, f"\n--- Página {page_num + 1} ---\n{page_text}"
            except Exception as e:
                logger.error(f"Error en página {page_num + 1}: {str(e)}")
                return page_num, f"\n--- Página {page_num + 1} ---\n[Error procesando página]"

        for batch_start in range(0, total_pages, batch_size):
            batch_end = min(batch_start + batch_size, total_pages)
            page_nums = range(batch_start, batch_end)

            logger.info(f"Procesando lote paralelo {batch_start + 1}-{batch_end}")

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for page_num, text in executor.map(process_page, page_nums):
                    text_parts[page_num] = text

            gc.collect()

        return "".join(text_parts), used_ocr

    def _ocr_page(self, page: fitz.Page, page_num: int, language: str) -> str:
        """OCR optimizado con DPI dinámico - Fallback a extracción básica si falla"""
        try:
            rect = page.rect
            # DPI más alto para texto pequeño, más bajo para páginas grandes
            base_dpi = 400 if max(rect.width, rect.height) < 1000 else 300
            zoom = base_dpi / 72
            mat = fitz.Matrix(zoom, zoom)

            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_bytes = pix.tobytes("png")

            return self.ocr_service.extract_text_from_image(img_bytes, language)

        except Exception as e:
            logger.warning(f"OCR falló en página {page_num + 1}: {str(e)}. Usando extracción básica...")
            # Fallback: intenta extraer texto sin OCR
            try:
                return page.get_text()
            except:
                return ""

    def search_in_text(self, text: str, search_term: str,
                      case_sensitive: bool = False, context_chars: int = 100) -> List[dict]:
        """Búsqueda rápida con detección eficiente de página"""
        if not text or not search_term:
            return []

        # Detectar separadores de página
        page_matches = list(re.finditer(r'--- Página (\d+) ---', text))
        if not page_matches:
            return []  # Sin estructura de páginas

        page_positions = [(int(m.group(1)), m.start()) for m in page_matches]
        positions = [pos for _, pos in page_positions]

        search_text = text if case_sensitive else text.lower()
        search_term_adj = search_term if case_sensitive else search_term.lower()

        results = []
        for match in re.finditer(re.escape(search_term_adj), search_text):
            start_pos = match.start()

            # Búsqueda binaria para encontrar página
            idx = bisect_left(positions, start_pos)
            if idx == 0:
                page_num = page_positions[0][0]
            elif idx >= len(positions):
                page_num = page_positions[-1][0]
            else:
                page_num = page_positions[idx - 1][0]

            start = max(0, start_pos - context_chars)
            end = min(len(text), match.end() + context_chars)

            context = text[start:end]
            snippet = text[match.start():match.end()]

            results.append({
                'page': page_num,
                'position': start_pos,
                'context': context,
                'snippet': snippet,
                'score': self._calculate_relevance_score(context, search_term)
            })

        results.sort(key=lambda x: x['score'], reverse=True)
        return results

    def _calculate_relevance_score(self, context: str, search_term: str) -> float:
        term_count = context.lower().count(search_term.lower())
        context_len = len(context)
        return (term_count * 10) + (100 / max(1, context_len / 100))

    def save_extracted_text(self, text: str, pdf_id: str) -> str:
        filename = f"{pdf_id}.txt"
        filepath = Path(settings.EXTRACTED_FOLDER) / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if len(text) > 10 * 1024 * 1024:
            import gzip
            with gzip.open(str(filepath) + '.gz', 'wt', encoding='utf-8') as f:
                f.write(text)
            final_path = str(filepath) + '.gz'
        else:
            filepath.write_text(text, encoding='utf-8')
            final_path = str(filepath)

        logger.info(f"Texto guardado en {final_path}")
        return final_path

    def generate_pdf_id(self, filename: str, file_bytes: bytes) -> str:
        """ID único basado en contenido (ideal para deduplicación)"""
        content_hash = hashlib.sha256(file_bytes).hexdigest()[:16]
        safe_name = re.sub(r'[^\w\-_]', '_', Path(filename).stem)[:30]
        return f"{safe_name}_{content_hash}"

    def get_pdf_info(self, pdf_path: str) -> dict:
        try:
            doc = fitz.open(pdf_path)
            info = {
                'pages': len(doc),
                'size_mb': round(os.path.getsize(pdf_path) / (1024 * 1024), 2),
                'has_text': False,
                'estimated_processing_time': round(len(doc) * 0.5 / 60, 2),
                'page_sizes': []
            }

            sample_pages = min(5, len(doc))
            for i in range(sample_pages):
                page = doc.load_page(i)
                if page.get_text().strip():
                    info['has_text'] = True
                rect = page.rect
                info['page_sizes'].append({'width': round(rect.width), 'height': round(rect.height)})

            doc.close()
            return info
        except Exception as e:
            logger.error(f"Error obteniendo info PDF: {str(e)}")
            return {}

    def analyze_pdf_structure(self, pdf_path: str) -> dict:
        try:
            doc = fitz.open(pdf_path)
            total = len(doc)
            analysis = {
                'total_pages': total,
                'likely_scanned': True,
                'recommended_dpi': 300,
                'suggested_batch_size': 15,
                'processing_strategy': 'hybrid',
                'recommend_ocrmypdf': True  # Recomendar usar ocrmypdf por defecto
            }

            sample_indices = [0, total // 2, total - 1] if total > 2 else [0]
            text_pages = 0

            for idx in sample_indices[:3]:
                if idx < total:
                    page_text = doc.load_page(idx).get_text()
                    if len(page_text.strip()) > 100:
                        text_pages += 1

            if text_pages == len(sample_indices):
                analysis.update({
                    'processing_strategy': 'text_only',
                    'likely_scanned': False,
                    'recommended_dpi': 150,
                    'recommend_ocrmypdf': False  # No necesita ocrmypdf si ya tiene texto
                })
            elif text_pages == 0:
                analysis.update({
                    'processing_strategy': 'ocr_only',
                    'likely_scanned': True,
                    'recommended_dpi': 350,
                    'suggested_batch_size': 8,
                    'recommend_ocrmypdf': True  # Altamente recomendado para PDFs escaneados
                })

            if total > 300:
                analysis['suggested_batch_size'] = 20
                analysis['recommend_ocrmypdf'] = True  # Para documentos grandes

            doc.close()
            return analysis
        except Exception as e:
            logger.error(f"Error analizando estructura: {str(e)}")
            return {}

    def search_across_documents(self, search_term: str, case_sensitive: bool = False, 
                               context_chars: int = 100, max_documents: int = 50) -> List[Dict]:
        """
        Busca en múltiples textos extraídos guardados en EXTRACTED_FOLDER.
        
        Args:
            search_term: Término de búsqueda
            case_sensitive: Sensible a mayúsculas
            context_chars: Caracteres de contexto
            max_documents: Máximo de documentos a procesar (para rendimiento)
        
        Returns:
            Lista de dicts con: pdf_id, filepath, results (lista de matches por doc)
        """
        results = []
        extracted_path = Path(settings.EXTRACTED_FOLDER)
        if not extracted_path.exists():
            logger.warning(f"Carpeta {extracted_path} no existe")
            return results

        # Listar archivos .txt y .txt.gz
        files = list(extracted_path.glob('*.txt')) + list(extracted_path.glob('*.txt.gz'))
        files = files[:max_documents]  # Limitar para evitar sobrecarga

        logger.info(f"Buscando en {len(files)} documentos")

        def load_text(filepath: Path) -> Tuple[str, str]:
            """Carga texto, manejando gz"""
            try:
                if filepath.suffix == '.gz':
                    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                        return f.read(), str(filepath)
                else:
                    return filepath.read_text(encoding='utf-8'), str(filepath)
            except Exception as e:
                logger.error(f"Error cargando {filepath}: {str(e)}")
                return "", ""

        # Cargar y buscar en paralelo
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(load_text, f): f for f in files}
            
            for future in concurrent.futures.as_completed(futures):
                text, filepath = future.result()
                if not text:
                    continue
                    
                pdf_id = Path(filepath).stem  # Quitar .txt o .txt.gz
                if pdf_id.endswith('.txt'):
                    pdf_id = pdf_id[:-4]  # Limpiar si es .txt.gz
                
                matches = self.search_in_text(
                    text, 
                    search_term, 
                    case_sensitive=case_sensitive, 
                    context_chars=context_chars
                )
                
                if matches:
                    results.append({
                        'pdf_id': pdf_id,
                        'filepath': filepath,
                        'results': matches
                    })

        # Ordenar por relevancia total (suma de scores en results)
        results.sort(key=lambda x: sum(r['score'] for r in x['results']), reverse=True)
        
        return results
    
    # METODOS NUEVOS PARA GENERAR PDFs CON OCR usando OCRmyPDF desde ocr_service
    
    def generate_ocr_pdf(self, input_pdf_path: str, output_pdf_path: str = None, 
                        language: str = 'spa') -> Tuple[bool, str]:
        """
        Genera un PDF con OCR incrustado usando OCRmyPDF
        Utiliza la funcionalidad agregada en ocr_service
        
        Args:
            input_pdf_path: Ruta al PDF de entrada
            output_pdf_path: Ruta de salida (opcional)
            language: Idioma para OCR
        
        Returns:
            Tuple[bool, str]: (éxito, mensaje)
        """
        try:
            input_path = Path(input_pdf_path)
            if not input_path.exists():
                return False, f"Archivo no encontrado: {input_pdf_path}"
            
            # Generar ruta de salida si no se proporciona
            if output_pdf_path is None:
                output_path = input_path.parent / f"{input_path.stem}_ocr.pdf"
            else:
                output_path = Path(output_pdf_path)
            
            # Usar la nueva funcionalidad de ocr_service
            success = self.ocr_service.generate_ocr_pdf_file(input_path, output_path, language)
            
            if success:
                return True, f"PDF con OCR generado: {output_path}"
            else:
                return False, "Error generando PDF con OCR"
                
        except Exception as e:
            logger.error(f"Error en generate_ocr_pdf: {str(e)}")
            return False, f"Error: {str(e)}"
    
    def generate_ocr_pdf_from_bytes(self, pdf_bytes: bytes, output_pdf_path: str,
                                   language: str = 'spa') -> Tuple[bool, str]:
        """
        Genera PDF con OCR desde bytes en memoria
        
        Args:
            pdf_bytes: Bytes del PDF
            output_pdf_path: Ruta donde guardar el PDF con OCR
            language: Idioma para OCR
        
        Returns:
            Tuple[bool, str]: (éxito, mensaje)
        """
        try:
            output_path = Path(output_pdf_path)
            
            # Usar la nueva funcionalidad de ocr_service
            success = self.ocr_service.generate_ocr_pdf_from_bytes(pdf_bytes, output_path, language)
            
            if success:
                return True, f"PDF con OCR generado: {output_path}"
            else:
                return False, "Error generando PDF con OCR desde bytes"
                
        except Exception as e:
            logger.error(f"Error en generate_ocr_pdf_from_bytes: {str(e)}")
            return False, f"Error: {str(e)}"
    
    def batch_generate_ocr_pdfs(self, input_dir: str, output_dir: str = None,
                               language: str = 'spa', max_workers: int = 2) -> dict:
        """
        Genera PDFs con OCR en lote (múltiples archivos)
        
        Args:
            input_dir: Directorio con PDFs de entrada
            output_dir: Directorio de salida (opcional)
            language: Idioma para OCR
            max_workers: Número de procesos paralelos
        
        Returns:
            dict: Estadísticas del procesamiento
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            return {"success": 0, "failed": 0, "total": 0, "error": "Directorio no existe"}
        
        # Directorio de salida
        if output_dir is None:
            output_path = input_path.parent / f"{input_path.name}_ocr"
        else:
            output_path = Path(output_dir)
        
        # Usar la nueva funcionalidad de ocr_service
        results = self.ocr_service.batch_generate_ocr_pdfs(input_path, output_path, language, max_workers)
        return results
    
    def get_ocr_capabilities(self) -> dict:
        """
        Obtiene información sobre las capacidades OCR disponibles
        
        Returns:
            dict: Información de OCRmyPDF y Tesseract
        """
        # Obtener info de OCRmyPDF desde ocr_service
        ocrmypdf_info = self.ocr_service.get_ocrmypdf_info()
        
        # Información del sistema
        capabilities = {
            'tesseract': {
                'available': hasattr(self.ocr_service, 'tesseract_config'),
                'language': getattr(self.ocr_service, 'language', 'spa')
            },
            'ocrmypdf': ocrmypdf_info,
            'recommendation': 'ocrmypdf' if ocrmypdf_info.get('available') else 'tesseract'
        }
        
        return capabilities