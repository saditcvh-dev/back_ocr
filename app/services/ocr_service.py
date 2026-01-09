# C:\Angular\OCR\pdf_api\app\services\ocr_service.py

import pytesseract
from PIL import Image
import io
import concurrent.futures
import logging
from typing import List, Tuple, Optional
import numpy as np
from pdf2image import convert_from_bytes
from pathlib import Path
import subprocess
import shutil
import tempfile

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OCRService:
    def __init__(self, tesseract_cmd: str):
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
        # Configuración optimizada de Tesseract
        self.tesseract_config = {
            'lang': 'spa+eng',  # Español + Inglés como fallback
            'oem': 3,  # OEM_LSTM_ONLY - más rápido y preciso
            'psm': 3,  # PSM_AUTO - automático
            'dpi': 300,  # DPI óptimo para balance velocidad/calidad
        }
        
        # Configuración OCRmyPDF
        self.ocrmypdf_language = "spa"
        self._verify_ocrmypdf_installation()
        
        self.verify_tessdata()
    
    def _verify_ocrmypdf_installation(self):
        """Verifica que OCRmyPDF esté instalado en el sistema"""
        if not shutil.which("ocrmypdf"):
            logger.warning("OCRmyPDF no encontrado. Funcionalidad de PDF OCR limitada.")
            self.ocrmypdf_available = False
        else:
            logger.info("✅ OCRmyPDF está disponible para generar PDFs con OCR")
            self.ocrmypdf_available = True
    
    def verify_tessdata(self):
        """Verifica que los archivos de idioma estén disponibles"""
        try:
            langs = pytesseract.get_languages()
            logger.info(f"Idiomas Tesseract disponibles: {langs}")
            
            if 'spa' not in langs:
                logger.warning("Idioma 'spa' no encontrado, usando inglés como fallback")
        except Exception as e:
            logger.error(f"Error verificando idiomas Tesseract: {e}")
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocesa la imagen para mejorar OCR"""
        # Convertir a escala de grises si es necesario
        if image.mode != 'L':
            image = image.convert('L')
        
        # Mejorar contraste (opcional, solo si necesario)
        # from PIL import ImageEnhance
        # enhancer = ImageEnhance.Contrast(image)
        # image = enhancer.enhance(1.5)
        
        return image
    
    def extract_text_from_image(self, image_bytes: bytes, language: str = "spa") -> str:
        """Extrae texto de una imagen individual con manejo de errores"""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            
            # Preprocesar imagen
            image = self._preprocess_image(image)
            
            # Configurar parámetros para esta extracción
            config = self.tesseract_config.copy()
            config['lang'] = language if language else 'spa+eng'
            
            # Extraer texto
            text = pytesseract.image_to_string(
                image,
                config=f'--oem {config["oem"]} --psm {config["psm"]}'
            )
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error en OCR para imagen: {e}")
            return ""
    
    def process_page_batch(self, pages: List, start_page: int, language: str = "spa") -> List[Tuple[int, str]]:
        """Procesa un lote de páginas en paralelo"""
        results = []
        
        # Preparar tareas
        tasks = []
        for i, page in enumerate(pages):
            page_num = start_page + i
            image_bytes = page.tobytes("png")
            tasks.append((page_num, image_bytes))
        
        # Procesar en paralelo
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_page = {
                executor.submit(self.extract_text_from_image, img_bytes, language): page_num
                for page_num, img_bytes in tasks
            }
            
            for future in concurrent.futures.as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    text = future.result(timeout=60)  # Timeout por página
                    results.append((page_num, text))
                except concurrent.futures.TimeoutError:
                    logger.warning(f"Timeout en página {page_num}")
                    results.append((page_num, ""))
                except Exception as e:
                    logger.error(f"Error procesando página {page_num}: {e}")
                    results.append((page_num, ""))
        
        return results
    
    def extract_text_from_pdf_images(self, pdf_bytes: bytes, language: str = "spa", 
                                     batch_size: int = 10) -> str:
        """
        Extrae texto de PDF usando OCR con procesamiento por lotes
        
        Args:
            pdf_bytes: Bytes del PDF
            language: Idioma para OCR
            batch_size: Número de páginas por lote (optimizado para memoria)
        """
        try:
            # Convertir PDF a imágenes con configuración optimizada
            # DPI más bajo para documentos de texto, más alto para escaneados complejos
            dpi = 200  # Balance entre calidad y velocidad
            
            # Convertir en lotes para reducir uso de memoria
            all_text = []
            page_offset = 0
            
            # Determinar número total de páginas sin cargar todas las imágenes
            from pdf2image.pdf2image import pdfinfo_from_bytes
            info = pdfinfo_from_bytes(pdf_bytes)
            total_pages = info["Pages"]
            
            logger.info(f"Procesando PDF de {total_pages} páginas en lotes de {batch_size}")
            
            # Procesar por lotes
            for start_page in range(0, total_pages, batch_size):
                end_page = min(start_page + batch_size, total_pages)
                
                logger.info(f"Procesando páginas {start_page + 1} a {end_page}")
                
                # Convertir lote actual
                images = convert_from_bytes(
                    pdf_bytes,
                    dpi=dpi,
                    first_page=start_page + 1,
                    last_page=end_page,
                    thread_count=2,  # Reducir para evitar sobrecarga
                    use_pdftocairo=True,  # Usar pdftocairo que es más rápido
                    fmt='png',
                    grayscale=True  # Convertir a escala de grises directamente
                )
                
                # Procesar lote
                batch_results = self.process_page_batch(images, start_page, language)
                
                # Ordenar y agregar resultados
                batch_results.sort(key=lambda x: x[0])
                for page_num, page_text in batch_results:
                    all_text.append(f"\n--- Página {page_num + 1} ---\n{page_text}")
                
                # Liberar memoria
                del images
                
            return "\n".join(all_text)
            
        except Exception as e:
            logger.error(f"Error procesando PDF con OCR: {e}")
            return ""
    
    def extract_text_with_fallback(self, pdf_bytes: bytes, language: str = "spa") -> Tuple[str, bool]:
        """
        Extrae texto primero con PyMuPDF y usa OCR solo si es necesario
        """
        try:
            import fitz
            
            # Abrir PDF
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text_parts = []
            used_ocr = False
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Intentar extraer texto nativo primero
                page_text = page.get_text()
                
                # Si hay poco texto o está vacío, usar OCR
                if len(page_text.strip()) < 50:  # Umbral configurable
                    logger.info(f"Usando OCR en página {page_num + 1}")
                    pix = page.get_pixmap(dpi=200)
                    img_bytes = pix.tobytes("png")
                    page_text = self.extract_text_from_image(img_bytes, language)
                    used_ocr = True
                
                text_parts.append(f"\n--- Página {page_num + 1} ---\n{page_text}")
            
            doc.close()
            return "\n".join(text_parts), used_ocr
            
        except Exception as e:
            logger.error(f"Error en extracción con fallback: {e}")
            return "", False

    # ============================================================================
    # NUEVAS FUNCIONALIDADES OCRmyPDF (AGREGADAS SIN MODIFICAR LO EXISTENTE)
    # ============================================================================
    
    def generate_ocr_pdf_file(self, input_pdf_path: Path, output_pdf_path: Path, 
                              language: str = "spa", force_ocr: bool = True) -> bool:
        """
        GENERA PDF CON OCR INCUBIERTO USANDO OCRmyPDF
        
        Función nueva que NO modifica las existentes
        Propósito: Crear PDFs con texto OCR encima (searchable PDFs)
        
        Args:
            input_pdf_path: Ruta al PDF de entrada
            output_pdf_path: Ruta donde guardar el PDF con OCR
            language: Idioma para OCR (ej: 'spa', 'eng', 'spa+eng')
            force_ocr: Forzar OCR incluso si el PDF ya tiene texto
        
        Returns:
            bool: True si se generó exitosamente, False si falló
        """
        if not self.ocrmypdf_available:
            logger.error("❌ No se puede generar PDF OCR: OCRmyPDF no está instalado")
            logger.info("💡 Instala con: pip install ocrmypdf")
            return False
        
        if not input_pdf_path.exists():
            logger.error(f"❌ Archivo no encontrado: {input_pdf_path}")
            return False
        
        logger.info(f"🚀 Generando PDF con OCR: {input_pdf_path.name} -> {output_pdf_path.name}")
        
        try:
            # Crear directorio de salida si no existe
            output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Comando OCRmyPDF optimizado
            command = [
                "ocrmypdf",
                "-l", language,
                "--output-type", "pdf",      # PDF normal (no PDF/A)
                "--optimize", "1",           # Optimización leve
                "--jpeg-quality", "90",      # Calidad balanceada
                "--skip-text" if not force_ocr else "--force-ocr",
                "--deskew",                  # Enderezar páginas inclinadas
                "--clean",                   # Limpiar imágenes
                "--quiet",                   # Menos output en consola
                str(input_pdf_path),
                str(output_pdf_path)
            ]
            
            logger.debug(f"Ejecutando: {' '.join(command)}")
            
            # Ejecutar OCRmyPDF
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutos timeout
            )
            
            if result.returncode == 0:
                logger.info(f"✅ PDF con OCR generado exitosamente: {output_pdf_path}")
                return True
            elif result.returncode in [1, 2, 3] and output_pdf_path.exists():
                # OCRmyPDF puede terminar con warnings pero el archivo existe
                logger.warning(f"⚠️ OCRmyPDF terminó con advertencias (código {result.returncode})")
                logger.info(f"✅ PDF generado a pesar de warnings: {output_pdf_path}")
                return True
            else:
                logger.error(f"❌ Error en OCRmyPDF (código {result.returncode})")
                logger.error(f"Error: {result.stderr[:500]}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"⏰ Timeout: OCRmyPDF tardó más de 5 minutos")
            return False
        except Exception as e:
            logger.error(f"❌ Error inesperado generando PDF OCR: {e}")
            return False
    
    def generate_ocr_pdf_from_bytes(self, pdf_bytes: bytes, output_pdf_path: Path,
                                   language: str = "spa") -> bool:
        """
        GENERA PDF CON OCR DESDE BYTES EN MEMORIA
        
        Función nueva para procesar PDFs desde APIs o archivos en memoria
        
        Args:
            pdf_bytes: Bytes del PDF
            output_pdf_path: Ruta donde guardar el PDF con OCR
            language: Idioma para OCR
        
        Returns:
            bool: True si se generó exitosamente
        """
        # Crear archivo temporal
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(pdf_bytes)
        
        try:
            # Usar la función de archivo
            result = self.generate_ocr_pdf_file(temp_path, output_pdf_path, language)
            return result
        finally:
            # Limpiar archivo temporal
            try:
                temp_path.unlink()
            except:
                pass
    
    def batch_generate_ocr_pdfs(self, input_dir: Path, output_dir: Path,
                               language: str = "spa", max_workers: int = 2) -> dict:
        """
        GENERA PDFs CON OCR EN LOTE (múltiples archivos)
        
        Función nueva para procesamiento masivo
        
        Args:
            input_dir: Directorio con PDFs de entrada
            output_dir: Directorio donde guardar PDFs con OCR
            language: Idioma para OCR
            max_workers: Número de procesos paralelos
        
        Returns:
            dict: Estadísticas del procesamiento
        """
        if not input_dir.exists():
            logger.error(f"❌ Directorio no encontrado: {input_dir}")
            return {"success": 0, "failed": 0, "total": 0}
        
        # Crear directorio de salida
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Encontrar todos los PDFs
        pdf_files = list(input_dir.glob("*.pdf"))
        total_files = len(pdf_files)
        
        if total_files == 0:
            logger.warning(f"⚠️ No se encontraron PDFs en: {input_dir}")
            return {"success": 0, "failed": 0, "total": 0}
        
        logger.info(f"📊 Procesando {total_files} PDFs en lote...")
        
        results = {"success": 0, "failed": 0, "total": total_files}
        
        # Procesar en paralelo
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {}
            
            for pdf_file in pdf_files:
                output_file = output_dir / f"{pdf_file.stem}_ocr.pdf"
                future = executor.submit(
                    self.generate_ocr_pdf_file,
                    pdf_file,
                    output_file,
                    language
                )
                future_to_file[future] = pdf_file.name
            
            # Recoger resultados
            for future in concurrent.futures.as_completed(future_to_file):
                filename = future_to_file[future]
                try:
                    success = future.result(timeout=300)  # 5 minutos por archivo
                    if success:
                        results["success"] += 1
                        logger.info(f"✅ {filename} - OK")
                    else:
                        results["failed"] += 1
                        logger.error(f"❌ {filename} - Falló")
                except concurrent.futures.TimeoutError:
                    results["failed"] += 1
                    logger.error(f"⏰ {filename} - Timeout")
                except Exception as e:
                    results["failed"] += 1
                    logger.error(f"⚠️ {filename} - Error: {e}")
        
        logger.info(f"📊 Resumen lote: {results['success']}/{results['total']} exitosos")
        return results
    
    def get_ocrmypdf_info(self) -> dict:
        """
        OBTIENE INFORMACIÓN SOBRE OCRmyPDF INSTALADO
        
        Función nueva para diagnóstico
        
        Returns:
            dict: Información de la instalación
        """
        info = {
            "available": self.ocrmypdf_available,
            "version": None,
            "tesseract_version": None
        }
        
        if not self.ocrmypdf_available:
            return info
        
        try:
            # Obtener versión de OCRmyPDF
            result = subprocess.run(
                ["ocrmypdf", "--version"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                info["version"] = result.stdout.strip()
            
            # Obtener versión de Tesseract que usa OCRmyPDF
            result = subprocess.run(
                ["ocrmypdf", "--help"],
                capture_output=True,
                text=True
            )
            # Buscar información de Tesseract en el output
            for line in result.stdout.split('\n'):
                if "tesseract" in line.lower():
                    info["tesseract_info"] = line.strip()
                    break
                    
        except Exception as e:
            logger.warning(f"No se pudo obtener info de OCRmyPDF: {e}")
        
        return info

    # ============================================================================
    # FUNCIÓN CONVENIENCIA PARA USO RÁPIDO
    # ============================================================================
    
    def quick_ocr_pdf(self, input_path: str, output_path: str = None, 
                     language: str = "spa") -> Tuple[bool, str]:
        """
        FUNCIÓN CONVENIENCIA PARA GENERAR PDF OCR RÁPIDAMENTE
        
        Uso simple: ocr_service.quick_ocr_pdf("entrada.pdf", "salida_ocr.pdf")
        
        Args:
            input_path: Ruta al PDF de entrada
            output_path: Ruta de salida (opcional, genera automáticamente)
            language: Idioma para OCR
        
        Returns:
            Tuple[bool, str]: (éxito, mensaje/ruta)
        """
        input_pdf = Path(input_path)
        
        if not input_pdf.exists():
            return False, f"Archivo no encontrado: {input_path}"
        
        if output_path is None:
            output_pdf = input_pdf.parent / f"{input_pdf.stem}_ocr.pdf"
        else:
            output_pdf = Path(output_path)
        
        success = self.generate_ocr_pdf_file(input_pdf, output_pdf, language)
        
        if success:
            return True, f"PDF OCR generado: {output_pdf}"
        else:
            return False, "Error generando PDF OCR"
# ============================================================================
# FIN DE AGREGADOS OCRmyPDF
# Se mantuvo toda la funcionalidad existente de pytesseract
# Se agregaron nuevas funciones para generar PDFs con OCR
# Compatibilidad total: puedes usar ambos sistemas
# ============================================================================