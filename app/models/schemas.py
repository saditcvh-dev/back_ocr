from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime


# ===============================
# RESPUESTA AL SUBIR PDF (QUEUE)
# ===============================
class PDFUploadResponse(BaseModel):
    id: str
    filename: str
    size: int
    task_id: str  # ID de la tarea en Celery
    status: str  # "pending", "processing", "completed", "failed"
    message: str
    estimated_wait_time: Optional[float] = None  # segundos estimados


# ===============================
# ESTADO DEL PROCESAMIENTO
# ===============================
class PDFUploadStatus(BaseModel):
    pdf_id: str
    task_id: str
    status: str  # "pending", "processing", "completed", "failed"
    pages: Optional[int] = None
    extracted_text_path: Optional[str] = None
    used_ocr: Optional[bool] = None
    progress: Optional[int] = None  # 0-100
    error: Optional[str] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


# ===============================
# PETICIÓN DE BÚSQUEDA
# ===============================
class SearchRequest(BaseModel):
    term: str
    case_sensitive: bool = False
    use_regex: bool = False


# ===============================
# RESULTADO INDIVIDUAL
# ===============================
class SearchResult(BaseModel):
    page: int
    position: int
    context: str
    snippet: str


# ===============================
# RESPUESTA DE BÚSQUEDA
# ===============================
class SearchResponse(BaseModel):
    term: str
    total_matches: int
    results: List[SearchResult]
    pdf_id: str
    execution_time: float


# ===============================
# INFORMACIÓN GENERAL DEL PDF
# ===============================
class PDFInfo(BaseModel):
    id: str
    filename: str
    upload_date: datetime
    size: int
    pages: int
    has_text: bool
    text_file_size: Optional[int] = None


# ===============================
# 🔥 ANALYSIS (EVITA TU ERROR)
# ===============================
class PDFAnalysis(BaseModel):
    pdf_id: str
    pages: int
    used_ocr: bool
    extracted_text_path: Optional[str] = None


# ===============================
# ITEM INDIVIDUAL EN LISTA DE PDFs
# ===============================
class PDFListItem(BaseModel):
    id: str
    filename: str
    size_bytes: int
    size_mb: float
    status: str  # "completed", "processing", "pending", "failed"
    progress: int  # 0-100
    pages: Optional[int] = None
    task_id: str
    upload_time: float
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    extracted_text_path: Optional[str] = None
    used_ocr: bool
    error: Optional[str] = None


# ===============================
# RESPUESTA AGRUPADA DE LISTA DE PDFs
# ===============================
class PDFListResponse(BaseModel):
    total: int
    by_status: dict  # {"completed": 5, "processing": 2, "pending": 1, "failed": 0}
    pdfs: List[PDFListItem]
    summary: dict  # Contiene las listas agrupadas por estado

# ============================================================================
# NUEVOS ESQUEMAS PARA OCRmyPDF (AGREGADOS SIN MODIFICAR LOS EXISTENTES)
# ============================================================================

# ===============================
# GENERACIÓN DE PDF CON OCR
# ===============================
class GenerateOCRPDFRequest(BaseModel):
    """Parámetros para generar PDF con OCR incrustado"""
    language: str = "spa"
    output_filename: Optional[str] = None
    force_ocr: bool = True
    optimize: bool = True


class GenerateOCRPDFResponse(BaseModel):
    """Respuesta al generar PDF con OCR"""
    message: str
    task_id: str
    ocr_task_id: str
    input_file: str
    output_file: str
    status: str


class OCRPDFTaskStatus(BaseModel):
    """Estado de tarea de generación de PDF OCR"""
    task_id: str
    status: str  # "pending", "processing", "completed", "failed"
    input_file: str
    output_file: Optional[str] = None
    language: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    file_size: Optional[int] = None
    processing_time: Optional[float] = None


# ===============================
# PROCESO COMPLETO
# ===============================
class FullProcessPDFRequest(BaseModel):
    """Parámetros para proceso completo (extracción + generación PDF OCR)"""
    use_ocr: bool = True
    generate_ocr_pdf: bool = False
    language: str = "spa"


class FullProcessPDFResponse(BaseModel):
    """Respuesta para proceso completo"""
    message: str
    pdf_id: str
    task_id: str
    extract_text: bool
    generate_ocr_pdf: bool
    status: str


class FullProcessPDFResult(BaseModel):
    """Resultado del proceso completo"""
    pdf_id: str
    pages: int
    text_length: int
    extracted_text_path: Optional[str] = None
    ocr_pdf_path: Optional[str] = None
    ocr_pdf_generated: bool = False
    used_ocr: bool
    status: str
    processing_time: float


# ===============================
# CAPACIDADES OCR DEL SISTEMA
# ===============================
class OCRCapability(BaseModel):
    """Capacidad OCR individual"""
    name: str
    available: bool
    version: Optional[str] = None
    description: Optional[str] = None


class OCRCapabilitiesResponse(BaseModel):
    """Respuesta con capacidades OCR disponibles"""
    system: str
    timestamp: datetime
    capabilities: Dict[str, OCRCapability]
    recommendation: str
    installation_help: Optional[str] = None


# ===============================
# ANÁLISIS DE DOCUMENTO
# ===============================
class PDFAnalysisRequest(BaseModel):
    """Solicitud para analizar un PDF"""
    check_ocr_need: bool = True
    check_structure: bool = True


class PDFAnalysisResult(BaseModel):
    """Resultado del análisis de PDF"""
    pdf_id: str
    total_pages: int
    likely_scanned: bool
    has_text: bool
    recommend_ocrmypdf: bool
    processing_strategy: str  # "text_only", "ocr_only", "hybrid"
    recommended_dpi: int
    suggested_batch_size: int
    estimated_processing_time: float  # segundos
    page_sizes: List[Dict[str, float]] = []
    sample_text_pages: int = 0


# ===============================
# DESCARGAS
# ===============================
class DownloadOCRPDFResponse(BaseModel):
    """Respuesta para descargar PDF OCR"""
    status: str
    message: str
    task_state: Optional[str] = None
    download_url: Optional[str] = None
    filename: Optional[str] = None
    file_size: Optional[int] = None


# ===============================
# PROCESAMIENTO MASIVO
# ===============================
class BatchOCRPDFRequest(BaseModel):
    """Solicitud para procesamiento masivo"""
    input_dir: str
    output_dir: Optional[str] = None
    language: str = "spa"
    max_workers: int = 2
    recursive: bool = False


class BatchOCRPDFResponse(BaseModel):
    """Respuesta de procesamiento masivo"""
    message: str
    task_id: str
    input_dir: str
    output_dir: str
    total_files: int
    status: str


class BatchOCRPDFResult(BaseModel):
    """Resultado de procesamiento masivo"""
    success: int
    failed: int
    total: int
    processing_time: float
    success_files: List[str] = []
    failed_files: List[Dict[str, str]] = []  # [{"filename": "x.pdf", "error": "..."}]


# ===============================
# ESTADÍSTICAS DEL SISTEMA
# ===============================
class OCRSystemStats(BaseModel):
    """Estadísticas del sistema OCR"""
    total_pdfs_processed: int
    pdfs_with_ocr: int
    pdfs_with_ocrmypdf: int
    total_pages_processed: int
    avg_processing_time: float
    ocrmypdf_available: bool
    tesseract_available: bool
    last_processed: Optional[datetime] = None
    most_used_language: str = "spa"


# ===============================
# ERRORES ESPECÍFICOS
# ===============================
class OCRmyPDFError(BaseModel):
    """Error específico de OCRmyPDF"""
    error_type: str  # "installation", "execution", "timeout", "permission"
    message: str
    solution: Optional[str] = None
    command: Optional[str] = None
    exit_code: Optional[int] = None