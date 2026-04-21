"""
Syllabus Router
===============
POST /api/v1/syllabus/pdf    — PDF → structured syllabus JSON + schedule
POST /api/v1/syllabus/image  — Image → structured syllabus JSON + schedule
"""

from datetime import date

from fastapi import APIRouter, HTTPException, status

from app.models.schemas import (
    APIResponse,
    ImageSyllabusRequest,
    PDFSyllabusRequest,
    SyllabusJSON,
)
from app.services.llm_service import llm_service
from app.services.parser_service import doc_parser
from app.services.schedule_service import schedule_builder
from app.utils.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter()


async def _generate_syllabus(
    raw_text: str,
    target_date: str,
    daily_hours: float,
    subject_name: str | None,
) -> SyllabusJSON:
    """Shared logic for both PDF and Image routes."""
    start_date = str(date.today())
    resolved_subject = subject_name or "General Subject"

    # 1. Ask LLM to extract & structure topics + schedule
    llm_data = await llm_service.extract_syllabus_from_text(
        raw_text     = raw_text,
        subject_name = resolved_subject,
        target_date  = target_date,
        daily_hours  = daily_hours,
        start_date   = start_date,
    )

    # 2. Post-process & validate schedule
    syllabus = schedule_builder.build_syllabus_response(
        llm_data     = llm_data,
        target_date  = target_date,
        daily_hours  = daily_hours,
        subject_name = subject_name,
    )
    return syllabus


# ── PDF Endpoint ─────────────────────────────────────────────────────────────

@router.post(
    "/pdf",
    response_model=APIResponse,
    summary="Parse PDF syllabus → structured JSON with schedule",
    description=(
        "Upload a base64-encoded PDF. Returns a structured syllabus with "
        "topics, difficulty levels, estimated hours, and a day-by-day "
        "calendar schedule with alarm times."
    ),
)
async def pdf_to_syllabus(request: PDFSyllabusRequest):
    logger.info(f"[PDF→Syllabus] target_date={request.target_date}, daily_hours={request.daily_learning_hours}")

    # Step 1: Extract text from PDF
    try:
        raw_text, page_count = doc_parser.extract_text_from_pdf_base64(request.pdf_base64)
        logger.info(f"[PDF→Syllabus] Extracted {len(raw_text)} chars from {page_count} pages.")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    # Step 2: Generate structured syllabus
    try:
        syllabus = await _generate_syllabus(
            raw_text     = raw_text,
            target_date  = request.target_date,
            daily_hours  = request.daily_learning_hours,
            subject_name = request.subject_name,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))

    return APIResponse(
        success = True,
        message = f"Syllabus generated: {syllabus.total_topics} topics across {len(syllabus.schedule)} days.",
        data    = syllabus.dict(),
    )


# ── Image Endpoint ────────────────────────────────────────────────────────────

@router.post(
    "/image",
    response_model=APIResponse,
    summary="Parse Image syllabus → structured JSON with schedule",
    description=(
        "Upload a base64-encoded image (PNG/JPG/WEBP) of a handwritten or "
        "printed syllabus. OCR extracts the text, then AI structures it."
    ),
)
async def image_to_syllabus(request: ImageSyllabusRequest):
    logger.info(f"[Image→Syllabus] target_date={request.target_date}, mime={request.image_mime_type}")

    # Step 1: OCR
    try:
        raw_text = doc_parser.extract_text_from_image_base64(
            request.image_base64, request.image_mime_type
        )
        logger.info(f"[Image→Syllabus] OCR extracted {len(raw_text)} chars.")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    # Step 2: Generate structured syllabus
    try:
        syllabus = await _generate_syllabus(
            raw_text     = raw_text,
            target_date  = request.target_date,
            daily_hours  = request.daily_learning_hours,
            subject_name = request.subject_name,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))

    return APIResponse(
        success = True,
        message = f"Syllabus generated from image: {syllabus.total_topics} topics across {len(syllabus.schedule)} days.",
        data    = syllabus.dict(),
    )