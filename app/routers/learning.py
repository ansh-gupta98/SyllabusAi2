"""
Learning Router
===============
POST /api/v1/learning/next-lecture
  Input:  syllabus_id, last_topic, difficulty
  Output: full AI lecture for the next topic
"""

from fastapi import APIRouter, HTTPException, status

from app.models.schemas import APIResponse, LearningRequest, LectureContent
from app.services.llm_service import llm_service
from app.utils.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter()


@router.post(
    "/next-lecture",
    response_model=APIResponse,
    summary="Get next AI lecture",
    description=(
        "Pass the last completed topic. AI determines the next topic and "
        "delivers a full text-based lecture with key points, examples, "
        "and practice questions."
    ),
)
async def get_next_lecture(request: LearningRequest):
    logger.info(
        f"[Learning] syllabus_id={request.syllabus_id}, "
        f"last_topic='{request.last_topic}', difficulty={request.difficulty}"
    )

    # NOTE: In a full production system with a DB, you'd:
    #   1. Load the syllabus from DB by syllabus_id
    #   2. Find the topic after last_topic in the ordered list
    #   3. Pass that topic's subtopics to the LLM
    # For now (stateless), we pass what we have and let LLM infer.

    try:
        lecture_data = await llm_service.generate_lecture(
            topic_title  = f"Next topic after: {request.last_topic}",
            subtopics    = [],          # Frontend can pass these once DB is integrated
            difficulty   = request.difficulty.value,
            student_name = request.student_name,
            next_topic   = None,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))

    # Validate & build response
    try:
        lecture = LectureContent(
            topic_id               = "AUTO",
            topic_title            = lecture_data.get("topic_title", "Next Topic"),
            lecture_text           = lecture_data.get("lecture_text", ""),
            key_points             = lecture_data.get("key_points", []),
            examples               = lecture_data.get("examples", []),
            practice_questions     = lecture_data.get("practice_questions", []),
            estimated_read_minutes = int(lecture_data.get("estimated_read_minutes", 15)),
            next_topic_preview     = lecture_data.get("next_topic_preview"),
        )
    except Exception as e:
        logger.error(f"[Learning] Schema validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Lecture generated but response formatting failed. Please retry."
        )

    return APIResponse(
        success = True,
        message = f"Lecture ready: '{lecture.topic_title}' (~{lecture.estimated_read_minutes} min read)",
        data    = lecture.dict(),
    )