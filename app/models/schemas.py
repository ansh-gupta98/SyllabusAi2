"""
Pydantic Models — Request & Response Schemas
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


# ── Enums ────────────────────────────────────────────────────────────────────

class DifficultyLevel(str, Enum):
    BEGINNER     = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED     = "advanced"


class LectureStatus(str, Enum):
    PENDING    = "pending"
    COMPLETED  = "completed"
    SKIPPED    = "skipped"


# ── Shared ───────────────────────────────────────────────────────────────────

class APIResponse(BaseModel):
    """Standard envelope for every API response."""
    success: bool
    message: str
    data: Optional[Any] = None
    error: Optional[str] = None


# ── Syllabus ─────────────────────────────────────────────────────────────────

class Topic(BaseModel):
    topic_id:    str   = Field(..., description="Unique ID e.g. 'T001'")
    title:       str
    subtopics:   List[str] = []
    estimated_hours: float = Field(..., description="Estimated hours to study this topic")
    difficulty:  DifficultyLevel = DifficultyLevel.INTERMEDIATE
    resources:   List[str] = Field(default=[], description="Suggested resources/links")


class DailySchedule(BaseModel):
    date:          str   = Field(..., description="ISO date YYYY-MM-DD")
    day_number:    int
    topics:        List[str] = Field(..., description="List of topic_ids for the day")
    topic_titles:  List[str]
    total_hours:   float
    alarm_time:    str   = Field(..., description="Suggested alarm time HH:MM")
    is_revision:   bool  = False
    notes:         str   = ""


class SyllabusJSON(BaseModel):
    syllabus_id:        str
    subject:            str
    total_topics:       int
    total_hours:        float
    daily_learning_hours: float
    target_date:        str
    start_date:         str
    topics:             List[Topic]
    schedule:           List[DailySchedule]
    generated_at:       str


class PDFSyllabusRequest(BaseModel):
    """Used when frontend sends base64-encoded PDF."""
    pdf_base64:         str  = Field(..., description="Base64-encoded PDF content")
    target_date:        str  = Field(..., description="Target completion date YYYY-MM-DD")
    daily_learning_hours: float = Field(default=3.0, ge=0.5, le=12.0,
                                        description="Hours student can study per day")
    subject_name:       Optional[str] = Field(None, description="Subject name override")

    @validator("target_date")
    def validate_target_date(cls, v):
        try:
            d = date.fromisoformat(v)
            if d <= date.today():
                raise ValueError("target_date must be in the future")
            return v
        except ValueError as e:
            raise ValueError(str(e))


class ImageSyllabusRequest(BaseModel):
    """Used when frontend sends base64-encoded image."""
    image_base64:       str  = Field(..., description="Base64-encoded image (PNG/JPG/WEBP)")
    image_mime_type:    str  = Field(default="image/jpeg",
                                    description="MIME type: image/jpeg, image/png, image/webp")
    target_date:        str  = Field(..., description="Target completion date YYYY-MM-DD")
    daily_learning_hours: float = Field(default=3.0, ge=0.5, le=12.0)
    subject_name:       Optional[str] = None

    @validator("target_date")
    def validate_target_date(cls, v):
        try:
            d = date.fromisoformat(v)
            if d <= date.today():
                raise ValueError("target_date must be in the future")
            return v
        except ValueError as e:
            raise ValueError(str(e))

    @validator("image_mime_type")
    def validate_mime(cls, v):
        allowed = {"image/jpeg", "image/png", "image/webp", "image/gif"}
        if v not in allowed:
            raise ValueError(f"Unsupported MIME type. Allowed: {allowed}")
        return v


# ── Learning / Lecture ────────────────────────────────────────────────────────

class LearningRequest(BaseModel):
    """
    Request the next lecture.
    Pass the last completed topic; AI returns the next topic in full detail.
    """
    syllabus_id:  str  = Field(..., description="Reference to the generated syllabus")
    last_topic:   str  = Field(..., description="Title or topic_id of the last completed topic")
    student_name: Optional[str] = Field(None, description="Used to personalise the lecture")
    difficulty:   DifficultyLevel = DifficultyLevel.INTERMEDIATE


class LectureContent(BaseModel):
    topic_id:        str
    topic_title:     str
    lecture_text:    str  = Field(..., description="Full AI-generated lecture in Markdown")
    key_points:      List[str]
    examples:        List[str]
    practice_questions: List[str]
    estimated_read_minutes: int
    next_topic_preview: Optional[str] = None


# ── Doubt ─────────────────────────────────────────────────────────────────────

class DoubtRequest(BaseModel):
    topic:       str  = Field(..., description="Topic the doubt is about")
    doubt:       str  = Field(..., description="The student's specific question/doubt")
    context:     Optional[str] = Field(None,
                    description="Optional extra context (e.g. previous explanation seen)")
    student_name: Optional[str] = None


class DoubtAnswer(BaseModel):
    topic:           str
    doubt:           str
    answer:          str  = Field(..., description="Detailed AI answer in Markdown")
    related_concepts: List[str]
    follow_up_questions: List[str]
    answered_at:     str