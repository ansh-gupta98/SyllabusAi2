"""
HuggingFace LLM Service
========================
Single source-of-truth for all AI calls.
Uses HuggingFace Inference API (serverless) — no GPU needed on Railway.

Model: mistralai/Mistral-7B-Instruct-v0.3  (free tier, very capable)
Fallback: HuggingFaceH4/zephyr-7b-beta
"""

from __future__ import annotations

import os
import json
import re
import httpx
from typing import Optional

from app.utils.logger import setup_logger

logger = setup_logger(__name__)

HF_TOKEN      = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
HF_MODEL      = os.getenv(
    "HF_LLM_MODEL",
    "mistralai/Mistral-7B-Instruct-v0.3"
)
HF_API_BASE   = "https://api-inference.huggingface.co/models"
TIMEOUT_SECS  = 120   # HF cold-start can be slow


class HFLLMService:
    """Thin async wrapper around HuggingFace Inference API."""

    def __init__(self):
        if not HF_TOKEN:
            logger.warning("HUGGINGFACEHUB_API_TOKEN not set — AI calls will fail!")
        self._client = httpx.AsyncClient(timeout=TIMEOUT_SECS)
        self._headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json",
        }

    async def _call(self, prompt: str, max_new_tokens: int = 1500) -> str:
        """Raw call to HF Inference API. Returns generated text."""
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0.4,
                "top_p": 0.9,
                "do_sample": True,
                "return_full_text": False,
            },
            "options": {
                "wait_for_model": True,   # wait instead of 503 on cold start
                "use_cache": False,
            },
        }
        url = f"{HF_API_BASE}/{HF_MODEL}"
        try:
            resp = await self._client.post(url, json=payload, headers=self._headers)
            resp.raise_for_status()
            result = resp.json()
            # HF returns list[{generated_text: ...}]
            if isinstance(result, list) and result:
                return result[0].get("generated_text", "").strip()
            raise ValueError(f"Unexpected HF response format: {result}")
        except httpx.HTTPStatusError as e:
            logger.error(f"HF API HTTP error: {e.response.status_code} — {e.response.text}")
            raise RuntimeError(f"LLM service error: {e.response.status_code}")
        except Exception as e:
            logger.error(f"HF API call failed: {e}")
            raise RuntimeError(f"LLM service unavailable: {str(e)}")

    # ── Public methods ───────────────────────────────────────────────────────

    async def extract_syllabus_from_text(
        self,
        raw_text: str,
        subject_name: str,
        target_date: str,
        daily_hours: float,
        start_date: str,
    ) -> dict:
        """
        Given raw syllabus text, return structured JSON with topics + schedule.
        """
        prompt = f"""<s>[INST]
You are an expert academic planner AI. A student has provided their syllabus text.
Your job is to:
1. Extract all topics and subtopics from the syllabus.
2. Estimate hours needed per topic based on complexity.
3. Create a day-by-day study schedule from {start_date} to {target_date}.
4. Student can study {daily_hours} hours per day.

Subject: {subject_name}
Syllabus Text:
\"\"\"
{raw_text[:4000]}
\"\"\"

Return ONLY a valid JSON object (no markdown, no explanation) with this exact structure:
{{
  "subject": "<subject name>",
  "total_topics": <int>,
  "total_hours": <float>,
  "daily_learning_hours": {daily_hours},
  "topics": [
    {{
      "topic_id": "T001",
      "title": "<topic title>",
      "subtopics": ["<subtopic1>", "<subtopic2>"],
      "estimated_hours": <float>,
      "difficulty": "beginner|intermediate|advanced",
      "resources": ["<resource suggestion>"]
    }}
  ],
  "schedule": [
    {{
      "date": "YYYY-MM-DD",
      "day_number": <int>,
      "topics": ["T001"],
      "topic_titles": ["<title>"],
      "total_hours": <float>,
      "alarm_time": "08:00",
      "is_revision": false,
      "notes": "<optional note>"
    }}
  ]
}}
[/INST]"""

        raw = await self._call(prompt, max_new_tokens=3000)
        return self._parse_json_response(raw)

    async def generate_lecture(
        self,
        topic_title: str,
        subtopics: list[str],
        difficulty: str,
        student_name: Optional[str],
        next_topic: Optional[str],
    ) -> dict:
        """Generate a full AI lecture for a given topic."""
        name_str = f"Student name: {student_name}." if student_name else ""
        subtopics_str = ", ".join(subtopics) if subtopics else "general concepts"

        prompt = f"""<s>[INST]
You are EduAI, an expert teacher conducting a live lecture. {name_str}
Deliver a complete, engaging lecture on the following topic.

Topic: {topic_title}
Subtopics to cover: {subtopics_str}
Difficulty level: {difficulty}
Next topic after this: {next_topic or "End of syllabus"}

Return ONLY a valid JSON object (no markdown fences) with this structure:
{{
  "topic_title": "{topic_title}",
  "lecture_text": "<Full lecture in Markdown format. Use headings, bullet points, code blocks if needed. Minimum 500 words.>",
  "key_points": ["<point1>", "<point2>", "<point3>", "<point4>", "<point5>"],
  "examples": ["<example1>", "<example2>", "<example3>"],
  "practice_questions": ["<q1>", "<q2>", "<q3>"],
  "estimated_read_minutes": <int>,
  "next_topic_preview": "<1-2 sentence teaser about the next topic>"
}}
[/INST]"""

        raw = await self._call(prompt, max_new_tokens=2500)
        return self._parse_json_response(raw)

    async def solve_doubt(
        self,
        topic: str,
        doubt: str,
        context: Optional[str],
        student_name: Optional[str],
    ) -> dict:
        """Answer a student's specific doubt about a topic."""
        name_str = f"The student's name is {student_name}." if student_name else ""
        ctx_str  = f"\nAdditional context: {context}" if context else ""

        prompt = f"""<s>[INST]
You are EduAI, a patient and brilliant tutor. {name_str}
A student has a doubt about a topic. Answer it thoroughly, clearly, and with examples.

Topic: {topic}
Student's doubt: {doubt}{ctx_str}

Return ONLY a valid JSON object (no markdown fences) with this structure:
{{
  "topic": "{topic}",
  "doubt": "{doubt}",
  "answer": "<Detailed answer in Markdown. Use examples, analogies. Minimum 300 words.>",
  "related_concepts": ["<concept1>", "<concept2>", "<concept3>"],
  "follow_up_questions": ["<follow-up q1>", "<follow-up q2>"]
}}
[/INST]"""

        raw = await self._call(prompt, max_new_tokens=1500)
        return self._parse_json_response(raw)

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_json_response(raw: str) -> dict:
        """Robustly extract JSON from LLM output."""
        # Try direct parse first
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Strip markdown code fences
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Find first { ... } block
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        logger.error(f"Could not parse LLM JSON. Raw output (first 500 chars): {raw[:500]}")
        raise ValueError("LLM returned malformed JSON. Please retry.")


# Singleton — import this everywhere
llm_service = HFLLMService()