"""
Schedule Builder Service
========================
Takes structured topics from LLM and builds a proper
day-by-day calendar with alarms, revision days, etc.

This is a deterministic post-processor on top of LLM output.
It validates, repairs, and enriches the LLM-generated schedule.
"""

from __future__ import annotations

import uuid
from datetime import date, timedelta
from typing import List, Optional

from app.models.schemas import DailySchedule, SyllabusJSON, Topic
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

# Every N study days, insert a revision day
REVISION_EVERY_N_DAYS = 7
DEFAULT_ALARM_TIME    = "08:00"


class ScheduleBuilderService:

    def build_syllabus_response(
        self,
        llm_data: dict,
        target_date: str,
        daily_hours: float,
        subject_name: Optional[str] = None,
    ) -> SyllabusJSON:
        """
        Takes raw LLM JSON output, validates/repairs it, and
        returns a full SyllabusJSON ready to send to the frontend.
        """
        subject   = subject_name or llm_data.get("subject", "Unknown Subject")
        raw_topics    = llm_data.get("topics", [])
        raw_schedule  = llm_data.get("schedule", [])

        # ── Build Topic objects ──────────────────────────────────────────────
        topics: List[Topic] = []
        for i, t in enumerate(raw_topics):
            try:
                topics.append(Topic(
                    topic_id         = t.get("topic_id") or f"T{i+1:03d}",
                    title            = t.get("title", f"Topic {i+1}"),
                    subtopics        = t.get("subtopics", []),
                    estimated_hours  = float(t.get("estimated_hours", daily_hours)),
                    difficulty       = t.get("difficulty", "intermediate"),
                    resources        = t.get("resources", []),
                ))
            except Exception as e:
                logger.warning(f"Skipping malformed topic {i}: {e}")

        if not topics:
            raise ValueError("No valid topics found in LLM output.")

        total_hours = sum(t.estimated_hours for t in topics)
        start_date  = date.today()

        # ── Build or repair schedule ─────────────────────────────────────────
        schedule: List[DailySchedule] = []

        if raw_schedule:
            schedule = self._parse_llm_schedule(raw_schedule, topics)
        else:
            # LLM didn't return schedule — build it ourselves
            logger.warning("LLM did not return a schedule. Building fallback schedule.")
            schedule = self._build_fallback_schedule(
                topics, start_date, target_date, daily_hours
            )

        # Validate schedule completeness; fill gaps if needed
        schedule = self._ensure_completeness(schedule, topics, start_date, target_date, daily_hours)

        syllabus_id = str(uuid.uuid4())[:8].upper()

        return SyllabusJSON(
            syllabus_id           = syllabus_id,
            subject               = subject,
            total_topics          = len(topics),
            total_hours           = round(total_hours, 1),
            daily_learning_hours  = daily_hours,
            target_date           = target_date,
            start_date            = str(start_date),
            topics                = topics,
            schedule              = schedule,
            generated_at          = str(date.today()),
        )

    # ── Schedule helpers ─────────────────────────────────────────────────────

    def _parse_llm_schedule(self, raw: list, topics: List[Topic]) -> List[DailySchedule]:
        topic_map = {t.topic_id: t.title for t in topics}
        result = []
        for i, entry in enumerate(raw):
            try:
                t_ids    = entry.get("topics", [])
                t_titles = entry.get("topic_titles") or [topic_map.get(tid, tid) for tid in t_ids]
                result.append(DailySchedule(
                    date          = entry.get("date", ""),
                    day_number    = entry.get("day_number", i + 1),
                    topics        = t_ids,
                    topic_titles  = t_titles,
                    total_hours   = float(entry.get("total_hours", 2.0)),
                    alarm_time    = entry.get("alarm_time", DEFAULT_ALARM_TIME),
                    is_revision   = bool(entry.get("is_revision", False)),
                    notes         = entry.get("notes", ""),
                ))
            except Exception as e:
                logger.warning(f"Skipping malformed schedule entry {i}: {e}")
        return result

    def _build_fallback_schedule(
        self,
        topics: List[Topic],
        start_date: date,
        target_date: str,
        daily_hours: float,
    ) -> List[DailySchedule]:
        """Greedy scheduler: fill each day up to daily_hours budget."""
        end_date    = date.fromisoformat(target_date)
        current_day = start_date
        day_number  = 1
        topic_queue = list(topics)
        schedule    = []
        remaining_hours: dict[str, float] = {t.topic_id: t.estimated_hours for t in topics}

        while topic_queue and current_day <= end_date:
            # Revision day every N days
            if day_number > 1 and (day_number - 1) % REVISION_EVERY_N_DAYS == 0:
                prev_ids    = [s.topics[0] for s in schedule[-3:] if s.topics]
                prev_titles = [s.topic_titles[0] for s in schedule[-3:] if s.topic_titles]
                schedule.append(DailySchedule(
                    date         = str(current_day),
                    day_number   = day_number,
                    topics       = list(set(prev_ids))[:2],
                    topic_titles = list(set(prev_titles))[:2],
                    total_hours  = daily_hours,
                    alarm_time   = DEFAULT_ALARM_TIME,
                    is_revision  = True,
                    notes        = "📝 Revision day — review recent topics before moving on.",
                ))
                current_day += timedelta(days=1)
                day_number  += 1
                continue

            day_topics, day_titles, day_hours = [], [], 0.0
            budget = daily_hours

            while topic_queue and budget > 0:
                t = topic_queue[0]
                if remaining_hours[t.topic_id] <= budget:
                    day_topics.append(t.topic_id)
                    day_titles.append(t.title)
                    day_hours += remaining_hours[t.topic_id]
                    budget    -= remaining_hours[t.topic_id]
                    topic_queue.pop(0)
                else:
                    # Partial: split topic across days
                    day_topics.append(t.topic_id)
                    day_titles.append(f"{t.title} (cont.)")
                    day_hours += budget
                    remaining_hours[t.topic_id] -= budget
                    budget = 0

            if day_topics:
                schedule.append(DailySchedule(
                    date         = str(current_day),
                    day_number   = day_number,
                    topics       = day_topics,
                    topic_titles = day_titles,
                    total_hours  = round(day_hours, 1),
                    alarm_time   = DEFAULT_ALARM_TIME,
                    is_revision  = False,
                    notes        = "",
                ))

            current_day += timedelta(days=1)
            day_number  += 1

        return schedule

    def _ensure_completeness(
        self,
        schedule: List[DailySchedule],
        topics: List[Topic],
        start_date: date,
        target_date: str,
        daily_hours: float,
    ) -> List[DailySchedule]:
        """If LLM schedule is unusably short, rebuild it."""
        if len(schedule) < max(1, len(topics) // 3):
            logger.warning("LLM schedule too short — rebuilding with fallback scheduler.")
            return self._build_fallback_schedule(topics, start_date, target_date, daily_hours)
        return schedule


# Singleton
schedule_builder = ScheduleBuilderService()