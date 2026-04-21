"""
Health Router
"""

import os
from datetime import datetime

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/health", summary="Health check")
async def health():
    return JSONResponse({
        "status": "ok",
        "service": "EduAI Backend",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "llm_model": os.getenv("HF_LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.3"),
        "hf_token_set": bool(os.getenv("HUGGINGFACEHUB_API_TOKEN")),
    })