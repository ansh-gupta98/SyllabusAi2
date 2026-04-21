"""
EduAI Backend — Production-Ready FastAPI Application
=====================================================
AI-powered education platform with syllabus parsing,
schedule generation, AI lectures, and doubt solving.
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from app.routers import syllabus, learning, doubt, health
from app.utils.logger import setup_logger

# ── Logger ──────────────────────────────────────────────────────────────────
logger = setup_logger(__name__)


# ── Lifespan (startup / shutdown) ───────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 EduAI Backend starting up...")
    yield
    logger.info("🛑 EduAI Backend shutting down...")


# ── App factory ─────────────────────────────────────────────────────────────
def create_app() -> FastAPI:
    app = FastAPI(
        title="EduAI Platform API",
        description=(
            "AI-powered education backend: syllabus parsing (PDF/Image), "
            "schedule generation, AI lectures, and doubt solving."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── Middleware ───────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],          # tighten in prod: list your frontend URLs
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request timing middleware ────────────────────────────────────────────
    @app.middleware("http")
    async def add_process_time(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = round((time.perf_counter() - start) * 1000, 2)
        response.headers["X-Process-Time-Ms"] = str(elapsed)
        logger.info(f"{request.method} {request.url.path} → {response.status_code} ({elapsed}ms)")
        return response

    # ── Global exception handler ─────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled error on {request.url.path}: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Internal server error. Please try again later.",
                "path": str(request.url.path),
            },
        )

    # ── Routers ──────────────────────────────────────────────────────────────
    app.include_router(health.router,    prefix="/api/v1",           tags=["Health"])
    app.include_router(syllabus.router,  prefix="/api/v1/syllabus",  tags=["Syllabus"])
    app.include_router(learning.router,  prefix="/api/v1/learning",  tags=["Learning"])
    app.include_router(doubt.router,     prefix="/api/v1/doubt",     tags=["Doubt"])

    return app


app = create_app()