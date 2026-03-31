from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

import importlib
import importlib.util
import os

from services.qdrant_service import qdrant_service
from services.rate_limiter import (
    RATE_LIMITING_AVAILABLE,
    RateLimitExceeded,
    SlowAPIMiddleware,
    limiter,
    rate_limit_exceeded_handler,
)
from services.security_limits import increment_rate_counter, token_subject
from services.settings import configure_logging, get_settings

configure_logging()
logger = logging.getLogger("ahvi.main")
try:
    settings = get_settings()
except Exception as exc:
    logger.exception("Startup configuration validation failed")
    raise RuntimeError(f"Startup configuration validation failed: {exc}") from exc


# -------------------------
# OPTIONAL ROUTER LOADER
# -------------------------
def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _load_optional_router(module_name: str, attr: str = "router"):
    if not _has_module(module_name):
        logger.error("Router module not found: %s", module_name)
        if settings.is_production:
            raise RuntimeError(f"CRITICAL MVP FAILURE: Router {module_name} failed to load (module not found).")
        return None
    try:
        module = importlib.import_module(module_name)
        return getattr(module, attr)
    except Exception as exc:
        logger.exception("%s failed", module_name)
        if settings.is_production:
            raise RuntimeError(f"CRITICAL MVP FAILURE: Router {module_name} failed to load.") from exc
        return None


# -------------------------
# LOAD ALL ROUTERS (SAFE)
# -------------------------
chat_router = _load_optional_router("routers.chat")
data_router = _load_optional_router("routers.data")
utilities_router = _load_optional_router("routers.utilities")
boards_router = _load_optional_router("routers.boards")
feedback_router = _load_optional_router("routers.feedback")

# AI
ai_router = _load_optional_router("api.ai")

# Optional
stylist_router = _load_optional_router("routers.stylist")
reddit_router = _load_optional_router("routers.reddit")

# Feature-based
bg_router = None
if os.getenv("ENABLE_BG_REMOVER", "false").lower() in ("1", "true", "yes"):
    bg_required_modules = ["transformers", "torch", "torchvision", "PIL"]
    missing_bg_modules = [m for m in bg_required_modules if not _has_module(m)]
    if missing_bg_modules:
        logger.error("bg_remover dependencies missing: %s", ", ".join(missing_bg_modules))
        if settings.is_production:
            raise RuntimeError("CRITICAL MVP FAILURE: bg_remover dependencies missing.")
    else:
        bg_router = _load_optional_router("routers.bg_remover")

vision_router = None
if os.getenv("ENABLE_VISION", "false").lower() in ("1", "true", "yes"):
    vision_required_modules = ["cv2", "sklearn", "numpy"]
    missing_vision_modules = [m for m in vision_required_modules if not _has_module(m)]
    if missing_vision_modules:
        logger.error("vision dependencies missing: %s", ", ".join(missing_vision_modules))
        if settings.is_production:
            raise RuntimeError("CRITICAL MVP FAILURE: vision dependencies missing.")
    else:
        vision_router = _load_optional_router("routers.vision")
wardrobe_capture_router = None
if os.getenv("ENABLE_VISION", "false").lower() in ("1", "true", "yes"):
    capture_required_modules = ["cv2", "numpy", "PIL"]
    missing_capture_modules = [m for m in capture_required_modules if not _has_module(m)]
    if missing_capture_modules:
        logger.error("wardrobe_capture dependencies missing: %s", ", ".join(missing_capture_modules))
        if settings.is_production:
            raise RuntimeError("CRITICAL MVP FAILURE: wardrobe_capture dependencies missing.")
    else:
        wardrobe_capture_router = _load_optional_router("routers.wardrobe_capture")

garment_router = None
if os.getenv("ENABLE_GARMENT_ANALYZER", "false").lower() in ("1", "true", "yes"):
    garment_required_modules = ["transformers", "PIL", "cv2", "sklearn", "numpy"]
    missing_garment_modules = [m for m in garment_required_modules if not _has_module(m)]
    if missing_garment_modules:
        logger.error("garment_analyzer dependencies missing: %s", ", ".join(missing_garment_modules))
        if settings.is_production:
            raise RuntimeError("CRITICAL MVP FAILURE: garment_analyzer dependencies missing.")
    else:
        garment_router = _load_optional_router("routers.garment_analyzer")


# -------------------------
# OPTIONAL IMPORTS
# -------------------------
try:
    from celery.result import AsyncResult
except Exception:
    AsyncResult = None

try:
    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration
except Exception:
    sentry_sdk = None
    FastApiIntegration = None

try:
    from worker import celery_app
except Exception:
    celery_app = None


# -------------------------
# SENTRY
# -------------------------
_sentry_dsn = os.getenv("SENTRY_DSN")
if _sentry_dsn and sentry_sdk and FastApiIntegration:
    sentry_sdk.init(
        dsn=_sentry_dsn,
        traces_sample_rate=1.0,
        send_default_pii=False,
        integrations=[FastApiIntegration()],
    )


# -------------------------
# APP INIT
# -------------------------
app = FastAPI(
    title="AHVI AI Master Brain API",
    version="2.2.0",
    docs_url="/docs" if settings.docs_enabled else None,
    redoc_url="/redoc" if settings.docs_enabled else None,
    openapi_url="/openapi.json" if settings.docs_enabled else None,
)

if RATE_LIMITING_AVAILABLE:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

logger.info("AHVI backend started")


_HEAVY_UPLOAD_PATHS = (
    "/api/bg-remove",
    "/api/background/remove-bg",
    "/api/background/remove-bg/async",
    "/api/vision/analyze-image",
    "/api/vision/analyze-image/async",
    "/api/wardrobe/capture/analyze",
    "/api/wardrobe/capture/analyze/async",
    "/api/wardrobe/capture/process-upload/async",
    "/api/uploads/avatar",
    "/api/uploads/wardrobe",
    "/api/garment/analyze",
    "/api/garment/analyze/",
)

_API_PREFIXES = ("/api", "/garment")


@app.middleware("http")
async def upload_size_guard(request: Request, call_next):
    path = request.url.path or ""
    if any(path.startswith(p) for p in _API_PREFIXES):
        client_ip = (request.client.host if request.client else "") or "unknown"
        if increment_rate_counter("ip", client_ip, limit=60, window_seconds=60):
            return JSONResponse(
                status_code=429,
                content={"success": False, "error": {"code": "IP_RATE_LIMITED", "message": "Too many requests from this IP"}},
            )

        auth_header = request.headers.get("authorization", "")
        token_key = token_subject(auth_header)
        if token_key and increment_rate_counter("user_token", token_key, limit=100, window_seconds=60):
            return JSONResponse(
                status_code=429,
                content={"success": False, "error": {"code": "USER_RATE_LIMITED", "message": "Too many requests for this user token"}},
            )

    if request.method.upper() in {"POST", "PUT", "PATCH"}:
        if any(path.startswith(p) for p in _HEAVY_UPLOAD_PATHS):
            cl_header = request.headers.get("content-length", "").strip()
            if cl_header:
                try:
                    content_length = int(cl_header)
                except Exception:
                    return JSONResponse(
                        status_code=400,
                        content={"success": False, "error": {"code": "BAD_CONTENT_LENGTH", "message": "Invalid content-length header"}},
                    )
                if content_length > int(settings.MAX_UPLOAD_BYTES):
                    return JSONResponse(
                        status_code=413,
                        content={
                            "success": False,
                            "error": {
                                "code": "REQUEST_TOO_LARGE",
                                "message": f"Payload exceeds max allowed size of {int(settings.MAX_UPLOAD_BYTES)} bytes",
                            },
                        },
                    )
    return await call_next(request)


# -------------------------
# STARTUP / SHUTDOWN EVENTS
# -------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("Starting AHVI services")

    try:
        qdrant_service.init()
    except Exception:
        logger.exception("Qdrant startup failed")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("AHVI shutting down")
    try:
        qdrant_service.close()
    except Exception:
        logger.exception("Qdrant shutdown failed")

    try:
        from services.appwrite_service import client as appwrite_client

        close_fn = getattr(appwrite_client, "close", None)
        if callable(close_fn):
            close_fn()
    except Exception:
        logger.exception("Appwrite shutdown failed")

    try:
        if celery_app:
            if os.getenv("CELERY_BROADCAST_SHUTDOWN_ON_API_EXIT", "true").lower() in ("1", "true", "yes"):
                celery_app.control.broadcast("shutdown")
            close_fn = getattr(celery_app, "close", None)
            if callable(close_fn):
                close_fn()
    except Exception:
        logger.exception("Celery shutdown failed")

    try:
        import redis

        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        redis_client = redis.Redis.from_url(redis_url)
        redis_client.close()
    except Exception:
        logger.exception("Redis shutdown failed")

    try:
        from brain.orchestrator import AhviOrchestrator

        AhviOrchestrator._cache.clear()
    except Exception:
        logger.exception("Orchestrator cache cleanup failed")

    try:
        from routers import bg_remover, wardrobe_capture

        if hasattr(bg_remover, "clear_model_cache"):
            bg_remover.clear_model_cache()
        if hasattr(wardrobe_capture, "clear_model_cache"):
            wardrobe_capture.clear_model_cache()
    except Exception:
        logger.exception("Model cache cleanup failed")


# -------------------------
# ERROR HANDLERS
# -------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Invalid request",
                "details": exc.errors(),
            },
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled API error")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "Internal server error",
            },
        },
    )


# -------------------------
# MIDDLEWARE
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------
# ROUTER REGISTRATION
# -------------------------
if chat_router:
    app.include_router(chat_router, prefix="/api", tags=["Chat"])

if data_router:
    app.include_router(data_router)

if utilities_router:
    app.include_router(utilities_router)

if boards_router:
    app.include_router(boards_router)

if ai_router:
    app.include_router(ai_router, prefix="/api", tags=["AI"])

if feedback_router:
    app.include_router(feedback_router, tags=["Feedback"])

if stylist_router:
    app.include_router(stylist_router, prefix="/api/stylist")

if reddit_router:
    app.include_router(reddit_router)

if vision_router:
    app.include_router(vision_router, prefix="/api/vision")
if wardrobe_capture_router:
    app.include_router(wardrobe_capture_router)

if bg_router:
    app.include_router(bg_router, prefix="/api/background")

if garment_router:
    app.include_router(garment_router, prefix="/api")


# -------------------------
# HEALTH
# -------------------------
@app.get("/")
def root():
    return {"message": "AHVI backend running"}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "environment": settings.ENVIRONMENT,
        "version": app.version,
    }


# -------------------------
# CELERY STATUS
# -------------------------
@app.get("/api/tasks/{job_id}")
def get_task_status(job_id: str):
    if not celery_app or AsyncResult is None:
        return {"status": "celery not configured"}

    task_result = AsyncResult(job_id, app=celery_app)

    if task_result.state == "PENDING":
        return {"status": "processing"}

    if task_result.state == "SUCCESS":
        return {"status": "completed", "result": task_result.result}

    if task_result.state == "FAILURE":
        return {"status": "failed", "error": str(task_result.info)}

    return {"status": task_result.state}
