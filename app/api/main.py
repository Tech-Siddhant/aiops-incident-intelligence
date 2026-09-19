"""FastAPI application entrypoint.

ponytail: Single lightweight app file. No heavy dependency injection
unless shared state (like ML models) grows beyond simple lazy loading.
"""
import logging
import time
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "message": "%(message)s"}',
)
logger = logging.getLogger("omniroute.api")

app = FastAPI(
    title="Omniroute AIOps Incident Intelligence API",
    description="Lightweight backend for telemetry ML operations",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = (time.perf_counter() - start_time) * 1000
    logger.info(
        f"path={request.url.path} method={request.method} status={response.status_code} latency_ms={process_time:.2f}"
    )
    return response

from fastapi.responses import RedirectResponse

from app.config import FRONTEND_STATIC_DIR

app.include_router(router)

# Mount static files for the dashboard
if FRONTEND_STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_STATIC_DIR)), name="static")

@app.get("/", include_in_schema=False)
def redirect_to_dashboard():
    return RedirectResponse(url="/static/index.html")

@app.get("/health", include_in_schema=True)
def root_health_check():
    return {"status": "ok", "service": "omniroute-aiops"}

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"path={request.url.path} error={str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "message": str(exc)},
    )
