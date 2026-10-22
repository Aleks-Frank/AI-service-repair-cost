import asyncio
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.api import health, requests, webhooks
from app.core.config import Settings
from app.core.errors import ServiceError
from app.core.middleware import BodyLimitMiddleware
from app.integrations.agent import Agent, EstimatorAgent
from app.integrations.bitrix import BitrixClient
from app.processing import ProcessingWorker
from app.repository import RequestRepository


def create_app(
    settings: Settings | None = None,
    *,
    bitrix_transport: httpx.AsyncBaseTransport | None = None,
    agent: Agent | None = None,
) -> FastAPI:
    settings = settings or Settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.CRITICAL)
    logging.getLogger("httpcore").setLevel(logging.CRITICAL)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        repository = RequestRepository(settings.database_path)
        await run_in_threadpool(repository.initialize)
        application.state.repository = repository
        async with httpx.AsyncClient(
            timeout=settings.bitrix_timeout_seconds,
            follow_redirects=False,
            trust_env=False,
            transport=bitrix_transport,
        ) as client:
            application.state.bitrix = BitrixClient(settings, client)
            worker = task = None
            if settings.ai_enabled:
                active_agent = agent if agent is not None else EstimatorAgent(settings)
                worker = ProcessingWorker(repository, settings, active_agent)
                task = asyncio.create_task(worker.run(), name="request-processor")
            try:
                yield
            finally:
                if worker is not None:
                    worker.stopping.set()
                    try:
                        await task
                    finally:
                        await run_in_threadpool(worker.agent.close)

    application = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        lifespan=lifespan,
    )
    application.state.settings = settings
    application.add_middleware(BodyLimitMiddleware, max_bytes=settings.max_request_bytes)

    @application.exception_handler(ServiceError)
    async def service_error_handler(request: Request, error: ServiceError) -> JSONResponse:
        return JSONResponse(
            status_code=error.status_code,
            content={"detail": {"code": error.code, "message": error.message}},
        )

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        issues = [{"loc": item["loc"], "type": item["type"]} for item in error.errors()]
        return JSONResponse(
            status_code=422,
            content={"detail": {"code": "validation_error", "issues": issues}},
        )

    application.include_router(health.router)
    application.include_router(webhooks.router, prefix="/api/v1")
    application.include_router(requests.router, prefix="/api/v1")
    return application
