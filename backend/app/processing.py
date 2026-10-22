import asyncio
import logging

from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from app.core.config import Settings
from app.core.errors import ServiceError
from app.integrations.agent import Agent, agent_input
from app.models import RequestRecord
from app.repository import RequestRepository

logger = logging.getLogger(__name__)


class ProcessingWorker:
    def __init__(self, repository: RequestRepository, settings: Settings, agent: Agent) -> None:
        self.repository = repository
        self.settings = settings
        self.agent = agent
        self.stopping = asyncio.Event()

    async def run(self) -> None:
        while not self.stopping.is_set():
            try:
                record = await run_in_threadpool(self.repository.claim_processing)
                if record is not None:
                    await self.process(record)
                    continue
            except Exception as error:
                logger.error("Processing worker error type=%s", type(error).__name__)
            try:
                await asyncio.wait_for(self.stopping.wait(), timeout=self.settings.ai_poll_seconds)
            except TimeoutError:
                pass

    async def process(self, record: RequestRecord) -> None:
        request_id = str(record.id)
        logger.info("Processing started request_id=%s", request_id)
        try:
            data = await run_in_threadpool(
                agent_input, record.payload, record.source, self.settings
            )
            result = await run_in_threadpool(self.agent, data)
            await run_in_threadpool(self.repository.save_result, request_id, result)
        except Exception as error:
            if isinstance(error, ServiceError):
                code = error.code
            elif isinstance(error, ValidationError):
                code = "agent_invalid_result"
            else:
                code = "processing_internal_error"
            await run_in_threadpool(self.repository.fail_processing, request_id, code)
            logger.warning(
                "Processing failed request_id=%s code=%s type=%s",
                request_id,
                code,
                type(error).__name__,
            )
            return
        logger.info("Processing completed request_id=%s", request_id)
