from fastapi import Request

from app.repository import RequestRepository


def get_repository(request: Request) -> RequestRepository:
    return request.app.state.repository
