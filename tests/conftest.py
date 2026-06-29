"""Shared test fixtures.

The dev `.env` sets `API_KEY`, which makes `verify_api_key` enforce auth on the
API routes. The endpoint tests exercise routing/validation, not auth, so we
override the dependency to a no-op for the test session. Production behaviour is
unchanged.
"""

import pytest

from src.api.main import app
from src.api.middleware.auth import verify_api_key


@pytest.fixture(autouse=True)
def _disable_api_auth():
    app.dependency_overrides[verify_api_key] = lambda: None
    yield
    app.dependency_overrides.pop(verify_api_key, None)
