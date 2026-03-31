import pytest
from fastapi import HTTPException

from middleware.auth_middleware import ensure_user_scope


def test_context_isolation_allows_same_user():
    ensure_user_scope({"user_id": "user_1"}, "user_1")


def test_context_isolation_blocks_cross_user_access():
    with pytest.raises(HTTPException) as exc:
        ensure_user_scope({"user_id": "user_1"}, "user_2")
    assert exc.value.status_code == 403
