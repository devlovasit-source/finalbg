from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from services.appwrite_service import build_request_account


_bearer = HTTPBearer(auto_error=False)


def get_current_user(credentials: HTTPAuthorizationCredentials | None = Depends(_bearer)):
    """
    Extracts user from Appwrite session JWT.
    """
    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    try:
        token = credentials.credentials.strip()
        if not token:
            raise HTTPException(status_code=401, detail="Malformed Authorization header")
        account = build_request_account(token)
        user = account.get()

        return {
            "user_id": user["$id"],
            "email": user.get("email"),
            "name": user.get("name"),
        }
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


def ensure_user_scope(auth_user: dict, requested_user_id: str) -> None:
    auth_user_id = str((auth_user or {}).get("user_id", "")).strip()
    requested = str(requested_user_id or "").strip()
    if not requested:
        return
    if not auth_user_id:
        raise HTTPException(status_code=401, detail="Unauthorized user context")
    if requested != auth_user_id:
        raise HTTPException(status_code=403, detail="User scope mismatch")
