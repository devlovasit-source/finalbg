from fastapi import Request, HTTPException

from services.appwrite_service import account


async def get_current_user(request: Request):
    """
    Extracts user from Appwrite session JWT.
    """
    auth_header = request.headers.get("authorization", "").strip()
    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if not auth_header.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Authorization header must be Bearer token")

    try:
        parts = auth_header.split(" ", 1)
        if len(parts) != 2 or not parts[1].strip():
            raise HTTPException(status_code=401, detail="Malformed Authorization header")

        token = parts[1].strip()
        account.client.set_jwt(token)
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
