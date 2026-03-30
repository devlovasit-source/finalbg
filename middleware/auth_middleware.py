from fastapi import Request, HTTPException
from services.appwrite_service import account


async def get_current_user(request: Request):
    """
    Extracts user from Appwrite session (JWT)
    """

    auth_header = request.headers.get("authorization")

    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    try:
        # Expecting: Bearer <token>
        token = auth_header.split(" ")[1]

        # 🔥 Attach token to client
        account.client.set_jwt(token)

        user = account.get()

        return {
            "user_id": user["$id"],
            "email": user.get("email"),
            "name": user.get("name")
        }

    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid or expired token")