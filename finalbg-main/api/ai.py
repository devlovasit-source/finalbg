from fastapi import APIRouter
from brain.orchestrator import ahvi_orchestrator

router = APIRouter(prefix="/ai")

@router.post("/run")
def run_ai(payload: dict):
    return ahvi_orchestrator.run(
        text=payload.get("message"),
        user_id=payload.get("userId"),
        context=payload.get("context", {})
    )