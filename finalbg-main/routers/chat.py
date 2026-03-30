from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import re
import os

from deep_translator import GoogleTranslator

try:
    from worker import run_heavy_audio_task
except Exception:
    run_heavy_audio_task = None

from brain.orchestrator import ahvi_orchestrator
from brain.outfit_pipeline import save_feedback
from services.appwrite_proxy import AppwriteProxy
from middleware.auth_middleware import ensure_user_scope, get_current_user
from services.prompt_safety import detect_prompt_injection

# ?? NEW
from services.weather_service import get_hourly_weather

router = APIRouter()


def _build_history(messages: List["Message"]) -> List[Dict[str, Any]]:
    history: List[Dict[str, Any]] = []
    for msg in messages[-8:]:
        role = str(getattr(msg, "role", "user")).lower()
        content = str(getattr(msg, "content", "")).strip()
        if not content:
            continue
        history.append({"role": role, "text": content})
    return history


def _is_fast_wardrobe_count_query(text: str) -> bool:
    lowered = str(text or "").lower()
    count_words = ["how many", "count", "number of", "total", "do i have"]
    wardrobe_words = [
        "wardrobe", "closet", "outfit", "outfits", "tops", "top", "shirts", "shirt",
        "pants", "trousers", "jeans", "bottoms", "shoes", "footwear", "dress",
        "dresses", "accessories", "jewelry", "bags", "bag",
    ]
    return any(k in lowered for k in count_words) and any(k in lowered for k in wardrobe_words)


def _fast_wardrobe_count_response(user_id: str, query_text: str) -> Dict[str, Any]:
    try:
        docs = AppwriteProxy().list_documents("outfits", user_id=user_id, limit=int(os.getenv("DEFAULT_WARDROBE_FETCH_LIMIT", "1000")))
    except Exception:
        docs = []

    counts = {"tops": 0, "bottoms": 0, "shoes": 0, "dresses": 0, "accessories": 0}
    for d in docs:
        category = str(d.get("category") or d.get("category_group") or "").lower()
        sub = str(d.get("sub_category") or d.get("subcategory") or "").lower()
        blob = f"{category} {sub}"
        if any(k in blob for k in ["top", "shirt", "blouse", "jacket", "blazer", "tee"]):
            counts["tops"] += 1
        elif any(k in blob for k in ["bottom", "pant", "trouser", "jean", "short", "skirt"]):
            counts["bottoms"] += 1
        elif any(k in blob for k in ["shoe", "sneaker", "heel", "boot", "sandal", "footwear"]):
            counts["shoes"] += 1
        elif "dress" in blob:
            counts["dresses"] += 1
        elif any(k in blob for k in ["accessory", "watch", "bag", "jewel", "necklace", "earring"]):
            counts["accessories"] += 1

    lowered = str(query_text or "").lower()
    if any(k in lowered for k in ["top", "tops", "shirt", "shirts", "blouse", "blouses"]):
        message = f"You have {counts['tops']} tops in your wardrobe."
    elif any(k in lowered for k in ["bottom", "bottoms", "pant", "pants", "trouser", "trousers", "jean", "jeans"]):
        message = f"You have {counts['bottoms']} bottoms in your wardrobe."
    elif any(k in lowered for k in ["shoe", "shoes", "footwear", "sneaker", "sneakers"]):
        message = f"You have {counts['shoes']} shoes in your wardrobe."
    else:
        total = len(docs)
        message = (
            f"You currently have {total} items: {counts['tops']} tops, {counts['bottoms']} bottoms, "
            f"{counts['shoes']} shoes, {counts['dresses']} dresses, and {counts['accessories']} accessories."
        )

    return {
        "success": True,
        "message": message,
        "board": "wardrobe",
        "type": "stats",
        "cards": [
            {"id": "tops", "title": "Tops", "kind": "stat", "value": counts["tops"]},
            {"id": "bottoms", "title": "Bottoms", "kind": "stat", "value": counts["bottoms"]},
            {"id": "shoes", "title": "Shoes", "kind": "stat", "value": counts["shoes"]},
            {"id": "dresses", "title": "Dresses", "kind": "stat", "value": counts["dresses"]},
            {"id": "accessories", "title": "Accessories", "kind": "stat", "value": counts["accessories"]},
        ],
        "data": {"counts": counts, "total_items": len(docs)},
        "meta": {"intent": "wardrobe_query", "domain": "wardrobe", "fast_path": True},
        "audio_job_id": "offline",
    }


# -------------------------
# MODELS
# -------------------------
class Message(BaseModel):
    role: str
    content: str


class TextChatRequest(BaseModel):
    messages: List[Message]
    language: str = "en"
    current_memory: Any = {}
    user_profile: Dict[str, Any] = {}
    user_id: str | None = None
    userID: str | None = None
    module_context: str | None = None


class OutfitFeedbackRequest(BaseModel):
    user_id: str
    feedback: str
    outfit: Dict[str, Any]


class OrganizeHubRequest(BaseModel):
    user_id: str
    user_profile: Dict[str, Any] = {}
    current_memory: Any = {}
    include_counts: bool = False


class PlanPackRequest(BaseModel):
    user_id: str
    prompt: str
    user_profile: Dict[str, Any] = {}
    current_memory: Any = {}


class DailyCardsRequest(BaseModel):
    user_id: str
    time_slot: str | None = None
    user_profile: Dict[str, Any] = {}
    current_memory: Any = {}


# -------------------------
# CHAT ENDPOINT
# -------------------------
@router.post("/text")
async def text_chat(request: TextChatRequest, user=Depends(get_current_user)):

    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    user_input = request.messages[-1].content.strip()

    if not user_input:
        raise HTTPException(status_code=400, detail="Empty message")
    blocked, signature = detect_prompt_injection(user_input)
    if blocked:
        raise HTTPException(
            status_code=400,
            detail=f"Potential prompt-injection content detected ({signature}). Please rephrase your request.",
        )

    resolved_user_id = request.user_id or request.userID or user.get("user_id") or "user_1"
    if isinstance(user, dict) and user.get("user_id"):
        ensure_user_scope(user, resolved_user_id)

    # Fast deterministic response for wardrobe count questions to avoid model latency.
    if _is_fast_wardrobe_count_query(user_input):
        return _fast_wardrobe_count_response(
            user_id=resolved_user_id,
            query_text=user_input,
        )

    # -------------------------
    # LANGUAGE DETECTION
    # -------------------------
    try:
        preferred_lang = str(request.language or "en").lower()
        has_telugu = bool(re.search(r"[\u0C00-\u0C7F]", user_input))
        has_hindi = bool(re.search(r"[\u0900-\u097F]", user_input))

        if preferred_lang == "te" or has_telugu:
            english_input = GoogleTranslator(source="te", target="en").translate(user_input)
            target_lang = "te"
        elif preferred_lang == "hi" or has_hindi:
            english_input = GoogleTranslator(source="hi", target="en").translate(user_input)
            target_lang = "hi"
        else:
            english_input = user_input
            target_lang = "en"

    except Exception:
        english_input = user_input
        target_lang = "en"

    # -------------------------
    # ?? WEATHER INJECTION (NEW)
    # -------------------------
    weather_data = {}

    try:
        location = request.user_profile.get("location", {})

        if location.get("lat") and location.get("lon"):
            weather_data = get_hourly_weather(
                lat=location.get("lat"),
                lon=location.get("lon")
            )
    except Exception as e:
        print("Weather error:", str(e))

    # -------------------------
    # ORCHESTRATOR CALL
    # -------------------------
    history = _build_history(request.messages[:-1]) if len(request.messages) > 1 else []
    memory_history = request.current_memory.get("history", []) if isinstance(request.current_memory, dict) else []
    merged_history = [h for h in memory_history if isinstance(h, dict)] + history

    slot_hints: Dict[str, Any] = {}
    if request.module_context:
        module = str(request.module_context).lower()
        if "occasion" in module:
            slot_hints["occasion"] = request.user_profile.get("occasion")
        if "work" in module or "office" in module:
            slot_hints["occasion"] = slot_hints.get("occasion") or "office"

    try:
        result = ahvi_orchestrator.run(
            text=english_input,
            user_id=resolved_user_id,
            context={
                "memory": request.current_memory,
                "user_profile": request.user_profile,
                "module_context": request.module_context,
                "history": merged_history[-20:],
                "slots": slot_hints,

                # ?? NEW CONTEXT
                "weather": weather_data.get("condition"),
                "time_of_day": weather_data.get("time_of_day"),
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Orchestrator failed: {exc}")

    # -------------------------
    # TRANSLATE RESPONSE BACK
    # -------------------------
    message = result.get("message", "")

    try:
        if target_lang != "en" and message:
            message = GoogleTranslator(source="en", target=target_lang).translate(message)
    except Exception:
        pass

    # -------------------------
    # AUDIO (OPTIONAL)
    # -------------------------
    try:
        if (
            run_heavy_audio_task is not None
            and os.getenv("ENABLE_AUDIO_TASKS", "false").lower() in ("1", "true", "yes")
        ):
            task = run_heavy_audio_task.delay(message, target_lang)
            audio_job_id = task.id
        else:
            audio_job_id = "offline"
    except Exception:
        audio_job_id = "offline"

    # -------------------------
    # FINAL RESPONSE
    # -------------------------
    return {
        "success": True,
        "message": message,
        "board": result.get("board"),
        "type": result.get("type"),
        "cards": result.get("cards", []),
        "data": result.get("data", {}),
        "meta": {
            **result.get("meta", {}),
            "weather": weather_data,
            "history_used": len(merged_history[-20:])
        },
        "audio_job_id": audio_job_id,
    }


# -------------------------
# FEEDBACK
# -------------------------
@router.post("/feedback/outfit")
def outfit_feedback(request: OutfitFeedbackRequest, user=Depends(get_current_user)):
    ensure_user_scope(user, request.user_id)

    fb = str(request.feedback).strip().lower()

    mapped = "up" if fb in ("up", "like", "liked", "thumbs_up") else "down"

    if fb not in (
        "up", "down", "like", "liked", "dislike", "disliked",
        "thumbs_up", "thumbs_down"
    ):
        raise HTTPException(status_code=400, detail="feedback must be up/down")

    try:
        return save_feedback(
            user_id=request.user_id,
            outfit=request.outfit,
            feedback=mapped
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {exc}")


@router.post("/organize/chips")
def organize_chips(request: OrganizeHubRequest, user=Depends(get_current_user)):
    ensure_user_scope(user, request.user_id)
    try:
        result = ahvi_orchestrator.run(
            text="open organize",
            user_id=request.user_id,
            context={
                "memory": request.current_memory,
                "user_profile": request.user_profile,
                "module_context": "organize",
                "history": [],
                "include_counts": request.include_counts,
            },
        )
        return {
            "success": True,
            "message": result.get("message", "Choose what you want to organize."),
            "board": result.get("board", "organize"),
            "type": result.get("type", "chips"),
            "chips": result.get("cards", []),
            "data": result.get("data", {}),
            "meta": result.get("meta", {}),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load organize chips: {exc}")


@router.post("/plan-pack")
def plan_pack(request: PlanPackRequest, user=Depends(get_current_user)):
    ensure_user_scope(user, request.user_id)
    blocked, signature = detect_prompt_injection(request.prompt)
    if blocked:
        raise HTTPException(
            status_code=400,
            detail=f"Potential prompt-injection content detected ({signature}). Please rephrase your request.",
        )
    try:
        weather_data = {}
        try:
            location = request.user_profile.get("location", {})
            if location.get("lat") and location.get("lon"):
                weather_data = get_hourly_weather(
                    lat=location.get("lat"),
                    lon=location.get("lon")
                )
        except Exception:
            weather_data = {}

        result = ahvi_orchestrator.run(
            text=request.prompt,
            user_id=request.user_id,
            context={
                "memory": request.current_memory,
                "user_profile": request.user_profile,
                "module_context": "plan_pack",
                "history": [],
                "weather": weather_data.get("condition"),
                "time_of_day": weather_data.get("time_of_day"),
                "weather_data": weather_data,
            },
        )
        return {
            "success": True,
            "message": result.get("message", ""),
            "board": result.get("board", "plan_pack"),
            "type": result.get("type", "checklists"),
            "cards": result.get("cards", []),
            "data": result.get("data", {}),
            "meta": {
                **result.get("meta", {}),
                "weather": weather_data,
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to build plan & pack flow: {exc}")


@router.post("/daily/cards")
@router.post("/text/daily/cards")
def daily_cards(request: DailyCardsRequest, user=Depends(get_current_user)):
    ensure_user_scope(user, request.user_id)
    try:
        weather_data = {}
        try:
            location = request.user_profile.get("location", {})
            if location.get("lat") and location.get("lon"):
                weather_data = get_hourly_weather(
                    lat=location.get("lat"),
                    lon=location.get("lon"),
                )
        except Exception:
            weather_data = {}

        slot_hint = (request.time_slot or "").strip().lower() if request.time_slot else ""
        prompt = f"{slot_hint} daily cards".strip() if slot_hint else "daily cards"

        result = ahvi_orchestrator.run(
            text=prompt,
            user_id=request.user_id,
            context={
                "memory": request.current_memory,
                "user_profile": request.user_profile,
                "module_context": "daily_dependency",
                "history": [],
                "time_slot": slot_hint or None,
                "weather": weather_data.get("condition"),
                "time_of_day": weather_data.get("time_of_day"),
                "weather_data": weather_data,
            },
        )

        return {
            "success": True,
            "message": result.get("message", ""),
            "board": result.get("board", "daily_dependency"),
            "type": result.get("type", "cards"),
            "cards": result.get("cards", [])[:3],
            "data": result.get("data", {}),
            "meta": {
                **result.get("meta", {}),
                "weather": weather_data,
            },
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to build daily cards: {exc}")



