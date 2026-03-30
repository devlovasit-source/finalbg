import json
import re
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import Any, Dict, List
from services import llm_service
from brain.outfit_pipeline import get_daily_outfits
from brain.personalization.style_dna_engine import style_dna_engine
from services.appwrite_proxy import AppwriteProxy
from services.rate_limiter import limiter
from middleware.auth_middleware import ensure_user_scope, get_current_user
from services.prompt_safety import detect_prompt_injection

router = APIRouter()

# Notice we no longer accept base64 images here. We accept the CLIP output.
class ItemContextRequest(BaseModel):
    main_category: str
    sub_category: str
    color_hex: str


class OutfitPipelineRequest(BaseModel):
    user_id: str
    query: str = "What should I wear today?"
    wardrobe: Any = None
    user_profile: Dict[str, Any] = {}
    context: Dict[str, Any] = {}


@router.post("/item-suggestions")
@limiter.limit("20/minute")
def get_item_suggestions(request: Request, payload: ItemContextRequest, user=Depends(get_current_user)):
    blocked, signature = detect_prompt_injection(
        f"{payload.main_category} {payload.sub_category} {payload.color_hex}"
    )
    if blocked:
        raise HTTPException(
            status_code=400,
            detail=f"Potential prompt-injection content detected ({signature}). Please rephrase your request.",
        )
    system_instruction = (
        "You are Ahvi's Fashion Knowledge Engine. The user just uploaded a new garment. "
        "Based on the provided attributes, return a JSON object with: "
        "1. 'name' (A catchy, descriptive name for the item) "
        "2. 'tags' (array of 4 style keywords like 'streetwear', 'vintage') "
        "3. 'pairing_rules' (array of 2 short rules on what to wear this with). "
        "Output ONLY raw JSON."
    )
    
    user_prompt = (
        f"Item: {payload.sub_category}\n"
        f"Category: {payload.main_category}\n"
        f"Color Hex: {payload.color_hex}"
    )
    
    try:
        messages = [{"role": "user", "content": user_prompt}]
        # Using the much faster, cheaper text model
        response_text = llm_service.chat_completion(
            messages,
            system_instruction,
            model="llama3.1",
        )
        
        # Safe JSON extraction
        clean_response = re.sub(r'```json|```', '', response_text).strip()
        return json.loads(clean_response)
        
    except Exception as e:
        print(f"Stylist Text Engine Error: {str(e)}")
        # Safe fallback
        return {
            "name": f"{payload.sub_category.title()}",
            "tags": ["versatile", "casual"],
            "pairing_rules": ["Pair with neutral basics.", "Layer depending on weather."]
        }


@router.post("/pipeline")
@limiter.limit("10/minute")
def run_outfit_pipeline(request: Request, payload: OutfitPipelineRequest, user=Depends(get_current_user)):
    ensure_user_scope(user, payload.user_id)
    blocked, signature = detect_prompt_injection(payload.query)
    if blocked:
        raise HTTPException(
            status_code=400,
            detail=f"Potential prompt-injection content detected ({signature}). Please rephrase your request.",
        )
    appwrite = AppwriteProxy()
    context = dict(payload.context or {})
    context["query"] = payload.query

    wardrobe = payload.wardrobe
    if wardrobe is None:
        try:
            wardrobe = appwrite.list_documents("outfits", user_id=payload.user_id)
        except Exception:
            wardrobe = []

    style_dna = style_dna_engine.build(
        {
            "user_id": payload.user_id,
            "user_profile": payload.user_profile or {},
            "history": context.get("history", []),
            "wardrobe": wardrobe,
        }
    )
    context["style_dna"] = style_dna

    try:
        result = get_daily_outfits(
            {
                "user_id": payload.user_id,
                "wardrobe": wardrobe,
                "context": context,
            }
        )
        return {
            "success": True,
            "message": "Outfit pipeline generated successfully.",
            "data": result,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to run outfit pipeline: {exc}")
