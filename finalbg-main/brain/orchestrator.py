# backend/brain/orchestrator.py

import traceback
import uuid
import hashlib
import json
import os
from time import perf_counter
from typing import Any, Dict

from brain.agent_system import agent_system
from brain.context.context_engine import context_engine
from brain.execution_engine import execution_engine
from brain.outfit_pipeline import get_daily_outfits
from brain.plan_pack_flow import build_plan_pack_response
from brain.personalization.style_dna_engine import style_dna_engine
from brain.intent_engine import detect_intent
from brain.daily_dependency_engine import build_daily_dependency_response

from services.appwrite_proxy import AppwriteProxy
from services.llm_service import generate_text  # 🔥 NEW


def _hash_outfit(outfit):
    normalized = json.dumps(outfit, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def _safe_log(text: str) -> None:
    try:
        print(text)
    except Exception:
        try:
            safe_text = str(text).encode("ascii", errors="ignore").decode("ascii")
            print(safe_text)
        except Exception:
            pass


class AhviOrchestrator:
    _cache: Dict[str, Dict[str, Any]] = {}
    _cache_ttl_seconds: float = 30.0
    _default_wardrobe_fetch_limit: int = int(os.getenv("DEFAULT_WARDROBE_FETCH_LIMIT", "1000"))

    def run(self, text: str, user_id: str = None, context: Dict[str, Any] = None) -> Dict[str, Any]:
        context = context or {}
        user_id = user_id or "anonymous"
        request_id = str(uuid.uuid4())
        started_at = perf_counter()

        appwrite = AppwriteProxy()

        try:
            # Fast-path organize entry for chat-open UX.
            early_organize = self._resolve_organize_request(text=text, context=context, slots={}, intent="general")
            if early_organize.get("active") and ("organize" in str(text or "").lower() or "organize" in str(context.get("module_context") or "").lower()):
                return self._organize_response(
                    request_id=request_id,
                    user_id=user_id,
                    appwrite=appwrite,
                    module_key=early_organize.get("module"),
                    include_counts=bool(context.get("include_counts", False)),
                )

            if self._is_plan_pack_request(text=text, context=context):
                pp = build_plan_pack_response(text=text, context=context)
                return {
                    "success": True,
                    "request_id": request_id,
                    "meta": {"intent": "plan_pack", "domain": "planning"},
                    "message": pp.get("message"),
                    "board": pp.get("board", "plan_pack"),
                    "type": pp.get("type", "checklists"),
                    "cards": pp.get("cards", []),
                    "data": pp.get("data", {}),
                }

            if self._is_daily_dependency_request(text=text, context=context):
                return {
                    "success": True,
                    "request_id": request_id,
                    **build_daily_dependency_response(
                        user_id=user_id,
                        context=context,
                        appwrite=appwrite,
                    ),
                }

            # -------------------------
            # INTENT ENGINE
            # -------------------------
            intent_data = detect_intent(text, context.get("history"))
            intent = intent_data.get("intent", "general")

            if intent == "daily_dependency":
                return {
                    "success": True,
                    "request_id": request_id,
                    **build_daily_dependency_response(
                        user_id=user_id,
                        context={**context, "time_slot": (intent_data.get("slots", {}) or {}).get("time")},
                        appwrite=appwrite,
                    ),
                }

            # -------------------------
            # SLOT MERGE
            # -------------------------
            fallback_slots = self._extract_slots(text=text, context=context)
            llm_slots = intent_data.get("slots", {}) or {}
            slots = {**fallback_slots, **llm_slots}
            intent_data["slots"] = slots

            # -------------------------
            # 🔥 SIGNALS (NEW)
            # -------------------------
            signals = {
                "context_mode": "styling",
                "emotion_state": self._infer_emotion_state(text),
            }

            # -------------------------
            # TRY-ON FLOW
            # -------------------------
            if intent == "try_on":
                return {
                    "success": True,
                    "request_id": request_id,
                    "meta": {"intent": intent, "domain": "styling"},
                    "message": "Select an outfit to try on.",
                    "board": "tryon",
                    "type": "tryon",
                    "cards": [],
                    "data": {
                        "action": "open_tryon",
                        "note": "Frontend should pass outfit to try-on API"
                    },
                }

            # -------------------------
            # WARDROBE QUERY FLOW
            # -------------------------
            if intent == "wardrobe_query" or self._is_wardrobe_count_query(text):
                return self._wardrobe_query_response(
                    request_id=request_id,
                    user_id=user_id,
                    appwrite=appwrite,
                    text=text,
                )

            # -------------------------
            # PLAN & PACK FLOW
            # -------------------------
            if intent == "plan_pack" or self._is_plan_pack_request(text=text, context=context):
                pp = build_plan_pack_response(text=text, context=context)
                return {
                    "success": True,
                    "request_id": request_id,
                    "meta": {"intent": "plan_pack", "domain": "planning"},
                    "message": pp.get("message"),
                    "board": pp.get("board", "plan_pack"),
                    "type": pp.get("type", "checklists"),
                    "cards": pp.get("cards", []),
                    "data": pp.get("data", {}),
                }

            # -------------------------
            # ORGANIZE HUB FLOW
            # -------------------------
            organize_signal = self._resolve_organize_request(text=text, context=context, slots=slots, intent=intent)
            if organize_signal.get("active"):
                return self._organize_response(
                    request_id=request_id,
                    user_id=user_id,
                    appwrite=appwrite,
                    module_key=organize_signal.get("module"),
                    include_counts=bool(context.get("include_counts", False)),
                )

            # -------------------------
            # STYLING FLOWS
            # -------------------------
            if intent in ("daily_outfit", "occasion_outfit", "explore_styles"):
                # -------------------------
                # FETCH WARDROBE
                # -------------------------
                try:
                    wardrobe_docs = appwrite.list_documents("outfits", user_id=user_id)
                    context["wardrobe"] = wardrobe_docs
                except Exception as e:
                    print("ERROR fetching wardrobe:", str(e))
                    context["wardrobe"] = []

                # -------------------------
                # CONTEXT ENGINE
                # -------------------------
                enriched_context = context_engine.build_context(
                    user_id=user_id,
                    intent_data=intent_data,
                    wardrobe=context.get("wardrobe", []),
                    user_profile=context.get("user_profile"),
                    history=context.get("history", []),
                    vision=context.get("vision"),
                )

                cache_key = self._cache_key(
                    text=text,
                    user_id=user_id,
                    context={**context, **enriched_context},
                )
                cached = self._get_cache(cache_key)
                if cached is not None:
                    cached_data = dict(cached)
                    cached_data["meta"] = {**cached_data.get("meta", {}), "cache_hit": True}
                    return cached_data

                # -------------------------
                # STYLE DNA
                # -------------------------
                style_dna = self._build_style_dna({**context, **enriched_context, "user_id": user_id})

                # -------------------------
                # PLAN
                # -------------------------
                plan = agent_system.plan(intent=intent, context=enriched_context)

                state: Dict[str, Any] = {"pipeline_result": {}}

                # -------------------------
                # EXECUTION STEPS
                # -------------------------
                def _normalize_context():
                    return {
                        "slots": enriched_context.get("slots", {}),
                        "meta": enriched_context.get("meta", {}),
                    }

                def _build_style_graph():
                    return {"status": "prepared"}

                def _generate_score_rank():
                    slots_ctx = enriched_context.get("slots", {}) or {}

                    state["pipeline_result"] = get_daily_outfits(
                        {
                            "user_id": user_id,
                            "wardrobe": enriched_context.get("wardrobe", []),
                            "context": {
                                "query": text,
                                "weather": slots_ctx.get("weather"),
                                "occasion": slots_ctx.get("occasion"),
                                "location": slots_ctx.get("location"),
                                "time_of_day": slots_ctx.get("time"),
                                "history": enriched_context.get("history", []),
                                "recent_outfits": context.get("memory", {}).get("recent_outfits", []),
                                "style_dna": style_dna,
                            },
                        }
                    )

                    return {"outfits_count": len(state["pipeline_result"].get("outfits", []))}

                def _persist_hooks():
                    return {"feedback_endpoint": "/api/feedback/outfit"}

                exec_result = execution_engine.execute(
                    plan=plan,
                    handlers={
                        "normalize_context": _normalize_context,
                        "build_style_graph": _build_style_graph,
                        "generate_score_rank": _generate_score_rank,
                        "persist_and_feedback_hooks": _persist_hooks,
                        "no_op": lambda: {"status": "noop"},
                    },
                    timeout_seconds=3.0,
                    slow_step_threshold_seconds=1.5,
                )

                pipeline_result = state.get("pipeline_result", {})
                outfits = pipeline_result.get("outfits", [])
                boards = pipeline_result.get("boards", [])
                cards_from_pipeline = pipeline_result.get("cards", [])

                # -------------------------
                # SAVE OUTFITS (UNCHANGED)
                # -------------------------
                saved_outfit_ids = []

                try:
                    existing_outfits = appwrite.list_documents("outfits", user_id=user_id)
                    existing_hashes = {_hash_outfit(o) for o in existing_outfits}
                except Exception as e:
                    print("ERROR loading existing outfits:", str(e))
                    existing_hashes = set()

                for outfit in outfits:
                    try:
                        outfit_hash = _hash_outfit(outfit)
                        if outfit_hash in existing_hashes:
                            continue

                        doc = appwrite.create_document(
                            "outfits",
                            {"userId": user_id, "hash": outfit_hash, **outfit}
                        )

                        saved_outfit_ids.append(doc.get("$id"))

                    except Exception as e:
                        print("ERROR saving outfit:", str(e))

                # -------------------------
                # CARDS (UNCHANGED LOGIC)
                # -------------------------
                cards = cards_from_pipeline if cards_from_pipeline else boards if boards else [
                    {
                        "title": f"Outfit {i+1}",
                        "outfit_id": oid,
                        "preview": outfits[i] if i < len(outfits) else {}
                    }
                    for i, oid in enumerate(saved_outfit_ids[:3])
                ]

                # -------------------------
                # 🔥 AHVI MESSAGE (UPGRADE ONLY)
                # -------------------------
                if outfits:
                    message = self._build_stylist_message(
                        text=text,
                        cards=cards,
                        context={**context, **enriched_context},
                        style_dna=style_dna,
                        user_profile=context.get("user_profile"),
                        signals=signals,
                    )
                else:
                    message = "I need more wardrobe items to generate outfits."

                # -------------------------
                # RESPONSE
                # -------------------------
                response = {
                    "success": True,
                    "request_id": request_id,
                    "meta": {
                        "intent": intent,
                        "domain": "styling",
                        "cache_hit": False,
                        "latency_ms": round((perf_counter() - started_at) * 1000.0, 2),
                    },
                    "message": message,
                    "board": "outfit",
                    "type": "boards" if boards else "cards",
                    "cards": cards,
                    "data": {
                        "outfits_count": len(outfits),
                        "execution": exec_result,
                        "boards_available": bool(boards),
                        "tryon_enabled": True,
                        "pipeline": pipeline_result.get("pipeline", {}),
                        "premium": pipeline_result.get("premium", {}),
                    },
                }

                self._set_cache(cache_key, response)
                return response

            # -------------------------
            # FALLBACK
            # -------------------------
            general_message = generate_text(
                text,
                user_profile=context.get("user_profile"),
                signals={"context_mode": "general"}
            )
            if not general_message or general_message == "none":
                general_message = "Tell me your occasion, weather, and vibe, and I will style you with your wardrobe."

            return {
                "success": True,
                "request_id": request_id,
                "meta": {"intent": intent, "domain": "general"},
                "message": general_message,
                "board": "general",
                "type": "text",
                "cards": [],
                "data": {"intent": intent},
            }

        except Exception:
            _safe_log("ERROR: Orchestrator error\n" + traceback.format_exc())
            return {
                "success": False,
                "request_id": request_id,
                "error": {
                    "code": "ORCHESTRATOR_ERROR",
                    "message": "Something went wrong",
                    "details": request_id,
                },
            }

    # -------------------------
    # HELPERS (UNCHANGED)
    # -------------------------
    def _extract_slots(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        lowered = (text or "").lower()
        slots = dict((context.get("slots") or {}))

        if not slots.get("occasion"):
            if "office" in lowered:
                slots["occasion"] = "office"
            elif "party" in lowered:
                slots["occasion"] = "party"
            elif "casual" in lowered:
                slots["occasion"] = "casual"

        if not slots.get("weather"):
            if "warm" in lowered or "hot" in lowered:
                slots["weather"] = "warm"
            elif "cold" in lowered:
                slots["weather"] = "cold"
            elif "rain" in lowered:
                slots["weather"] = "rainy"

        return slots

    def _build_style_dna(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if context.get("style_dna"):
            value = context.get("style_dna")
            return value if isinstance(value, dict) else {}
        if hasattr(style_dna_engine, "build"):
            return style_dna_engine.build(
                {
                    "user_id": context.get("user_id"),
                    "user_profile": context.get("user_profile"),
                    "history": context.get("history", []),
                    "signals": context.get("signals"),
                    "wardrobe": context.get("wardrobe", []),
                }
            )
        return {}

    def _cache_key(self, text: str, user_id: str, context: Dict[str, Any]) -> str:
        wardrobe = context.get("wardrobe", []) or []
        wardrobe_signature = "|".join(
            sorted(
                str(item.get("$id") or item.get("id") or item.get("item_id") or "")
                for item in wardrobe
                if isinstance(item, dict)
            )[:120]
        )
        slots = context.get("slots", {}) if isinstance(context.get("slots"), dict) else {}
        slot_signature = "|".join(f"{k}:{slots.get(k)}" for k in sorted(slots.keys()))
        profile = context.get("user_profile", {}) if isinstance(context.get("user_profile"), dict) else {}
        profile_signature = f"{profile.get('style', '')}|{','.join(profile.get('preferred_colors', profile.get('colors', []))[:5])}"
        seed = f"{user_id}|{text.strip().lower()}|{wardrobe_signature}|{slot_signature}|{profile_signature}"
        return hashlib.md5(seed.encode()).hexdigest()

    def _get_cache(self, key: str) -> Dict[str, Any] | None:
        data = self._cache.get(key)
        if not data:
            return None
        age = perf_counter() - float(data.get("_saved_at", 0.0))
        if age > self._cache_ttl_seconds:
            self._cache.pop(key, None)
            return None
        return data.get("payload")

    def _set_cache(self, key: str, payload: Dict[str, Any]) -> None:
        self._cache[key] = {"_saved_at": perf_counter(), "payload": payload}

    def _infer_emotion_state(self, text: str) -> str:
        t = str(text or "").lower()
        if any(k in t for k in ["urgent", "asap", "quick", "hurry"]):
            return "stressed"
        if any(k in t for k in ["anxious", "nervous", "confused"]):
            return "vulnerable"
        if any(k in t for k in ["excited", "date", "party", "celebrate"]):
            return "energized"
        return "neutral"

    def _build_stylist_message(
        self,
        text: str,
        cards: list,
        context: Dict[str, Any],
        style_dna: Dict[str, Any],
        user_profile: Dict[str, Any],
        signals: Dict[str, Any],
    ) -> str:
        top_cards = cards[:2] if isinstance(cards, list) else []
        card_titles = [str(c.get("title", "Look")) for c in top_cards if isinstance(c, dict)]
        occasion = (context.get("slots", {}) or {}).get("occasion") or context.get("occasion") or "today"
        weather = (context.get("slots", {}) or {}).get("weather") or context.get("weather") or "current weather"
        dna_style = style_dna.get("style", "your style")

        deterministic = (
            f"I curated your looks for {occasion} in {weather}. "
            f"Top picks: {', '.join(card_titles) if card_titles else 'your best wardrobe matches'}. "
            f"They align with your style DNA ({dna_style}) and recent feedback. "
            f"Pick one and I can refine it for accessories or try-on."
        )

        try:
            model_message = generate_text(
                f"""
User request: {text}
Occasion: {occasion}
Weather: {weather}
Style DNA: {style_dna}
Top cards: {top_cards}

Write a premium stylist response in 3-4 lines:
- mention why the top look works
- mention one styling tweak option
- keep it concise and natural
""",
                user_profile=user_profile,
                signals=signals,
            )
            if model_message and model_message != "none":
                return model_message
        except Exception:
            pass
        return deterministic

    def _resolve_organize_request(
        self,
        text: str,
        context: Dict[str, Any],
        slots: Dict[str, Any],
        intent: str,
    ) -> Dict[str, Any]:
        module_context = str(context.get("module_context") or "").lower()
        lowered = str(text or "").lower()
        module = str((slots or {}).get("module") or "").strip().lower() or None

        keyword_map = {
            "life board": "life_boards",
            "meal": "meal_planner",
            "medicine": "medicines",
            "meds": "medicines",
            "bill": "bills",
            "calendar": "calendar",
            "workout": "workout",
            "skin": "skincare",
            "contact": "contacts",
            "goal": "life_goals",
        }
        if not module:
            for key, value in keyword_map.items():
                if key in lowered or key in module_context:
                    module = value
                    break

        active = (
            intent == "organize_hub"
            or "organize" in lowered
            or "organize" in module_context
            or module is not None
        )
        return {"active": active, "module": module}

    def _build_organize_hub(self, user_id: str, appwrite: AppwriteProxy) -> Dict[str, Any]:
        module_specs = [
            ("life_boards", "Life Boards", "life_boards", "/organize/life-boards"),
            ("meal_planner", "Meal Planner", "meal_plans", "/organize/meal-planner"),
            ("medicines", "Medicines", "meds", "/organize/medicines"),
            ("bills", "Bills", "bills", "/organize/bills"),
            ("calendar", "Calendar", "plans", "/organize/calendar"),
            ("workout", "Workout", "workout_outfits", "/organize/workout"),
            ("skincare", "Skincare", "skincare", "/organize/skincare"),
            ("contacts", "Contacts", "users", "/organize/contacts"),
            ("life_goals", "Life Goals", "life_goals", "/organize/life-goals"),
        ]

        chips = []
        for module, title, resource, route in module_specs:
            count = 0
            chips.append(
                {
                    "id": module,
                    "title": title,
                    "kind": "chip",
                    "subtitle": f"{count} items",
                    "action": {
                        "type": "open_module",
                        "module": module,
                        "route": route,
                    },
                    "meta": {
                        "resource": resource,
                        "count": count,
                    },
                }
            )

        suggested_prompts = [
            "Plan my meals for this week",
            "Remind me about medicines",
            "Track this month's bills",
            "Add a life goal",
        ]

        return {"chips": chips, "suggested_prompts": suggested_prompts}

    def _organize_response(
        self,
        request_id: str,
        user_id: str,
        appwrite: AppwriteProxy,
        module_key: str | None,
        include_counts: bool,
    ) -> Dict[str, Any]:
        hub_payload = self._build_organize_hub(user_id=user_id, appwrite=appwrite)
        chips = hub_payload.get("chips", [])
        if include_counts:
            for chip in chips:
                meta = chip.get("meta", {}) if isinstance(chip.get("meta"), dict) else {}
                resource = str(meta.get("resource", ""))
                count = self._count_resource(appwrite=appwrite, resource=resource, user_id=user_id)
                meta["count"] = count
                chip["meta"] = meta
                chip["subtitle"] = f"{count} items"

        focus_card = self._build_organize_focus_card(module_key=module_key)
        message = "Choose what you want to organize."
        if module_key:
            message = f"Opening {focus_card.get('title', 'organize module')}. You can start from this chip."
            chips = [focus_card] + [c for c in chips if c.get("id") != focus_card.get("id")]

        return {
            "success": True,
            "request_id": request_id,
            "meta": {
                "intent": "organize_hub",
                "domain": "organize",
                "module": module_key,
            },
            "message": message,
            "board": "organize",
            "type": "chips",
            "cards": chips,
            "data": {
                "hub": hub_payload,
                "module": module_key,
                "action": "open_organize_module" if module_key else "show_organize_hub",
            },
        }

    def _build_organize_focus_card(self, module_key: str | None) -> Dict[str, Any]:
        mapping = {
            "life_boards": ("Life Boards", "/organize/life-boards"),
            "meal_planner": ("Meal Planner", "/organize/meal-planner"),
            "medicines": ("Medicines", "/organize/medicines"),
            "bills": ("Bills", "/organize/bills"),
            "calendar": ("Calendar", "/organize/calendar"),
            "workout": ("Workout", "/organize/workout"),
            "skincare": ("Skincare", "/organize/skincare"),
            "contacts": ("Contacts", "/organize/contacts"),
            "life_goals": ("Life Goals", "/organize/life-goals"),
        }
        title, route = mapping.get(module_key or "", ("Organize", "/organize"))
        return {
            "id": module_key or "organize",
            "title": title,
            "kind": "chip",
            "subtitle": "Ready",
            "action": {
                "type": "open_module",
                "module": module_key or "organize",
                "route": route,
            },
        }

    def _count_resource(self, appwrite: AppwriteProxy, resource: str, user_id: str) -> int:
        try:
            docs = appwrite.list_documents(resource, user_id=user_id, limit=50)
            return len(docs)
        except Exception:
            return 0

    def _is_wardrobe_count_query(self, text: str) -> bool:
        lowered = str(text or "").lower()
        count_words = ["how many", "count", "number of", "total", "do i have"]
        wardrobe_words = [
            "wardrobe", "closet", "outfit", "outfits", "tops", "top", "shirts", "shirt",
            "tshirt", "t-shirt", "pants", "trousers", "jeans", "bottoms", "shoes",
            "footwear", "dress", "dresses", "accessories", "jewelry", "bags", "bag",
        ]
        return any(k in lowered for k in count_words) and any(k in lowered for k in wardrobe_words)

    def _wardrobe_query_response(
        self,
        request_id: str,
        user_id: str,
        appwrite: AppwriteProxy,
        text: str,
    ) -> Dict[str, Any]:
        try:
            wardrobe_docs = appwrite.list_documents("outfits", user_id=user_id, limit=self._default_wardrobe_fetch_limit)
        except Exception:
            wardrobe_docs = []

        counts = {
            "tops": 0,
            "bottoms": 0,
            "shoes": 0,
            "dresses": 0,
            "accessories": 0,
            "unknown": 0,
        }

        for doc in wardrobe_docs:
            category = str(
                doc.get("category")
                or doc.get("category_group")
                or doc.get("type")
                or ""
            ).strip().lower()
            sub = str(doc.get("sub_category") or doc.get("subcategory") or "").strip().lower()
            text_blob = f"{category} {sub}"

            if any(k in text_blob for k in ["top", "shirt", "blouse", "tee", "t-shirt", "tshirt", "jacket", "blazer"]):
                counts["tops"] += 1
            elif any(k in text_blob for k in ["bottom", "pant", "trouser", "jean", "short", "skirt"]):
                counts["bottoms"] += 1
            elif any(k in text_blob for k in ["shoe", "sneaker", "heel", "boot", "sandal", "footwear"]):
                counts["shoes"] += 1
            elif any(k in text_blob for k in ["dress"]):
                counts["dresses"] += 1
            elif any(k in text_blob for k in ["accessory", "watch", "bag", "jewel", "necklace", "earring"]):
                counts["accessories"] += 1
            else:
                counts["unknown"] += 1

        total = len(wardrobe_docs)
        lowered = str(text or "").lower()
        requested_key = None
        requested_map = {
            "tops": ["top", "tops", "shirt", "shirts", "blouse", "blouses"],
            "bottoms": ["bottom", "bottoms", "pant", "pants", "trouser", "trousers", "jean", "jeans", "skirt", "skirts"],
            "shoes": ["shoe", "shoes", "sneaker", "sneakers", "footwear", "heels", "boots", "sandals"],
            "dresses": ["dress", "dresses"],
            "accessories": ["accessory", "accessories", "jewelry", "watch", "watches", "bag", "bags"],
        }
        for key, words in requested_map.items():
            if any(w in lowered for w in words):
                requested_key = key
                break

        if requested_key:
            message = f"You have {counts.get(requested_key, 0)} {requested_key} in your wardrobe."
        else:
            message = (
                f"You currently have {total} items: "
                f"{counts['tops']} tops, {counts['bottoms']} bottoms, {counts['shoes']} shoes, "
                f"{counts['dresses']} dresses, and {counts['accessories']} accessories."
            )

        cards = [
            {"id": "tops", "title": "Tops", "kind": "stat", "value": counts["tops"]},
            {"id": "bottoms", "title": "Bottoms", "kind": "stat", "value": counts["bottoms"]},
            {"id": "shoes", "title": "Shoes", "kind": "stat", "value": counts["shoes"]},
            {"id": "dresses", "title": "Dresses", "kind": "stat", "value": counts["dresses"]},
            {"id": "accessories", "title": "Accessories", "kind": "stat", "value": counts["accessories"]},
        ]

        return {
            "success": True,
            "request_id": request_id,
            "meta": {"intent": "wardrobe_query", "domain": "wardrobe"},
            "message": message,
            "board": "wardrobe",
            "type": "stats",
            "cards": cards,
            "data": {"counts": counts, "total_items": total},
        }

    def _is_plan_pack_request(self, text: str, context: Dict[str, Any]) -> bool:
        lowered = str(text or "").lower()
        module_context = str(context.get("module_context") or "").lower()
        keywords = [
            "plan trip", "trip plan", "packing list", "pack for", "business travel",
            "wedding checklist", "goa trip", "vacation", "travel checklist",
        ]
        return any(k in lowered for k in keywords) or "plan_pack" in module_context

    def _is_daily_dependency_request(self, text: str, context: Dict[str, Any]) -> bool:
        lowered = str(text or "").lower()
        module_context = str(context.get("module_context") or "").lower()
        keywords = [
            "daily plan", "daily cards", "morning flow", "midday flow", "afternoon flow",
            "evening flow", "night flow", "tomorrow preview", "day planner", "daily dependency",
        ]
        return any(k in lowered for k in keywords) or "daily_dependency" in module_context


ahvi_orchestrator = AhviOrchestrator()



