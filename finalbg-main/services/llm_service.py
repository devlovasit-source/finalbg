import os
import requests
import re
import time
import json
import hashlib
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

from services.output_safety import sanitize_llm_output
from brain.tone.tone_engine import tone_engine

# =========================
# CONFIG
# =========================
load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
MODEL_FALLBACKS = [
    m.strip()
    for m in os.getenv(
        "OLLAMA_MODEL_FALLBACKS",
        "mistral:7b,qwen2.5:3b,llama3:latest,qwen2.5vl:latest",
    ).split(",")
    if m.strip()
]
ALLOW_HEAVY_MODELS = os.getenv("OLLAMA_ALLOW_HEAVY_MODELS", "false").lower() in {"1", "true", "yes"}
DEFAULT_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "1024"))
DEFAULT_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "220"))
LLM_CACHE_ENABLED = os.getenv("LLM_CACHE_ENABLED", "true").lower() in {"1", "true", "yes"}
LLM_CACHE_TTL_SECONDS = int(os.getenv("LLM_CACHE_TTL_SECONDS", "900"))
LLM_CACHE_MAX_ENTRIES = int(os.getenv("LLM_CACHE_MAX_ENTRIES", "5000"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


def _is_heavy_model(model_name: str) -> bool:
    m = str(model_name or "").lower()
    heavy_markers = [":7b", ":8b", ":9b", ":13b", ":14b", ":32b", ":70b", "latest", "vl"]
    return any(marker in m for marker in heavy_markers)

# =========================
# SESSION WITH RETRIES
# =========================
session = requests.Session()
retries = Retry(total=2, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
session.mount("http://", HTTPAdapter(max_retries=retries))

_memory_cache: dict[str, tuple[str, float]] = {}


def _semantic_fingerprint(text: str) -> str:
    lowered = str(text or "").lower()
    tokens = re.findall(r"[a-z0-9]+", lowered)
    stop = {
        "the", "a", "an", "is", "are", "to", "for", "of", "and", "in", "on",
        "with", "please", "me", "my", "you", "your", "it", "this", "that",
    }
    compact = sorted({t for t in tokens if t and t not in stop})[:60]
    if not compact:
        compact = tokens[:30]
    basis = " ".join(compact)
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def _cache_key(kind: str, *, model: str, primary_text: str, extra: dict | None = None) -> str:
    payload = {
        "kind": kind,
        "model": model,
        "semantic": _semantic_fingerprint(primary_text),
        "extra": extra or {},
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "llm_cache:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _redis_get(key: str) -> str | None:
    if not LLM_CACHE_ENABLED:
        return None
    try:
        import redis

        client = redis.Redis.from_url(REDIS_URL)
        val = client.get(key)
        if val is None:
            return None
        return val.decode("utf-8") if isinstance(val, (bytes, bytearray)) else str(val)
    except Exception:
        return None


def _redis_set(key: str, value: str) -> None:
    if not LLM_CACHE_ENABLED:
        return
    try:
        import redis

        client = redis.Redis.from_url(REDIS_URL)
        client.setex(key, int(LLM_CACHE_TTL_SECONDS), value)
    except Exception:
        pass


def _memory_get(key: str) -> str | None:
    if not LLM_CACHE_ENABLED:
        return None
    row = _memory_cache.get(key)
    if not row:
        return None
    value, expires_at = row
    if time.time() > float(expires_at):
        _memory_cache.pop(key, None)
        return None
    return value


def _memory_set(key: str, value: str) -> None:
    if not LLM_CACHE_ENABLED:
        return
    if len(_memory_cache) > max(100, LLM_CACHE_MAX_ENTRIES):
        # opportunistic eviction of expired entries
        now = time.time()
        expired = [k for k, v in _memory_cache.items() if now > float(v[1])]
        for k in expired[:1000]:
            _memory_cache.pop(k, None)
        if len(_memory_cache) > max(100, LLM_CACHE_MAX_ENTRIES):
            _memory_cache.pop(next(iter(_memory_cache)))
    _memory_cache[key] = (value, time.time() + float(LLM_CACHE_TTL_SECONDS))


def _cache_get(key: str) -> str | None:
    return _redis_get(key) or _memory_get(key)


def _cache_set(key: str, value: str) -> None:
    _redis_set(key, value)
    _memory_set(key, value)


def _model_candidates(requested_model: str | None) -> list[str]:
    ordered: list[str] = []
    requested = str(requested_model or "").strip()
    if requested and (ALLOW_HEAVY_MODELS or not _is_heavy_model(requested)):
        first_model = requested
    else:
        first_model = DEFAULT_MODEL

    for model in [first_model, *MODEL_FALLBACKS]:
        m = str(model or "").strip()
        if not m:
            continue
        if not ALLOW_HEAVY_MODELS and _is_heavy_model(m):
            continue
        if m not in ordered:
            ordered.append(m)
    return ordered


def _merged_options(incoming: dict | None) -> dict:
    merged = {
        "num_ctx": DEFAULT_NUM_CTX,
        "num_predict": DEFAULT_NUM_PREDICT,
        "temperature": 0.7,
    }
    if incoming:
        merged.update(incoming)
    return merged


def _stylist_guidance(user_profile=None, signals=None) -> str:
    user_profile = user_profile or {}
    signals = signals or {}
    context_mode = str(signals.get("context_mode", "general")).lower()
    if context_mode != "styling":
        return ""

    preferred_colors = user_profile.get("preferred_colors", user_profile.get("colors", []))
    style = user_profile.get("style", "")
    body_type = user_profile.get("body_type", "")
    budget = user_profile.get("budget", "")

    return f"""
Advanced Stylist Rules:
- Prioritize occasion, weather, and comfort first.
- Use wardrobe-aware recommendations and avoid generic trends.
- Give one best choice first, then one alternative.
- Add one practical upgrade (accessory, layer, or color swap).
- Mention confidence rationale in plain language.
- Keep output actionable and premium.

User style profile:
- style: {style}
- preferred colors: {preferred_colors}
- body type: {body_type}
- budget: {budget}
"""


# =========================
# SAFE REQUEST HANDLER
# =========================
def safe_request(endpoint: str, payload: dict, timeout: int = 30):
    candidates = _model_candidates(payload.get("model"))
    last_error = ""

    for model in candidates:
        try:
            local_payload = dict(payload)
            local_payload["model"] = model
            local_payload["options"] = _merged_options(local_payload.get("options"))
            response = session.post(f"{OLLAMA_URL}/{endpoint}", json=local_payload, timeout=timeout)

            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict):
                    data["_model_used"] = model
                return data

            last_error = f"{response.status_code}: {response.text}"
            # Try next model when model is missing.
            if response.status_code == 404 and "not found" in response.text.lower():
                continue

        except Exception as e:
            last_error = str(e)
            continue

    if last_error:
        print(f"OLLAMA ERROR ({endpoint}): {last_error}")
    return None


# =========================
# TONE-AWARE TEXT GENERATION
# =========================
def generate_text(prompt: str, options: dict = None, user_profile=None, signals=None) -> str:
    if not prompt:
        return "none"

    tone = tone_engine.build_prompt_tone(user_profile, signals)

    full_prompt = f"""
You are AHVI, a premium AI fashion stylist.

Tone Instructions:
{tone.get("tone_instruction", "")}

Guidelines:
- Be natural and human
- Keep responses concise
- Sound confident but not arrogant
- Avoid robotic phrasing

{_stylist_guidance(user_profile=user_profile, signals=signals)}

{prompt}
"""

    payload = {
        "model": DEFAULT_MODEL,
        "prompt": full_prompt,
        "stream": False,
    }

    if options:
        payload["options"] = options

    key = _cache_key(
        "generate",
        model=payload.get("model", DEFAULT_MODEL),
        primary_text=prompt,
        extra={"options": payload.get("options", {})},
    )
    cached = _cache_get(key)
    if cached:
        return sanitize_llm_output(cached)

    data = safe_request("generate", payload, timeout=30)

    if not data:
        return "none"

    response = data.get("response", "").strip() or "none"
    _cache_set(key, response)
    response = tone_engine.apply(response, user_profile=user_profile, signals=signals)
    return sanitize_llm_output(response)


# =========================
# CHAT COMPLETION
# =========================
def chat_completion(messages: list, system_instruction: str = "", model: str = DEFAULT_MODEL,
                    user_profile=None, signals=None) -> str:

    if not messages:
        return "I didn't catch that!"

    tone = tone_engine.build_prompt_tone(user_profile, signals)

    formatted_messages = []

    system_msg = f"""
You are AHVI, an AI fashion stylist.

Tone:
{tone.get("tone_instruction", "")}

Rules:
- Speak naturally
- Keep it concise
- Be stylish and modern

{_stylist_guidance(user_profile=user_profile, signals=signals)}
"""

    if system_instruction:
        system_msg += "\n" + system_instruction[:2000]

    formatted_messages.append({"role": "system", "content": system_msg})

    safe_messages = messages[-10:]

    for msg in safe_messages:
        role = str(msg.get("role", "user")).lower()
        if role not in ["user", "assistant", "system"]:
            role = "assistant"

        content = str(msg.get("content", ""))[:4000]

        if content:
            formatted_messages.append({"role": role, "content": content})

    payload = {
        "model": model or DEFAULT_MODEL,
        "messages": formatted_messages,
        "stream": False,
    }

    last_user = ""
    for m in reversed(formatted_messages):
        if str(m.get("role", "")).lower() == "user":
            last_user = str(m.get("content", ""))
            break
    key = _cache_key(
        "chat",
        model=payload.get("model", DEFAULT_MODEL),
        primary_text=last_user or json.dumps(formatted_messages[-2:], ensure_ascii=False),
        extra={"system_instruction": system_instruction[:200]},
    )
    cached = _cache_get(key)
    if cached:
        return sanitize_llm_output(cached)

    data = safe_request("chat", payload, timeout=45)

    if not data:
        return "I'm having trouble thinking right now. Try again in a moment."

    try:
        response = data.get("message", {}).get("content", "").strip()
        _cache_set(key, response)
        response = tone_engine.apply(response, user_profile=user_profile, signals=signals)
        return sanitize_llm_output(response or "Something went wrong.")
    except Exception:
        return "AI response parsing failed."


# =========================
# WARDROBE FORMATTER
# =========================
def format_wardrobe_for_llm(items):
    if not items:
        return "The user's wardrobe is empty."

    msg = "User wardrobe:\n"

    for item in items[:50]:
        category = item.get("category_group", "")
        sub = item.get("subcategory", "")
        color = item.get("colors", {}).get("primary", "") if isinstance(item.get("colors"), dict) else item.get("color", "")
        msg += f"- {color} {sub} ({category})\n"

    return msg


# =========================
# OUTFIT EXPLANATION
# =========================
def generate_outfit_explanation(outfits: list, context: str = "", user_profile=None, signals=None):
    prompt = f"""
User wardrobe:
{context}

Outfits:
{outfits}

Explain:
- why these outfits work
- when to wear them

Keep it short (2-3 lines).
"""
    return generate_text(prompt, user_profile=user_profile, signals=signals)


# =========================
# STYLE ADVICE
# =========================
def generate_style_advice(user_input: str, wardrobe_summary: str, user_profile=None, signals=None):
    prompt = f"""
User request:
{user_input}

Wardrobe:
{wardrobe_summary}

Give practical styling advice using available wardrobe.
Keep it concise and helpful.
"""
    return generate_text(prompt, user_profile=user_profile, signals=signals)


# =========================
# SMART RESPONSE GENERATOR
# =========================
def generate_ai_response(user_input: str, outfits: list, wardrobe_items: list, user_profile=None, signals=None):
    wardrobe_summary = format_wardrobe_for_llm(wardrobe_items)

    if outfits:
        return generate_outfit_explanation(outfits, wardrobe_summary, user_profile=user_profile, signals=signals)

    return generate_style_advice(user_input, wardrobe_summary, user_profile=user_profile, signals=signals)
