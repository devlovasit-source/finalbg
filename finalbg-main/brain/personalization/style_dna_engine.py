import json
import os
from collections import Counter
from threading import Lock
from typing import Any, Dict, List


class StyleDNAEngine:
    """
    Style DNA Learning:
    - Builds personalization signals from profile + history + feedback memory.
    - Persists compact per-user DNA to support continuous improvement.
    """

    def __init__(self) -> None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._dna_path = os.path.join(base_dir, "data", "style_dna_memory.json")
        self._feedback_memory_path = os.path.join(base_dir, "data", "outfit_memory.json")
        self._lock = Lock()

    def build(self, context: Dict[str, Any]) -> Dict[str, Any]:
        user_id = str(context.get("user_id") or "anonymous")
        profile = context.get("user_profile") or {}
        history = context.get("history") or []

        with self._lock:
            dna_state = self._load_json(self._dna_path, fallback={"users": {}})
            feedback_memory = self._load_json(self._feedback_memory_path, fallback={"users": {}})

            learned_dna = self._build_dna(
                profile=profile,
                history=history,
                previous_dna=((dna_state.get("users", {}) or {}).get(user_id, {}) or {}),
                feedback_user=((feedback_memory.get("users", {}) or {}).get(user_id, {}) or {}),
            )

            dna_state.setdefault("users", {})[user_id] = learned_dna
            self._save_json(self._dna_path, dna_state)
            return learned_dna

    def enrich_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        context["style_dna"] = self.build(context)
        return context

    def score_item(self, item: Dict[str, Any], dna: Dict[str, Any]) -> int:
        score = 0
        item = item or {}
        dna = dna or {}

        color = str(item.get("color", "")).lower()
        fabric = str(item.get("fabric", "")).lower()
        style = str(item.get("style", "")).lower()
        item_type = str(item.get("type") or item.get("category") or "").lower()

        if color and color in dna.get("preferred_colors", []):
            score += 2
        if fabric and fabric in dna.get("preferred_fabrics", []):
            score += 1
        if style and style in dna.get("preferred_styles", []):
            score += 2
        if item_type and item_type in dna.get("preferred_types", []):
            score += 1
        if item_type and item_type in dna.get("disliked_items", []):
            score -= 3
        return score

    def _build_dna(
        self,
        profile: Dict[str, Any],
        history: List[Dict[str, Any]],
        previous_dna: Dict[str, Any],
        feedback_user: Dict[str, Any],
    ) -> Dict[str, Any]:
        profile = profile or {}
        previous_dna = previous_dna or {}
        feedback_user = feedback_user or {}

        liked = feedback_user.get("liked_outfits", []) or []
        disliked = feedback_user.get("disliked_outfits", []) or []

        liked_colors = Counter()
        liked_fabrics = Counter()
        liked_types = Counter()
        disliked_types = Counter()

        for outfit in liked[:80]:
            for part in ("top", "bottom", "shoes"):
                item = outfit.get(part, {}) if isinstance(outfit, dict) else {}
                liked_colors.update([str(item.get("color", "")).lower()])
                liked_fabrics.update([str(item.get("fabric", "")).lower()])
                liked_types.update([str(item.get("type") or item.get("category") or "").lower()])

        for outfit in disliked[:80]:
            for part in ("top", "bottom", "shoes"):
                item = outfit.get(part, {}) if isinstance(outfit, dict) else {}
                disliked_types.update([str(item.get("type") or item.get("category") or "").lower()])

        history_styles = Counter()
        for event in history[-40:]:
            if not isinstance(event, dict):
                continue
            slots = event.get("slots", {}) if isinstance(event.get("slots"), dict) else {}
            style_value = str(slots.get("style") or event.get("style") or "").lower()
            if style_value:
                history_styles.update([style_value])

        def _top(counter: Counter, n: int = 5) -> List[str]:
            return [k for k, _ in counter.most_common(n) if k]

        preferred_colors = self._merge_unique(
            [str(v).lower() for v in profile.get("preferred_colors", []) if v],
            previous_dna.get("preferred_colors", []),
            _top(liked_colors, 6),
        )
        preferred_fabrics = self._merge_unique(
            [str(v).lower() for v in profile.get("preferred_fabrics", []) if v],
            previous_dna.get("preferred_fabrics", []),
            _top(liked_fabrics, 6),
        )
        preferred_styles = self._merge_unique(
            [str(v).lower() for v in profile.get("preferred_styles", []) if v],
            [str(profile.get("style", "casual")).lower()],
            previous_dna.get("preferred_styles", []),
            _top(history_styles, 4),
        )
        preferred_types = self._merge_unique(
            previous_dna.get("preferred_types", []),
            _top(liked_types, 8),
        )
        disliked_items = self._merge_unique(
            [str(v).lower() for v in profile.get("disliked_items", []) if v],
            previous_dna.get("disliked_items", []),
            _top(disliked_types, 8),
        )

        return {
            "style": str(profile.get("style") or (preferred_styles[0] if preferred_styles else "casual")).lower(),
            "preferred_colors": preferred_colors[:10],
            "preferred_fabrics": preferred_fabrics[:10],
            "preferred_styles": preferred_styles[:8],
            "preferred_types": preferred_types[:10],
            "disliked_items": disliked_items[:10],
        }

    @staticmethod
    def _merge_unique(*groups: List[str]) -> List[str]:
        result: List[str] = []
        seen = set()
        for group in groups:
            if not isinstance(group, list):
                continue
            for value in group:
                key = str(value).strip().lower()
                if not key or key in seen:
                    continue
                seen.add(key)
                result.append(key)
        return result

    @staticmethod
    def _load_json(path: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
        if not os.path.exists(path):
            return dict(fallback)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
        return dict(fallback)

    @staticmethod
    def _save_json(path: str, payload: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)


style_dna_engine = StyleDNAEngine()

__all__ = ["style_dna_engine", "StyleDNAEngine"]
