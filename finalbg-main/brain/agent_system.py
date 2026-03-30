from typing import Any, Dict, List
import json

from services.llm_service import generate_text


class AgentSystem:
    """
    🔥 Hybrid Planner:
    - Uses LLM (Ollama) for dynamic planning
    - Falls back to deterministic rules
    """

    # -------------------------
    # MAIN ENTRY
    # -------------------------
    def plan(self, intent: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            ai_plan = self._llm_plan(intent, context)

            if ai_plan:
                return ai_plan

        except Exception as e:
            print("⚠️ LLM planning failed:", str(e))

        # 🔥 fallback (SAFE)
        return self._rule_based_plan(intent)

    # -------------------------
    # 🧠 LLM PLANNER
    # -------------------------
    def _llm_plan(self, intent: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt = self._build_prompt(intent, context)

        response = generate_text(prompt)

        if not response or response == "none":
            return []

        try:
            parsed = json.loads(response)

            if isinstance(parsed, list):
                return parsed

        except Exception:
            print("⚠️ Invalid LLM plan JSON:", response)

        return []

    # -------------------------
    # 🧾 PROMPT BUILDER
    # -------------------------
    def _build_prompt(self, intent: str, context: Dict[str, Any]) -> str:
        slots = context.get("slots", {})

        return f"""
You are an AI planning engine for a fashion assistant.

Your job:
Return a JSON list of steps to execute.

Rules:
- Output ONLY valid JSON
- No explanation
- Each step must have:
  - step
  - agent

Available steps:
- normalize_context
- build_style_graph
- generate_score_rank
- persist_and_feedback_hooks
- generate_boards
- prepare_tryon
- run_tryon_model

User intent: {intent}

Context:
- occasion: {slots.get("occasion")}
- weather: {slots.get("weather")}
- time: {slots.get("time_of_day")}
- location: {slots.get("location")}

Examples:

daily_outfit →
[
  {{"step": "normalize_context", "agent": "context_agent"}},
  {{"step": "build_style_graph", "agent": "style_graph_agent"}},
  {{"step": "generate_score_rank", "agent": "outfit_agent"}},
  {{"step": "persist_and_feedback_hooks", "agent": "memory_agent"}}
]

tryon →
[
  {{"step": "prepare_tryon", "agent": "tryon_agent"}},
  {{"step": "run_tryon_model", "agent": "vision_agent"}}
]

Now generate plan:
"""

    # -------------------------
    # 🛡️ FALLBACK RULES
    # -------------------------
    def _rule_based_plan(self, intent: str) -> List[Dict[str, Any]]:
        if intent == "daily_outfit":
            return [
                {"step": "normalize_context", "agent": "context_agent"},
                {"step": "build_style_graph", "agent": "style_graph_agent"},
                {"step": "generate_score_rank", "agent": "outfit_agent"},
                {"step": "persist_and_feedback_hooks", "agent": "memory_agent"},
            ]

        if intent == "tryon":
            return [
                {"step": "prepare_tryon", "agent": "tryon_agent"},
                {"step": "run_tryon_model", "agent": "vision_agent"},
            ]

        return [{"step": "no_op", "agent": "fallback_agent"}]


agent_system = AgentSystem()