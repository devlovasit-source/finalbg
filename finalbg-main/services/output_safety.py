from __future__ import annotations

import re


_BLOCKLIST_PATTERNS = [
    r"\bapi[_\s-]?key\b",
    r"\bsecret\b",
    r"\bsystem prompt\b",
    r"\bdeveloper message\b",
]


def sanitize_llm_output(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return "I couldn't generate a safe response right now."

    lowered = raw.lower()
    for pattern in _BLOCKLIST_PATTERNS:
        if re.search(pattern, lowered):
            return "I can’t share internal system details, but I can still help with your style request."
    return raw
