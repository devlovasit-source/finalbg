from __future__ import annotations

import re
from typing import Tuple


_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"reveal\s+(the\s+)?system\s+prompt",
    r"show\s+(the\s+)?system\s+prompt",
    r"output\s+(the\s+)?system\s+prompt",
    r"developer\s+message",
    r"jailbreak",
    r"bypass\s+safety",
    r"act\s+as\s+.*without\s+restrictions",
]


def detect_prompt_injection(text: str) -> Tuple[bool, str]:
    raw = str(text or "").strip()
    if not raw:
        return False, ""
    lowered = raw.lower()
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            return True, pattern
    return False, ""
