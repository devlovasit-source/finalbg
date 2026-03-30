import math
import re
from typing import Any, Dict, List


def _parse_days(text: str) -> int:
    lowered = (text or "").lower()
    patterns = [
        r"(\d+)\s*[- ]?\s*day",
        r"(\d+)\s*days",
        r"for\s+(\d+)\s*days",
    ]
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            try:
                return max(1, min(21, int(match.group(1))))
            except Exception:
                pass

    words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "week": 7,
    }
    for word, value in words.items():
        if f"{word} day" in lowered or f"{word}-day" in lowered:
            return value
    return 3


def _detect_scenario(text: str) -> str:
    lowered = (text or "").lower()
    if any(k in lowered for k in ["wedding", "marriage", "bride", "groom"]):
        return "wedding"
    if any(k in lowered for k in ["business", "work trip", "conference", "client meeting"]):
        return "business"
    if any(k in lowered for k in ["goa", "beach", "vacation", "holiday", "trip"]):
        return "travel"
    if any(k in lowered for k in ["pack", "packing"]):
        return "travel"
    return "general"


def _extract_destination(text: str) -> str:
    lowered = (text or "").lower()
    m = re.search(r"(?:to|for)\s+([a-z ]{2,30})(?:trip|travel|vacation|holiday|wedding)?", lowered)
    if m:
        return m.group(1).strip().title()
    if "goa" in lowered:
        return "Goa"
    if "business" in lowered:
        return "Business Travel"
    if "wedding" in lowered:
        return "Wedding Event"
    return "Your Trip"


def _packing_clothes(days: int, scenario: str) -> List[str]:
    tops = days + 1
    bottoms = max(2, math.ceil(days * 0.6))
    innerwear = days + 1
    socks = days + 1
    sleepwear = max(1, math.ceil(days / 3))
    footwear = 2
    outerwear = 1

    if scenario == "business":
        footwear = 2
        outerwear = 1
    if scenario == "wedding":
        footwear = 3
        outerwear = 1
    if scenario == "travel":
        footwear = 2
        outerwear = 1

    return [
        f"Tops x{tops}",
        f"Bottoms x{bottoms}",
        f"Innerwear x{innerwear}",
        f"Socks x{socks}",
        f"Sleepwear x{sleepwear}",
        f"Footwear x{footwear}",
        f"Outer layer x{outerwear}",
    ]


def _scenario_addons(scenario: str) -> List[str]:
    if scenario == "business":
        return [
            "Formal shirt/blouse",
            "Blazer",
            "Laptop + charger",
            "Business cards",
            "Meeting-ready shoes",
        ]
    if scenario == "wedding":
        return [
            "Main wedding outfit",
            "Backup festive outfit",
            "Jewelry/accessories",
            "Ethnic footwear",
            "Gift envelope",
        ]
    if scenario == "travel":
        return [
            "Sunscreen",
            "Sunglasses",
            "Beachwear/swimwear",
            "Toiletries kit",
            "Power bank",
        ]
    return [
        "Toiletries kit",
        "Phone + charger",
        "Personal medicine",
    ]


def _normalize_weather(context: Dict[str, Any]) -> str:
    weather = str(context.get("weather") or "").lower()
    if not weather:
        weather = str((context.get("weather_data") or {}).get("condition") or "").lower()
    if any(k in weather for k in ["rain", "storm", "drizzle"]):
        return "rainy"
    if any(k in weather for k in ["cold", "chill", "winter"]):
        return "cold"
    if any(k in weather for k in ["hot", "heat", "humid", "warm", "summer"]):
        return "hot"
    return "mild"


def _time_of_day(context: Dict[str, Any]) -> str:
    value = str(context.get("time_of_day") or context.get("time") or "").lower()
    if value in ("morning", "afternoon", "evening", "night"):
        return value
    return "daytime"


def _weather_layer_items(weather: str) -> List[str]:
    if weather == "rainy":
        return ["Compact rain jacket", "Waterproof footwear", "Quick-dry bag cover"]
    if weather == "cold":
        return ["Warm jacket", "Thermal innerwear", "Socks x2 extra"]
    if weather == "hot":
        return ["Breathable cotton/linen", "Cap/hat", "Hydration bottle"]
    return ["Light layer for evenings"]


def _time_based_tasks(time_of_day: str) -> List[str]:
    if time_of_day == "morning":
        return ["Pack documents and chargers the night before", "Keep a quick breakfast/snack ready"]
    if time_of_day == "evening":
        return ["Keep one ready-to-wear outfit on top", "Add travel-size freshening kit"]
    if time_of_day == "night":
        return ["Keep sleepwear and essentials accessible", "Add eye mask/comfort kit"]
    return ["Keep first-day essentials in carry-on"]


def _timeline_checklist(days: int, scenario: str) -> List[str]:
    base = [
        "Confirm travel/event dates",
        "Book transport and stay",
        "Prepare outfits by day",
        "Pack essentials the night before",
    ]
    if scenario == "business":
        base.extend(["Prepare meeting deck", "Keep IDs and booking invoices ready"])
    if scenario == "wedding":
        base.extend(["Confirm ceremony timeline", "Coordinate with family/group"])
    if days >= 5:
        base.append("Add laundry plan for longer stay")
    return base


def _ui_cards(days: int, destination: str, scenario: str, weather: str, time_of_day: str) -> List[Dict[str, Any]]:
    clothes = _packing_clothes(days=days, scenario=scenario)
    addons = _scenario_addons(scenario=scenario)
    timeline = _timeline_checklist(days=days, scenario=scenario)
    weather_items = _weather_layer_items(weather=weather)
    timeline = timeline + _time_based_tasks(time_of_day=time_of_day)

    if scenario == "wedding":
        primary_title = "Wedding Prep Checklist"
    elif scenario == "business":
        primary_title = "Business Travel Plan"
    else:
        primary_title = f"{days}-Day Plan"

    return [
        {
            "id": "trip_plan",
            "title": primary_title,
            "kind": "checklist",
            "subtitle": destination,
            "items": timeline,
            "action": {"type": "open_module", "module": "calendar", "route": "/organize/calendar"},
        },
        {
            "id": "packing_clothes",
            "title": "Packing List - Clothes",
            "kind": "checklist",
            "subtitle": f"{days} days",
            "items": clothes,
            "action": {"type": "open_module", "module": "life_boards", "route": "/organize/life-boards"},
        },
        {
            "id": "packing_essentials",
            "title": "Packing List - Essentials",
            "kind": "checklist",
            "subtitle": scenario.title(),
            "items": addons,
            "action": {"type": "open_module", "module": "life_boards", "route": "/organize/life-boards"},
        },
        {
            "id": "weather_time_adjustments",
            "title": "Weather & Time Adjustments",
            "kind": "checklist",
            "subtitle": f"{weather.title()} | {time_of_day.title()}",
            "items": weather_items,
            "action": {"type": "open_module", "module": "calendar", "route": "/organize/calendar"},
        },
    ]


def build_plan_pack_response(text: str, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    context = context or {}
    days = _parse_days(text)
    scenario = _detect_scenario(text)
    destination = _extract_destination(text)
    weather = _normalize_weather(context=context)
    time_of_day = _time_of_day(context=context)

    cards = _ui_cards(days=days, destination=destination, scenario=scenario, weather=weather, time_of_day=time_of_day)

    return {
        "intent": "plan_pack",
        "message": f"Built your {scenario} plan and weather-aware packing checklist for {days} days.",
        "board": "plan_pack",
        "type": "checklists",
        "cards": cards,
        "data": {
            "days": days,
            "destination": destination,
            "scenario": scenario,
            "weather": weather,
            "time_of_day": time_of_day,
            "can_save_to_life_board": True,
            "source_text": text,
        },
    }
