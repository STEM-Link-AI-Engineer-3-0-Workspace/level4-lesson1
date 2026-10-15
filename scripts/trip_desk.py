"""Checks for the 14 breakout. Not a lesson concept, and not part of your answer.

The right answer is stored as short hashes, so reading this file won't give it
away.
"""

import hashlib
import re

FIELDS = ("train_id", "travel_class", "hotel", "cost_per_person_usd", "within_budget", "reason")

ANSWER = {
    "train_id": "f03905023437",
    "travel_class": "c21365c7ea5b",
    "hotel": "21131e8431c9",
    "cost_per_person_usd": "111ffe697f73",
}


def _hash(value) -> str:
    return hashlib.sha256(str(value).strip().lower().encode()).hexdigest()[:12]


def check_prompt(prompt: str, hotel_names: list[str]) -> None:
    """Rule 1: a system prompt with no digits and no hotel names in it."""
    facts = re.findall(r"\d+", prompt) + [n for n in hotel_names if n.lower() in prompt.lower()]
    if facts:
        print("Rule 1 broken. Facts in the prompt:", facts)
    else:
        print("Rule 1 OK: no digits or hotel names in the prompt.")


def scorecard(plan, calls: list, prompt: str) -> None:
    """Score a TripPlan against the customer's message, out of 7."""
    shape_ok = plan is not None and all(hasattr(plan, f) for f in FIELDS)

    def right(field):
        if not shape_ok:
            return False
        value = getattr(plan, field)
        if field == "cost_per_person_usd":
            if not isinstance(value, (int, float)):
                return False
            value = f"{value:.2f}"
        return _hash(value) == ANSWER[field]

    checks = [
        ("Answered as a TripPlan with all six fields (1 point)", 1, shape_ok),
        ("The right train (1 point)", 1, right("train_id")),
        ("The right class, the best one actually available (2 points)", 2, right("travel_class")),
        ("The right hotel, best rated within budget (2 points)", 2, right("hotel")),
        ("The right cost per person (1 point)", 1, right("cost_per_person_usd")),
    ]
    total = sum(points for _, points, ok in checks if ok)
    for label, points, ok in checks:
        print("✓" if ok else "✗", label)
    print()
    print("Score:", total, "out of 7. The hidden round is worth 3 more.")

    used = [call[0] for call in calls]
    print()
    print("Checked seats before choosing a class:", "seats_left" in used)
    print("Did the sums with calculate:", "calculate" in used)
    print("Converted to dollars with lkr_to_usd:", "lkr_to_usd" in used)
    if re.findall(r"\d+", prompt):
        print()
        print("Rule 1 broken: points don't count until the prompt has no digits.")
