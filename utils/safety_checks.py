"""U5: transparent rule-based bias and personal-data screening."""
import re

# These patterns are deliberately conservative examples, not a complete DLP system.
PERSONAL_PATTERNS = {
    "Email address": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "Phone-like number": r"(?<!\w)(?:\+?\d{1,3}[-.\s]?)?(?:\d[-.\s]?){9,12}\d(?!\w)",
    "Possible Aadhaar-like 12-digit number": r"(?<!\d)\d{12}(?!\d)",
    "Possible API key/token marker": r"\b(?:api[_ -]?key|secret|access[_ -]?token|password)\s*[:=]\s*\S+",
}
BIAS_TERMS = [
    "all women are", "all men are", "women cannot", "men cannot",
    "people from this country are", "people of this religion are",
    "only young people", "only old people", "not suitable for women",
    "not suitable for men", "the disabled are", "normal people",
]
def run_safety_checks(text):
    personal = []
    for label, pattern in PERSONAL_PATTERNS.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            personal.append(label)
    lowered = text.lower()
    bias = [term for term in BIAS_TERMS if term in lowered]
    return {
        "personal_data": {"flagged": bool(personal), "matches": personal},
        "bias": {"flagged": bool(bias), "matches": bias},
    }
