"""Keyword-based baseline prompt grader (U1)."""
RUBRIC = {
    "Role": ["act as", "you are", "as a"],
    "Context": ["background", "for beginners", "audience", "context", "given that"],
    "Task": ["explain", "create", "write", "analyse", "analyze", "summarize", "compare", "describe", "generate"],
    "Format": ["bullet points", "table", "headings", "paragraph", "list", "format", "json"],
    "Constraints": ["under", "maximum", "limit", "must", "should", "don't", "less than", "word", "avoid"],
}
SUGGESTIONS = {
    "Role": "Add a role, e.g. 'Act as a Python teacher'.",
    "Context": "Add background information and identify the target audience.",
    "Task": "State exactly what the AI should do.",
    "Format": "Specify the desired output format.",
    "Constraints": "Add word limits, tone, scope or restrictions.",
}
def grade_prompt(prompt):
    text = prompt.lower()
    scores = {name: (20 if any(k in text for k in keys) else 0) for name, keys in RUBRIC.items()}
    return scores, sum(scores.values())
def get_feedback(scores):
    return [f"{name}: Missing. {SUGGESTIONS[name]}" for name, score in scores.items() if score == 0]
