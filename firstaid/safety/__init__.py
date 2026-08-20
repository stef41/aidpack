"""Safety layer: red flags, forbidden-advice lint, scope guard."""
from .forbidden import FORBIDDEN_RULES, SCOPE_RESPONSES, lint_advice, scope_guard
from .redflags import RedFlag, scan_red_flags

__all__ = ["scan_red_flags", "RedFlag", "lint_advice", "scope_guard",
           "SCOPE_RESPONSES", "FORBIDDEN_RULES"]
