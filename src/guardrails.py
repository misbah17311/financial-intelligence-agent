"""
Safety guardrails for the Financial Intelligence Agent.

These run BEFORE and AFTER the agent pipeline to catch:
  - SQL injection attempts
  - Prompt injection / jailbreak attempts
  - PII in queries (credit cards, SSNs, etc.)
  - Off-topic queries that waste compute
  - Toxic or harmful content in responses

Each check returns a (passed: bool, reason: str) tuple.
The pipeline short-circuits on the first failure — no LLM calls happen.
"""

import re
from src.logger import logger


# ---------------------------------------------------------------------------
# 1. SQL injection detection
# ---------------------------------------------------------------------------

_SQL_INJECTION_PATTERNS = [
    r";\s*(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|CREATE|EXEC|EXECUTE)\b",
    r"--\s*$",                             # SQL comment at end
    r"UNION\s+(ALL\s+)?SELECT",            # UNION-based injection
    r"'\s*OR\s+'?\d*'?\s*=\s*'?\d*",      # ' OR '1'='1
    r"'\s*;\s*--",                         # break + comment
    r"xp_cmdshell|sp_executesql",          # MSSQL procs
    r"LOAD_EXTENSION|COPY\s+TO|COPY\s+FROM",  # DuckDB-specific escape
    r"ATTACH\s+DATABASE",                  # attach external DB
    r"INTO\s+OUTFILE|INTO\s+DUMPFILE",     # file write
    r"INFORMATION_SCHEMA",                 # schema enumeration
]

def check_sql_injection(query: str) -> tuple[bool, str]:
    """Screen user input for SQL injection patterns."""
    upper = query.upper()
    for pattern in _SQL_INJECTION_PATTERNS:
        if re.search(pattern, upper):
            logger.warning(f"SQL injection blocked: matched pattern '{pattern}' in '{query[:80]}'")
            return False, (
                "Your query was blocked because it contains patterns that look like a SQL injection attempt. "
                "If this was unintentional, try rephrasing as a natural language question."
            )
    return True, ""


# ---------------------------------------------------------------------------
# 2. Prompt injection / jailbreak detection
# ---------------------------------------------------------------------------

_INJECTION_PHRASES = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?above\s+instructions",
    r"disregard\s+(all\s+)?prior\s+instructions",
    r"forget\s+(everything|all)\s+(you|that)",
    r"you\s+are\s+now\s+(a|an|the)\s+",
    r"act\s+as\s+(a|an)\s+",
    r"pretend\s+(to\s+be|you\s+are)",
    r"system\s*:\s*",                        # raw system prompt prefix
    r"<\|?(system|assistant|user)\|?>",      # chat template markers
    r"override\s+(safety|guardrail|restriction|filter)",
    r"bypass\s+(safety|guardrail|restriction|filter)",
    r"reveal\s+(your|the)\s+(system|initial|original)\s+(prompt|instruction)",
    r"what\s+(is|are)\s+your\s+(system|initial)\s+(prompt|instruction)",
    r"repeat\s+(the|your)\s+(system|initial)\s+(prompt|instruction)",
    r"do\s+anything\s+now",                  # "DAN" jailbreak
    r"jailbreak",
]

def check_prompt_injection(query: str) -> tuple[bool, str]:
    """Detect attempts to override the agent's instructions."""
    lower = query.lower()
    for pattern in _INJECTION_PHRASES:
        if re.search(pattern, lower):
            logger.warning(f"Prompt injection blocked: matched '{pattern}' in '{query[:80]}'")
            return False, (
                "Your query appears to contain instructions aimed at overriding the system's behavior. "
                "I can only answer questions about financial markets and company data. "
                "Please rephrase as a genuine financial question."
            )
    return True, ""


# ---------------------------------------------------------------------------
# 3. PII detection
# ---------------------------------------------------------------------------

_PII_PATTERNS = [
    (r"\b\d{3}-\d{2}-\d{4}\b", "Social Security Number"),
    (r"\b\d{9}\b", "potential SSN"),
    (r"\b(?:\d{4}[-\s]?){3}\d{4}\b", "credit/debit card number"),
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email address"),
    (r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "phone number"),
]

def check_pii(query: str) -> tuple[bool, str]:
    """Block queries containing personally identifiable information."""
    for pattern, pii_type in _PII_PATTERNS:
        if re.search(pattern, query):
            logger.warning(f"PII detected ({pii_type}) in query: '{query[:60]}'")
            return False, (
                f"Your query appears to contain a {pii_type}. "
                f"For your safety, please don't include personal information in your questions. "
                f"I only need financial topics — no personal data required."
            )
    return True, ""


# ---------------------------------------------------------------------------
# 4. Topic / scope guardrail
# ---------------------------------------------------------------------------

_OFFTOPIC_PATTERNS = [
    r"\b(recipe|cooking|bake|baking)\b",
    r"\b(weather|forecast|temperature)\b.*\b(today|tomorrow|this week)\b",
    r"\b(write|generate|create)\s+(me\s+)?(a\s+)?(poem|story|essay|song|code|script)\b",
    r"\b(how\s+to\s+)?(hack|exploit|crack|phish)\b",
    r"\b(medical|diagnosis|symptom|disease|treatment)\b",
    r"\b(legal\s+advice|lawsuit|sue|attorney)\b",
    r"\b(dating|relationship|breakup)\b",
]

def check_topic_relevance(query: str) -> tuple[bool, str]:
    """Reject queries that are clearly outside financial domain."""
    lower = query.lower()
    for pattern in _OFFTOPIC_PATTERNS:
        if re.search(pattern, lower):
            logger.info(f"Off-topic query blocked: '{query[:80]}'")
            return False, (
                "That question falls outside my area of expertise. "
                "I'm a financial intelligence agent — I can help with company financials, "
                "market trends, analyst opinions, sector comparisons, and similar topics. "
                "Try asking something about stocks, revenue, or market sentiment."
            )
    return True, ""


# ---------------------------------------------------------------------------
# 5. Input length / sanity checks
# ---------------------------------------------------------------------------

MAX_QUERY_LENGTH = 2000

def check_input_length(query: str) -> tuple[bool, str]:
    """Prevent absurdly long inputs that could be prompt stuffing."""
    if len(query) > MAX_QUERY_LENGTH:
        return False, (
            f"Your query is {len(query)} characters — the maximum is {MAX_QUERY_LENGTH}. "
            f"Please shorten your question."
        )
    if len(query.strip()) < 3:
        return False, "Please ask a more specific question."
    return True, ""


# ---------------------------------------------------------------------------
# 6. Response validation (runs AFTER the agent)
# ---------------------------------------------------------------------------

_RESPONSE_REDFLAGS = [
    (r"(?i)(ignore|disregard)\s+(previous|all)\s+instructions", "leaked system prompt behavior"),
    (r"(?i)as\s+an?\s+AI\s+(language\s+)?model", "generic AI disclaimer leak"),
    (r"(?i)I('m|\s+am)\s+sorry.{0,20}I\s+can('t|not)", "refusal pattern in financial context"),
]

def validate_response(response: str) -> tuple[bool, str]:
    """Check the agent's output for signs of prompt leaking or weirdness."""
    for pattern, issue in _RESPONSE_REDFLAGS:
        if re.search(pattern, response):
            logger.warning(f"Response guardrail triggered: {issue}")
            return False, issue
    return True, ""


# ---------------------------------------------------------------------------
# Combined pre-query validation
# ---------------------------------------------------------------------------

# order matters — cheapest checks first
_INPUT_CHECKS = [
    ("input_length", check_input_length),
    ("pii_detection", check_pii),
    ("sql_injection", check_sql_injection),
    ("prompt_injection", check_prompt_injection),
    ("topic_relevance", check_topic_relevance),
]


def validate_input(query: str) -> dict:
    """
    Run all input guardrails. Returns a dict with:
      - passed: bool — True if all checks pass
      - blocked_by: str — name of the guardrail that blocked (if any)
      - message: str — user-friendly explanation (if blocked)
      - checks_run: list — all guardrails that were evaluated
    """
    checks_run = []
    for name, check_fn in _INPUT_CHECKS:
        passed, message = check_fn(query)
        checks_run.append({"name": name, "passed": passed})
        if not passed:
            logger.info(f"Query blocked by {name}: {query[:80]}")
            return {
                "passed": False,
                "blocked_by": name,
                "message": message,
                "checks_run": checks_run,
            }

    return {
        "passed": True,
        "blocked_by": None,
        "message": "",
        "checks_run": checks_run,
    }
