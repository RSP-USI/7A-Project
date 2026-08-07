"""
MBA 606 - 7A Project
FastMCP server exposing a help desk ticket corpus for triage work.

Entrypoint for Prefect Horizon / FastMCP Cloud: server.py
The FastMCP object below is named `mcp`, so the inferred entrypoint works.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from fastmcp import FastMCP

mcp = FastMCP("rsp-usi-mba-606")

# --------------------------------------------------------------------------
# Data loading
#
# Loading happens at import time, so it must never raise. If a data file is
# missing the server still starts and reports an empty corpus, which keeps the
# platform's pre-flight import check passing.
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent

DATA_FILES = [
    "fixtures.json",
    "resolved_fixtures.json",
]


def _load_tickets() -> list[dict[str, Any]]:
    tickets: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for filename in DATA_FILES:
        path = REPO_ROOT / filename
        if not path.exists():
            continue
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict):
                continue
            ticket_id = record.get("id")
            if not ticket_id or ticket_id in seen_ids:
                continue
            seen_ids.add(ticket_id)
            record.setdefault("status", "open")
            tickets.append(record)

    return tickets


TICKETS: list[dict[str, Any]] = _load_tickets()
TICKETS_BY_ID: dict[str, dict[str, Any]] = {t["id"]: t for t in TICKETS}


# --------------------------------------------------------------------------
# Lightweight text index, built once at import time. Pure standard library so
# the deployment stays dependency free beyond FastMCP itself.
# --------------------------------------------------------------------------

STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "have", "has", "had", "not",
    "but", "was", "were", "are", "you", "your", "from", "when", "them", "they",
    "she", "her", "his", "him", "its", "it", "is", "to", "of", "in", "on",
    "at", "my", "me", "we", "our", "a", "an", "so", "do", "does", "did",
    "will", "can", "could", "would", "get", "got", "getting", "been", "being",
    "there", "here", "what", "which", "who", "how", "why", "all", "any",
    "some", "one", "two", "now", "then", "than", "very", "just", "also",
    "about", "after", "before", "again", "still", "keep", "keeps", "kept",
    "says", "said", "say", "nothing", "something", "anything", "everything",
    "every", "each", "back", "over", "out", "off", "up", "down", "into",
    "user", "ticket", "please", "thanks", "thank", "know", "think", "need",
    "needs", "want", "wants", "like", "time", "times", "day", "days", "week",
    "morning", "work", "working", "works", "worked", "confirmed", "verified",
}


def _stem(word: str) -> str:
    """Crude suffix stripping so printer, prints, and printing all collapse to
    the same token. Not linguistically correct, but good enough for matching."""
    for suffix in ("ing", "ers", "er", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def _tokenize(text: str) -> set[str]:
    words = "".join(c if c.isalnum() else " " for c in str(text).lower()).split()
    return {
        _stem(w)
        for w in words
        if len(w) > 2 and w not in STOPWORDS and not w.isdigit()
    }


def _build_index() -> tuple[dict[str, set[str]], dict[str, float]]:
    import math

    ticket_tokens: dict[str, set[str]] = {}
    for ticket in TICKETS:
        combined = " ".join(
            str(ticket.get(field, ""))
            for field in ("body", "category", "resolution_summary")
        )
        ticket_tokens[ticket["id"]] = _tokenize(combined)

    document_count = max(len(ticket_tokens), 1)
    frequency: Counter = Counter()
    for tokens in ticket_tokens.values():
        frequency.update(tokens)

    idf = {
        term: math.log(document_count / count)
        for term, count in frequency.items()
    }
    return ticket_tokens, idf


_TICKET_TOKENS, _IDF = _build_index()


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------


@mcp.tool
def corpus_summary() -> dict[str, Any]:
    """Report how many tickets are loaded and how they break down by band,
    status, and category. Useful as a first call to confirm the data loaded."""
    return {
        "total_tickets": len(TICKETS),
        "by_band": dict(Counter(t.get("band", "unspecified") for t in TICKETS)),
        "by_status": dict(Counter(t.get("status", "unspecified") for t in TICKETS)),
        "by_category": dict(Counter(t.get("category", "unspecified") for t in TICKETS)),
        "data_files_found": [f for f in DATA_FILES if (REPO_ROOT / f).exists()],
    }


@mcp.tool
def list_tickets(
    band: Optional[str] = None,
    status: Optional[str] = None,
    category: Optional[str] = None,
    limit: int = 25,
) -> list[dict[str, Any]]:
    """List tickets, optionally filtered.

    Args:
        band: Filter by band, e.g. clean, ambiguous, adversarial.
        status: Filter by status, e.g. open or resolved.
        category: Filter by category, e.g. printing or account_password.
        limit: Maximum number of tickets to return.
    """
    results = TICKETS

    if band:
        results = [t for t in results if t.get("band", "").lower() == band.lower()]
    if status:
        results = [t for t in results if t.get("status", "").lower() == status.lower()]
    if category:
        results = [
            t for t in results if t.get("category", "").lower() == category.lower()
        ]

    limit = max(1, min(limit, 100))
    return [
        {
            "id": t["id"],
            "band": t.get("band"),
            "category": t.get("category"),
            "status": t.get("status"),
            "body": t.get("body", "")[:160],
        }
        for t in results[:limit]
    ]


@mcp.tool
def get_ticket(ticket_id: str) -> dict[str, Any]:
    """Retrieve the full record for a single ticket by its id, for example F01
    or R042. Includes the resolution fields when the ticket is resolved."""
    ticket = TICKETS_BY_ID.get(ticket_id.strip().upper())
    if ticket is None:
        return {
            "error": f"No ticket found with id {ticket_id}.",
            "available_ids_sample": [t["id"] for t in TICKETS[:10]],
        }
    return ticket


@mcp.tool
def search_tickets(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Find tickets whose text mentions the given words. Matching is on the
    ticket body, category, and resolution summary.

    Args:
        query: Words to look for, e.g. "printer offline" or "vpn".
        limit: Maximum number of matches to return.
    """
    terms = [w for w in query.lower().split() if w]
    if not terms:
        return []

    scored = []
    for ticket in TICKETS:
        haystack = " ".join(
            str(ticket.get(field, ""))
            for field in ("body", "category", "resolution_summary")
        ).lower()
        score = sum(1 for term in terms if term in haystack)
        if score:
            scored.append((score, ticket))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    limit = max(1, min(limit, 50))

    return [
        {
            "id": t["id"],
            "band": t.get("band"),
            "category": t.get("category"),
            "matched_terms": score,
            "body": t.get("body", "")[:200],
        }
        for score, t in scored[:limit]
    ]


@mcp.tool
def get_resolution(ticket_id: str) -> dict[str, Any]:
    """Return only the resolution detail for a ticket: what was done, the steps
    taken, who handled it, and how long it took."""
    ticket = TICKETS_BY_ID.get(ticket_id.strip().upper())
    if ticket is None:
        return {"error": f"No ticket found with id {ticket_id}."}
    if ticket.get("status") != "resolved":
        return {
            "id": ticket["id"],
            "status": ticket.get("status", "open"),
            "note": "This ticket has no recorded resolution.",
        }
    return {
        "id": ticket["id"],
        "category": ticket.get("category"),
        "resolution_summary": ticket.get("resolution_summary"),
        "resolution_steps": ticket.get("resolution_steps", []),
        "resolved_by": ticket.get("resolved_by"),
        "time_to_resolve_minutes": ticket.get("time_to_resolve_minutes"),
    }


@mcp.tool
def find_similar_resolved(body: str, limit: int = 3) -> list[dict[str, Any]]:
    """Given the text of a new ticket, return the most similar resolved tickets
    and how they were fixed. Intended as a triage aid.

    Args:
        body: The text of the incoming ticket.
        limit: How many similar resolved tickets to return.
    """
    terms = _tokenize(body)
    if not terms:
        return []

    scored = []
    for ticket in TICKETS:
        if ticket.get("status") != "resolved":
            continue
        tokens = _TICKET_TOKENS.get(ticket["id"], set())
        shared = terms & tokens
        if not shared:
            continue
        # Weight each shared word by how rare it is across the corpus, then
        # add a bonus for breadth so that agreement on several words outranks
        # a single unusual word appearing by coincidence.
        score = sum(_IDF.get(term, 0.0) for term in shared)
        score *= 1 + 0.25 * (len(shared) - 1)
        scored.append((score, shared, ticket))

    scored.sort(key=lambda triple: triple[0], reverse=True)
    limit = max(1, min(limit, 10))

    return [
        {
            "id": t["id"],
            "similarity_score": round(score, 2),
            "matched_terms": sorted(shared)[:8],
            "category": t.get("category"),
            "band": t.get("band"),
            "original_ticket": t.get("body", "")[:200],
            "resolution_summary": t.get("resolution_summary"),
            "resolved_by": t.get("resolved_by"),
        }
        for score, shared, t in scored[:limit]
    ]


@mcp.tool
def resolution_time_stats(band: Optional[str] = None) -> dict[str, Any]:
    """Summarize how long resolved tickets took, overall or within one band.

    Args:
        band: Optional band to restrict the statistics to.
    """
    pool = [t for t in TICKETS if t.get("status") == "resolved"]
    if band:
        pool = [t for t in pool if t.get("band", "").lower() == band.lower()]

    times = [
        t["time_to_resolve_minutes"]
        for t in pool
        if isinstance(t.get("time_to_resolve_minutes"), (int, float))
    ]
    if not times:
        return {"band": band or "all", "resolved_count": 0, "note": "No timing data."}

    times.sort()
    mid = len(times) // 2
    median = times[mid] if len(times) % 2 else (times[mid - 1] + times[mid]) / 2

    return {
        "band": band or "all",
        "resolved_count": len(times),
        "min_minutes": times[0],
        "median_minutes": median,
        "mean_minutes": round(sum(times) / len(times), 1),
        "max_minutes": times[-1],
    }


# --------------------------------------------------------------------------
# Local development only. The hosting platform imports `mcp` directly and does
# not execute this block, so nothing runs at import time.
# --------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
