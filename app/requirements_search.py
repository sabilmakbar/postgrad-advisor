"""Runtime requirements search (story 005, AD-3): Tavily primary, DuckDuckGo
(`ddgs`) fallback on Tavily failure or missing key (AC 2). The authorized search
budget is a counter in graph state, enforced in code by the fit node — not
here and not in any prompt.
"""

import logging
from dataclasses import dataclass
from time import monotonic

from ddgs import DDGS
from langchain_tavily import TavilySearch

from app import tracing
from app.config import settings

logger = logging.getLogger(__name__)

MAX_RESULTS = 5


@dataclass
class SearchResult:
    query: str
    text: str  # snippets, each prefixed with its source URL; "" = nothing found
    sources: list[str]  # result URLs


def search_requirements(query: str) -> SearchResult:
    """One web search for admission requirements: text + source URLs. Empty
    text means nothing was found — the caller's `unknown` path, never an
    exception the graph has to survive."""
    search_operation_id = tracing.operation_id()
    if settings.tavily_api_key:
        started = monotonic()
        results = None
        try:
            results = TavilySearch(
                tavily_api_key=settings.tavily_api_key, max_results=MAX_RESULTS
            ).invoke({"query": query})["results"]
            if results:
                tracing.write(
                    {
                        "event": "requirements_search",
                        "search_operation_id": search_operation_id,
                        "backend": "tavily",
                        "duration_ms": tracing.elapsed(started),
                        "result_count": len(results),
                        "outcome": "success",
                        "cause": None,
                    }
                )
                return SearchResult(
                    query=query,
                    text="\n\n".join(f"[{r['url']}] {r['content']}" for r in results),
                    sources=[r["url"] for r in results],
                )
        except Exception:
            logger.warning(
                "Tavily search failed; falling back to DuckDuckGo",
            )
        tracing.write(
            {
                "event": "requirements_search",
                "search_operation_id": search_operation_id,
                "backend": "tavily",
                "duration_ms": tracing.elapsed(started),
                "result_count": len(results or ()),
                "outcome": "empty" if results is not None else "error",
                "cause": None,
            }
        )
    started = monotonic()
    try:
        hits = DDGS().text(query, max_results=MAX_RESULTS)
    except Exception:
        logger.warning("DuckDuckGo search failed")
        hits = []
        outcome = "error"
    else:
        outcome = "success" if hits else "empty"
    tracing.write(
        {
            "event": "requirements_search",
            "search_operation_id": search_operation_id,
            "backend": "duckduckgo",
            "duration_ms": tracing.elapsed(started),
            "result_count": len(hits),
            "outcome": outcome,
            "cause": "fallback" if settings.tavily_api_key else None,
        }
    )
    return SearchResult(
        query=query,
        text="\n\n".join(f"[{h['href']}] {h['body']}" for h in hits),
        sources=[h["href"] for h in hits],
    )
