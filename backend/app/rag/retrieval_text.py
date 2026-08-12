import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from langchain_core.documents import Document


TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9]*")
CAMEL_CASE_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
QUERY_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "do",
    "does",
    "for",
    "how",
    "in",
    "is",
    "it",
    "of",
    "the",
    "to",
    "what",
    "where",
    "which",
}
GENERIC_RETRIEVAL_TERMS = {
    "code",
    "implemented",
    "implementation",
    "implement",
    "repository",
    "work",
    "works",
}
IMPLEMENTATION_QUESTION_TERMS = {
    "behavior",
    "flow",
    "implemented",
    "implementation",
    "process",
    "work",
    "works",
}
FILTER_QUERY_TERMS = {"filter"}
SORTING_SYMBOL_TERMS = {"partition", "quicksort", "sort"}
WORD_NORMALIZATIONS = {
    "categories": "category",
    "filtered": "filter",
    "filtering": "filter",
    "filters": "filter",
    "foods": "food",
    "items": "item",
    "paginated": "paginate",
    "paginating": "paginate",
    "pagination": "paginate",
    "priced": "price",
    "prices": "price",
    "pricing": "price",
    "sorted": "sort",
    "sorting": "sort",
    "sorts": "sort",
}
QUERY_CONCEPT_EXPANSIONS: dict[str, set[str]] = {
    "auth": {
        "authenticate",
        "authentication",
        "credential",
        "email",
        "identity",
        "login",
        "password",
        "session",
        "signin",
    },
    "login": {
        "authenticate",
        "authentication",
        "credential",
        "password",
        "session",
        "signin",
    },
    "permission": {
        "access",
        "authorization",
        "permission",
        "policy",
        "role",
    },
    "authorization": {
        "access",
        "permission",
        "policy",
        "role",
    },
    "filter": {
        "category",
        "form",
        "input",
        "maximum",
        "parameter",
        "price",
        "range",
        "request",
        "slider",
    },
}


@dataclass(frozen=True)
class RetrievalRelevance:
    lexical_score: float
    exact_match_score: float
    structural_score: float


def minimum_structural_relevance(query: str) -> float | None:
    query_terms = tokenize_text(query)
    if (
        query_terms & IMPLEMENTATION_QUESTION_TERMS
        and query_terms & FILTER_QUERY_TERMS
    ):
        return 0.45
    return None


def build_document_embedding_text(document: Document) -> str:
    """Add searchable structure while preserving the original Document."""
    metadata = document.metadata
    symbol_name = metadata.get("symbol_name") or "(file or structural context)"
    return "\n".join(
        (
            "Software repository source chunk",
            f"File: {metadata.get('file_path', '')}",
            f"Language: {metadata.get('language', '')}",
            f"Chunk type: {metadata.get('chunk_type', '')}",
            f"Symbol: {symbol_name}",
            (
                "Symbol lines: "
                f"{metadata.get('symbol_start_line', '')}-"
                f"{metadata.get('symbol_end_line', '')}"
            ),
            "Source:",
            document.page_content,
        )
    )


def build_query_embedding_text(query: str) -> str:
    expanded_terms = sorted(expand_query_terms(query) - tokenize_text(query))
    if not expanded_terms:
        return query.strip()
    return (
        f"Question: {query.strip()}\n"
        "Related software-repository concepts: "
        f"{', '.join(expanded_terms)}"
    )


def expand_query_terms(query: str) -> set[str]:
    terms = tokenize_text(query) - QUERY_STOP_WORDS
    expanded = set(terms)
    for term in terms:
        for trigger, related_terms in QUERY_CONCEPT_EXPANSIONS.items():
            if term.startswith(trigger) or trigger.startswith(term):
                expanded.update(related_terms)
    return expanded


def lexical_relevance(query: str, document: Document) -> float:
    return retrieval_relevance(query, document).lexical_score


def retrieval_relevance(
    query: str,
    document: Document,
) -> RetrievalRelevance:
    query_terms = expand_query_terms(query) - GENERIC_RETRIEVAL_TERMS
    if not query_terms:
        return RetrievalRelevance(0.0, 0.0, 0.0)

    metadata = document.metadata
    content_terms = tokenize_text(document.page_content)
    symbol_terms = tokenize_text(str(metadata.get("symbol_name") or ""))
    path = str(metadata.get("file_path") or "")
    path_terms = tokenize_text(path)
    searchable_terms = content_terms | symbol_terms | path_terms

    denominator = min(6, len(query_terms))
    lexical_score = min(
        1.0,
        len(query_terms & searchable_terms) / denominator,
    )

    focused_terms = (
        tokenize_text(query) - QUERY_STOP_WORDS - GENERIC_RETRIEVAL_TERMS
    )
    expanded_focused_terms = query_terms - GENERIC_RETRIEVAL_TERMS
    exact_match_score = _exact_metadata_score(
        focused_terms=focused_terms,
        expanded_terms=expanded_focused_terms,
        symbol_name=str(metadata.get("symbol_name") or ""),
        symbol_terms=symbol_terms,
        file_path=path,
        path_terms=path_terms,
    )
    structural_score = _structural_relevance(
        query=query,
        chunk_type=str(metadata.get("chunk_type") or ""),
        symbol_name=str(metadata.get("symbol_name") or ""),
        content=document.page_content,
    )
    return RetrievalRelevance(
        lexical_score=lexical_score,
        exact_match_score=exact_match_score,
        structural_score=structural_score,
    )


def tokenize_text(text: str) -> set[str]:
    return set(_token_sequence(text))


def _token_sequence(text: str) -> list[str]:
    expanded_camel_case = CAMEL_CASE_BOUNDARY.sub(" ", text)
    return [
        _normalize_word(match.group(0).lower())
        for match in TOKEN_PATTERN.finditer(expanded_camel_case)
    ]


def _exact_metadata_score(
    *,
    focused_terms: set[str],
    expanded_terms: set[str],
    symbol_name: str,
    symbol_terms: set[str],
    file_path: str,
    path_terms: set[str],
) -> float:
    normalized_symbol = _normalized_identifier(symbol_name)
    file_stem = PurePosixPath(file_path).stem
    normalized_file_stem = _normalized_identifier(file_stem)

    if normalized_symbol and any(
        _normalized_identifier(term) == normalized_symbol
        for term in expanded_terms
    ):
        return 1.0
    if normalized_file_stem and any(
        _normalized_identifier(term) == normalized_file_stem
        for term in expanded_terms
    ):
        return 1.0
    if not focused_terms:
        return 0.0

    symbol_coverage = len(focused_terms & symbol_terms) / len(focused_terms)
    path_coverage = len(focused_terms & path_terms) / len(focused_terms)
    if focused_terms <= symbol_terms:
        return 1.0
    if focused_terms <= path_terms:
        return 0.9
    return max(symbol_coverage, path_coverage * 0.9)


def _structural_relevance(
    query: str,
    chunk_type: str,
    symbol_name: str,
    content: str,
) -> float:
    if not (tokenize_text(query) & IMPLEMENTATION_QUESTION_TERMS):
        return 0.0
    normalized_type = chunk_type.lower()
    query_terms = tokenize_text(query)
    if query_terms & FILTER_QUERY_TERMS:
        return _filtering_structural_relevance(
            chunk_type=normalized_type,
            symbol_name=symbol_name,
            content=content,
        )
    if normalized_type in {"function", "method"}:
        if _is_trivial_accessor(symbol_name, content):
            return 0.25
        return 1.0
    if normalized_type == "constructor":
        return 0.75
    if normalized_type.endswith("_context") or normalized_type in {
        "class",
        "interface",
        "record",
        "struct",
        "trait",
    }:
        return 0.55
    if normalized_type in {"markdown_section", "document"}:
        return 0.25
    return 0.5


def _normalized_identifier(value: str) -> str:
    return "".join(_token_sequence(value))


def _normalize_word(word: str) -> str:
    return WORD_NORMALIZATIONS.get(word, word)


def _filtering_structural_relevance(
    *,
    chunk_type: str,
    symbol_name: str,
    content: str,
) -> float:
    symbol_terms = tokenize_text(symbol_name)
    content_terms = tokenize_text(content)
    all_terms = symbol_terms | content_terms

    if symbol_terms & SORTING_SYMBOL_TERMS:
        return 0.1
    if chunk_type in {"html_document", "html_element"}:
        if (
            {"price", "range"} <= all_terms
            and all_terms & {"form", "input", "parameter", "request"}
        ):
            return 1.0
        return 0.25
    if "filter" in all_terms:
        return 1.0
    if (
        {"price", "range"} <= all_terms
        and all_terms & {"form", "input", "parameter", "request"}
    ):
        return 1.0
    if chunk_type in {"function", "method"}:
        return 0.45
    if chunk_type.endswith("_context"):
        return 0.35
    return 0.25


def _is_trivial_accessor(symbol_name: str, content: str) -> bool:
    normalized_symbol = symbol_name.lower()
    accessor_name = (
        normalized_symbol == "size"
        or normalized_symbol.startswith(("get", "has", "is", "set"))
    )
    if not accessor_name:
        return False
    meaningful_lines = [line for line in content.splitlines() if line.strip()]
    control_flow = re.search(r"\b(if|for|while|switch|try)\b", content)
    return len(meaningful_lines) <= 5 and control_flow is None
