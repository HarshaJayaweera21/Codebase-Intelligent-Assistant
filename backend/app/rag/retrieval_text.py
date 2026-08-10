import re

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
}


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
    query_terms = expand_query_terms(query)
    if not query_terms:
        return 0.0
    document_terms = tokenize_text(build_document_embedding_text(document))
    matches = len(query_terms & document_terms)
    return min(1.0, matches / min(4, len(query_terms)))


def tokenize_text(text: str) -> set[str]:
    expanded_camel_case = CAMEL_CASE_BOUNDARY.sub(" ", text)
    return {match.group(0).lower() for match in TOKEN_PATTERN.finditer(expanded_camel_case)}
