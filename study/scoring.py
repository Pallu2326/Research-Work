from dataclasses import dataclass
import re


STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "for", "by", "with", "on", "at", "is", "are",
    "be", "this", "that", "it", "as", "from", "not", "shall", "will", "may", "must", "party", "parties",
}


@dataclass
class Score:
    unsupported_claim_rate: float
    expected_coverage: float
    risk_flag: str


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _keywords(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", _normalize(text))
    return {tok for tok in tokens if tok not in STOPWORDS and len(tok) > 2}


def _split_clauses(text: str) -> list[str]:
    pieces = re.split(r"[\.\n;:]+", text)
    return [part.strip() for part in pieces if part.strip()]


def _claim_supported(claim: str, source_clauses: list[str]) -> bool:
    claim_keys = _keywords(claim)
    if not claim_keys:
        return True

    best_overlap = 0.0
    for source in source_clauses:
        source_keys = _keywords(source)
        if not source_keys:
            continue
        overlap = len(claim_keys & source_keys) / len(claim_keys)
        if overlap > best_overlap:
            best_overlap = overlap

    # 0.45 works as a pragmatic threshold for paraphrases in short legal clauses
    return best_overlap >= 0.45


def compute_score(output_text: str, expected_points: list[str], source_text: str) -> Score:
    output_lower = _normalize(output_text)

    covered = sum(1 for point in expected_points if _normalize(point) in output_lower)
    coverage = covered / max(len(expected_points), 1)

    source_clauses = _split_clauses(source_text)
    claims = _split_clauses(output_text)

    unsupported = sum(1 for claim in claims if not _claim_supported(claim=claim, source_clauses=source_clauses))
    unsupported_rate = unsupported / max(len(claims), 1)

    if unsupported_rate > 0.55:
        risk = "high"
    elif unsupported_rate > 0.30:
        risk = "medium"
    else:
        risk = "low"

    return Score(unsupported_claim_rate=unsupported_rate, expected_coverage=coverage, risk_flag=risk)
