from collections import defaultdict
import csv
from pathlib import Path
import os

from study.config import Settings
from study.dataset import EvalSample
from study.providers import GeminiProvider, GroqProvider, OpenAIProvider, Provider
from study.scoring import compute_score


def build_prompt(sample: EvalSample, with_context: bool) -> str:
    if with_context:
        return (
            f"Task: {sample.task}\n"
            f"Jurisdiction: {sample.jurisdiction}\n"
            f"Source contract text:\n{sample.source_text}\n\n"
            f"Instruction:\n{sample.prompt}\n"
            "Return a concise legal drafting answer and avoid unsupported claims."
        )
    return f"Jurisdiction: {sample.jurisdiction}\nInstruction: {sample.prompt}"


def select_providers() -> dict[str, Provider]:
    configured = os.getenv("EVAL_PROVIDERS", "gemini,groq").split(",")
    selected = {name.strip().lower() for name in configured if name.strip()}

    all_providers: dict[str, Provider] = {
        "openai": OpenAIProvider("gpt-4.1-mini"),
        "gemini": GeminiProvider("gemini-2.5-flash"),
        "groq": GroqProvider("llama-3.3-70b-versatile"),
    }

    active: dict[str, Provider] = {}
    for provider_name in selected:
        provider = all_providers.get(provider_name)
        if provider and provider.is_available():
            active[provider_name] = provider

    return active


def _write_csv(rows: list[dict], output_path: Path) -> None:
    fieldnames = [
        "sample_id",
        "provider",
        "mode",
        "unsupported_claim_rate",
        "expected_coverage",
        "risk_flag",
        "output",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows: list[dict]) -> None:
    grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: {"count": 0.0, "ucr": 0.0, "cov": 0.0})
    for row in rows:
        key = (row["provider"], row["mode"])
        grouped[key]["count"] += 1
        grouped[key]["ucr"] += float(row["unsupported_claim_rate"])
        grouped[key]["cov"] += float(row["expected_coverage"])

    print("provider | mode | unsupported_claim_rate | expected_coverage")
    print("---|---|---:|---:")
    for provider, mode in sorted(grouped):
        total = grouped[(provider, mode)]["count"] or 1
        avg_ucr = grouped[(provider, mode)]["ucr"] / total
        avg_cov = grouped[(provider, mode)]["cov"] / total
        print(f"{provider} | {mode} | {avg_ucr:.4f} | {avg_cov:.4f}")


def run_evaluation(settings: Settings, dataset: list[EvalSample]) -> None:
    providers = select_providers()
    if not providers:
        raise RuntimeError(
            "No providers available. Set GEMINI_API_KEY and/or GROQ_API_KEY, "
            "or configure EVAL_PROVIDERS to providers with valid keys."
        )

    rows: list[dict] = []
    for sample in dataset:
        for provider_name, provider in providers.items():
            for mode in ("naive", "grounded"):
                prompt = build_prompt(sample=sample, with_context=(mode == "grounded"))
                output_text = provider.generate(
                    prompt=prompt,
                    max_tokens=settings.max_tokens,
                    temperature=settings.temperature,
                )
                score = compute_score(
                    output_text=output_text,
                    expected_points=sample.expected_points,
                    source_text=sample.source_text,
                )
                rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "provider": provider_name,
                        "mode": mode,
                        "unsupported_claim_rate": score.unsupported_claim_rate,
                        "expected_coverage": score.expected_coverage,
                        "risk_flag": score.risk_flag,
                        "output": output_text,
                    }
                )

    output_path = Path(settings.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(rows=rows, output_path=output_path)
    _print_summary(rows=rows)
