from dataclasses import dataclass
import os


@dataclass
class Settings:
    dataset_path: str = "data/legal_contract_eval.jsonl"
    output_path: str = "results/eval_results.csv"
    temperature: float = 0.0
    max_tokens: int = 800



def load_settings() -> Settings:
    dataset_path = os.getenv("EVAL_DATASET_PATH", "data/legal_contract_eval.jsonl")
    output_path = os.getenv("EVAL_OUTPUT_PATH", "results/eval_results.csv")
    temperature = float(os.getenv("EVAL_TEMPERATURE", "0.0"))
    max_tokens = int(os.getenv("EVAL_MAX_TOKENS", "800"))
    return Settings(
        dataset_path=dataset_path,
        output_path=output_path,
        temperature=temperature,
        max_tokens=max_tokens,
    )
