from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class EvalSample:
    sample_id: str
    task: str
    jurisdiction: str
    prompt: str
    source_text: str
    expected_points: list[str]


def load_dataset(path: Path) -> list[EvalSample]:
    samples: list[EvalSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            samples.append(
                EvalSample(
                    sample_id=row["sample_id"],
                    task=row["task"],
                    jurisdiction=row["jurisdiction"],
                    prompt=row["prompt"],
                    source_text=row["source_text"],
                    expected_points=row["expected_points"],
                )
            )
    return samples
