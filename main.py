from pathlib import Path

from study.config import load_settings
from study.dataset import load_dataset
from study.evaluate import run_evaluation


def main() -> None:
    settings = load_settings()
    dataset = load_dataset(Path(settings.dataset_path))
    run_evaluation(settings=settings, dataset=dataset)


if __name__ == "__main__":
    main()
