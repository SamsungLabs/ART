import argparse
import json
import logging
import string
from pathlib import Path
from typing import Dict, Tuple

_PUNCT_XLAT = str.maketrans("", "", string.punctuation)


def normalize_text(text: str | None) -> str:
    """
    Lowercase, strip and remove punctuation from `text`. If `text` is None, returns an empty string.
    """
    if not text:
        return ""
    return text.translate(_PUNCT_XLAT).lower().strip()


def load_local_metadata(dataset_dir: Path) -> dict[str, list[dict]]:
    """Read task references without loading audio or inference dependencies."""
    metadata_paths = sorted(dataset_dir.glob("*/metadata.json"))
    if not metadata_paths:
        raise ValueError(f"No task metadata.json files found in {dataset_dir}")

    tasks = {}
    seen_ids = set()
    for path in metadata_paths:
        with path.open(encoding="utf-8") as f:
            records = json.load(f)
        if not isinstance(records, list) or not records:
            raise ValueError(f"Expected a non-empty list of records in {path}")
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"Expected metadata objects in {path}")
            iid = record.get("instance_id")
            answer = record.get("answer")
            if not isinstance(iid, str) or not iid.strip():
                raise ValueError(f"Invalid instance_id in {path}: {iid!r}")
            if iid in seen_ids:
                raise ValueError(f"Duplicate instance_id: {iid}")
            if not isinstance(answer, str) or normalize_text(answer) not in {"yes", "no"}:
                raise ValueError(f"Invalid reference answer for {iid} in {path}")
            seen_ids.add(iid)
        tasks[path.parent.name] = records
    return tasks


def load_outputs(path: Path) -> dict[str, str]:
    """Require an existing, non-empty JSON mapping of IDs to text predictions."""
    with path.open(encoding="utf-8") as f:
        outputs = json.load(f)
    if not isinstance(outputs, dict) or not outputs:
        raise ValueError("Outputs must be a non-empty JSON object")
    for iid, answer in outputs.items():
        if not iid.strip() or not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"Expected a non-empty instance ID and text prediction: {iid!r}")
    return outputs


def calculate_task_accuracy(name: str, ds: list[dict], outputs: Dict[str, str]) -> Tuple[int, int]:
    """
    Calculate (correct, total) for one task of the benchmark.

    Args:
        ds (list[dict]): Task metadata records.
        outputs (dict[str, str]): Mapping `instance_id` -> `model_output`.
    
    Returns:
        (correct, total) (int, int): Number of correct responses and total number of evaluated
            samples.
    """
    correct, total = 0, 0

    for x in ds:
        iid = x["instance_id"]
        reference = normalize_text(x["answer"])
        hypothesis = normalize_text(outputs.get(iid))

        correct += int(reference == hypothesis)
        total += 1

    return correct, total


def evaluate(ds: dict[str, list[dict]], outputs: dict[str, str]) -> Tuple[float, dict[str, float]]:
    """
    Evaluate all tasks in the benchmark and return per-task and overall accuracy.

    Args:
        ds (dict[str, list[dict]]): ART task metadata.
        outputs (dict[str, str]): Mapping `instance_id` -> `model_output`.
    
    Returns:
        (overall, per_task) (float, dict[str, float]): Overall and per-task accuracy.
    """
    per_task: Dict[str, float] = {}
    overall_correct, overall_total = 0, 0

    for task_name, samples in ds.items():
        covered = sum(x["instance_id"] in outputs for x in samples)
        logging.info("Coverage [%s]: %d/%d (missing=%d)",
                     task_name, covered, len(samples), len(samples) - covered)
        c, t = calculate_task_accuracy(task_name, samples, outputs)
        per_task[task_name] = (c / t) if t else 0.0
        overall_correct += c
        overall_total += t

    overall = (overall_correct / overall_total) if overall_total else 0.0

    return overall, per_task


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Evaluate ART predictions against local task metadata. Missing predictions "
                    "count as incorrect; answers use exact match after case/punctuation normalization.",
    )
    parser.add_argument("outputs_path", type=Path, help="JSON file mapping instance IDs to answers.")
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path(__file__).resolve().parent / "dataset",
        help="Local dataset directory containing task folders with metadata.json.",
    )
    args = parser.parse_args()
    logging.info("Using model outputs from file: %s", args.outputs_path.resolve())
    try:
        outputs = load_outputs(args.outputs_path)
        ds = load_local_metadata(args.dataset_dir)
        reference_ids = {x["instance_id"] for samples in ds.values() for x in samples}
        covered = len(reference_ids.intersection(outputs))
        if not covered:
            raise ValueError("No prediction IDs match the local dataset")
    except (OSError, ValueError) as e:
        parser.error(str(e))

    unknown = len(outputs.keys() - reference_ids)
    if unknown:
        logging.warning("Ignoring %d prediction IDs absent from the dataset", unknown)
    logging.info("Overall coverage: %d/%d (missing=%d)",
                 covered, len(reference_ids), len(reference_ids) - covered)
    if covered < len(reference_ids):
        logging.warning("Missing predictions count as incorrect in accuracy")
    
    overall, per_task = evaluate(ds, outputs)

    for name, acc in per_task.items():
        logging.info("Accuracy [%s]: %.4f", name, acc)
    print(f"overall_accuracy={overall:.4f}")


if __name__ == "__main__":
    main()
