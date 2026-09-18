import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict

import soundfile as sf
from tqdm import tqdm

from art import (
    RunConfig, 
    vLLMProcessor, 
    ensure_output_dir, 
    load_json_if_exists, 
    save_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on the ART benchmark using an OpenAI/vLLM model.",
    )
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path(__file__).resolve().parent / "dataset",
        help="Local dataset directory containing task folders with metadata.json and FLAC files.",
    )
    parser.add_argument(
        "--output-path", type=Path, default=Path("./outputs.json"),
        help="Path to where the model responses will be saved.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Ignore existing output file and overwrite it.",
    )
    parser.add_argument(
        "--save-every", type=int, default=100,
        help="Save results every N processed samples per task.",
    )
    parser.add_argument(
        "--model-name", type=str, required=True, help="Model name on the API.",
    )
    parser.add_argument(
        "--base-url", type=str, default="http://localhost:8000/v1",
        help="Base URL of OpenAI-compatbile API.",
    )
    parser.add_argument(
        "--api-key", type=str, default=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        help="API key (use 'EMPTY' for local vLLM)."
    )
    parser.add_argument(
        "--max-completion-tokens", type=int, default=512,
        help="Maximum number of tokens to generate per request.",
    )
    parser.add_argument(
        "--descriptive", action="store_true",
        help="Allow free-form answers instead of 'Yes/No' only.",
    )
    parser.add_argument(
        "--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO",
        help="Logging verbosity level.",
    )
    return parser.parse_args()


def parse_config() -> RunConfig:
    args = parse_args()

    return RunConfig(
        model_name=args.model_name,
        dataset_dir=args.dataset_dir,
        output_path=args.output_path,
        overwrite=args.overwrite,
        save_every=args.save_every,
        base_url=args.base_url,
        api_key=args.api_key,
        max_completion_tokens=args.max_completion_tokens,
        descriptive=args.descriptive,
        log_level=args.log_level,
    )


def load_local_tasks(dataset_dir: Path) -> dict[str, list[dict]]:
    """Validate local metadata and audio paths without decoding audio into memory."""
    metadata_paths = sorted(dataset_dir.glob("*/metadata.json"))
    if not metadata_paths:
        raise ValueError(f"No task metadata.json files found in {dataset_dir}")

    tasks = {}
    seen_ids = set()
    for metadata_path in metadata_paths:
        with metadata_path.open(encoding="utf-8") as f:
            records = json.load(f)
        if not isinstance(records, list) or not records:
            raise ValueError(f"Expected a non-empty list of records in {metadata_path}")
        samples = []
        for record in records:
            iid = record.get("instance_id") if isinstance(record, dict) else None
            if not isinstance(iid, str) or not iid or "/" in iid or "\\" in iid:
                raise ValueError(f"Invalid instance_id in {metadata_path}: {iid!r}")
            if iid in seen_ids:
                raise ValueError(f"Duplicate instance_id: {iid}")
            audio_path = metadata_path.parent / f"{iid}.flac"
            if not audio_path.is_file():
                raise FileNotFoundError(f"Missing audio for {iid}: {audio_path}")
            seen_ids.add(iid)
            samples.append({"instance_id": iid, "audio_path": audio_path})
        tasks[metadata_path.parent.name] = samples
    return tasks


def build_run_metadata(cfg: RunConfig, tasks: dict[str, list[dict]]) -> dict:
    """Identify the requested run without storing credentials."""
    digest = hashlib.sha256()
    for name, samples in sorted(tasks.items()):
        paths = [cfg.dataset_dir / name / "metadata.json"]
        paths.extend(x["audio_path"] for x in sorted(samples, key=lambda x: x["instance_id"]))
        for path in paths:
            file_digest = hashlib.sha256()
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    file_digest.update(chunk)
            digest.update(json.dumps([name, path.name, file_digest.hexdigest()]).encode("utf-8"))
    return {
        "schema_version": 1,
        "model_name": cfg.model_name,
        "base_url": cfg.base_url.rstrip("/"),
        "max_completion_tokens": cfg.max_completion_tokens,
        "descriptive": cfg.descriptive,
        "dataset_sha256": digest.hexdigest(),
    }


def prepare_outputs(cfg: RunConfig, metadata: dict, instance_ids: set[str]) -> dict[str, str]:
    """Only resume predictions with matching run metadata."""
    metadata_path = cfg.output_path.with_suffix(cfg.output_path.suffix + ".run.json")
    outputs = {} if cfg.overwrite else load_json_if_exists(cfg.output_path)
    if not isinstance(outputs, dict) or any(
        not isinstance(value, str) or not value.strip() for value in outputs.values()
    ):
        raise ValueError("Existing outputs must map instance IDs to non-empty strings")
    if not cfg.overwrite:
        if outputs and not metadata_path.is_file():
            raise ValueError(
                f"Cannot verify existing predictions: {metadata_path} is missing. "
                "Use a new --output-path or --overwrite to start fresh."
            )
        if metadata_path.exists():
            with metadata_path.open(encoding="utf-8") as f:
                previous = json.load(f)
            if previous != metadata:
                raise ValueError(
                    "Existing run metadata does not match the requested model, settings, or dataset. "
                    "Use a new --output-path or --overwrite to start fresh."
                )
        if outputs.keys() - instance_ids:
            raise ValueError("Existing outputs contain IDs absent from the dataset")

    ensure_output_dir(cfg.output_path)
    # Clear old predictions before replacing their metadata, so an interrupted
    # overwrite cannot label old predictions as belonging to the new run.
    save_json_atomic(cfg.output_path, outputs)
    save_json_atomic(metadata_path, metadata)
    logging.info("%s %d existing outputs.", "Discarded; starting with" if cfg.overwrite else "Loaded", len(outputs))
    return outputs


def process_task(
    name: str,
    ds: list[dict],
    proc: vLLMProcessor,
    outputs: Dict[str, str],
    save_path: Path,
    save_every: int,
) -> int:
    processed_since_save = 0
    failed = 0

    pbar = tqdm(ds, desc=f"{name}", dynamic_ncols=True)
    try:
        for x in pbar:
            iid = x.get("instance_id")
            if iid in outputs:
                continue

            try:
                waveform, sampling_rate = sf.read(x["audio_path"], dtype="float32")
                response = proc.infer_sample({
                    "instance_id": iid,
                    "audio": {"array": waveform, "sampling_rate": sampling_rate},
                })
                if not isinstance(response, str) or not response.strip():
                    raise ValueError("Model returned an empty or non-text response")
            except Exception as e:
                failed += 1
                logging.exception(f"Error processing {name} ({iid=}): {e}")
                continue

            outputs[iid] = response
            processed_since_save += 1

            if processed_since_save >= save_every:
                save_json_atomic(save_path, outputs)
                logging.info(f"Saved {save_path.name} ({processed_since_save} new items).")
                processed_since_save = 0
    except KeyboardInterrupt:
        logging.warning(f"Interrupted! Saving partial results to {save_path}")
        raise
    finally:
        save_json_atomic(save_path, outputs)
        logging.info(f"Saved {save_path.name}")
    logging.info("Task [%s]: %d/%d complete; %d failed this run", name,
                 sum(x["instance_id"] in outputs for x in ds), len(ds), failed)
    return failed


def main() -> int:
    cfg = parse_config()

    logging.basicConfig(level=getattr(logging, cfg.log_level), format="%(levelname)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore.connection").setLevel(logging.WARNING)
    logging.getLogger("httpcore.http11").setLevel(logging.WARNING)
    logging.getLogger("httpcore.connection_pool").setLevel(logging.WARNING)

    if cfg.save_every <= 0 or cfg.max_completion_tokens <= 0:
        raise ValueError("--save-every and --max-completion-tokens must be positive")
    ds = load_local_tasks(cfg.dataset_dir)
    instance_ids = {x["instance_id"] for samples in ds.values() for x in samples}
    logging.info("Fingerprinting local dataset for resume validation.")
    metadata = build_run_metadata(cfg, ds)
    outputs = prepare_outputs(cfg, metadata, instance_ids)

    processor = vLLMProcessor(
        model_name=cfg.model_name,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        max_completion_tokens=cfg.max_completion_tokens,
        descriptive=cfg.descriptive,
    )

    failed = 0
    for task_name, samples in ds.items():
        logging.info(f"Processing task: {task_name} (n_samples={len(samples)})")
        failed += process_task(
            name=task_name,
            ds=samples,
            proc=processor,
            outputs=outputs,
            save_path=cfg.output_path,
            save_every=cfg.save_every,
        )
    missing = len(instance_ids - outputs.keys())
    logging.info("Run summary: %d/%d complete; %d failed this run; %d missing",
                 len(instance_ids) - missing, len(instance_ids), failed, missing)
    if missing:
        logging.error("Inference incomplete. Successful predictions are saved; rerun with the "
                      "same settings to retry missing samples.")
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (OSError, ValueError, RuntimeError) as e:
        logging.error("%s", e)
        raise SystemExit(1)
