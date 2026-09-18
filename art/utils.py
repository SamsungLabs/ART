import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


def load_json_if_exists(path: Path) -> Dict[str, str]:
    """
    Load a JSON dict from disk if it exists; otherwise return an empty dict.

    Raises:
        RuntimeError: if the file exists and is non-empty but cannot be parsed as valid JSON.
    """
    if not path.exists():
        return {}
    if path.stat().st_size == 0:
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return {}
            return json.loads(content)
    except json.JSONDecodeError:
        raise RuntimeError(f"File {path} exists but is not valid JSON.")
    except Exception as e:
        raise RuntimeError(f"Failed to read JSON from {path}: {e}") from e


def save_json_atomic(path: Path, data: dict[str, object]) -> None:
    """Atomically save a dict as JSON to `path` (write to .tmp then replace)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    tmp.replace(path)


def ensure_output_dir(path: Path) -> None:
    """Create parent directories for `path` if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class RunConfig:
    """Configuration for dataset processing with an OpenAI-compatible model."""
    model_name: str
    output_path: Path = Path("./outputs.json")
    overwrite: bool = False
    save_every: int = 100
    base_url: str = "http://localhost:8000/v1"
    api_key: str = os.environ.get("OPENAI_API_KEY", "EMPTY")
    max_completion_tokens: int = 512
    descriptive: bool = False
    log_level: str = "INFO"
    dataset_dir: Path = Path(__file__).resolve().parents[1] / "dataset"
