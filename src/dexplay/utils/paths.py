from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNS_ROOT = REPO_ROOT / "runs"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_dir(run_name: str) -> Path:
    return ensure_dir(RUNS_ROOT / run_name)


def checkpoint_dir(run_name: str) -> Path:
    return ensure_dir(run_dir(run_name) / "checkpoints")


def train_log_path(run_name: str) -> Path:
    return run_dir(run_name) / "train.jsonl"


def eval_log_path(run_name: str) -> Path:
    return run_dir(run_name) / "eval.jsonl"


def eval_report_path(run_name: str) -> Path:
    return run_dir(run_name) / "eval_report.md"
