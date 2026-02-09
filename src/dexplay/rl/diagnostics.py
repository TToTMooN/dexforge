from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from statistics import mean

from dexplay.utils.logging import load_jsonl


def failure_histogram(records: Iterable[dict]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for rec in records:
        counts[str(rec.get("termination_reason", "UNKNOWN"))] += 1
    return dict(sorted(counts.items(), key=lambda kv: kv[0]))


def summary_stats(records: list[dict]) -> dict[str, float]:
    if not records:
        return {
            "num_episodes": 0,
            "success_rate": 0.0,
            "mean_episode_length": 0.0,
            "mean_episode_return": 0.0,
            "mean_final_orientation_error": 0.0,
            "mean_action_norm": 0.0,
        }

    n = len(records)
    success = sum(1 for r in records if r.get("termination_reason") == "SUCCESS")
    return {
        "num_episodes": float(n),
        "success_rate": float(success / n),
        "mean_episode_length": float(mean(float(r.get("episode_length", 0.0)) for r in records)),
        "mean_episode_return": float(mean(float(r.get("episode_return", 0.0)) for r in records)),
        "mean_final_orientation_error": float(
            mean(float(r.get("final_orientation_error", 0.0)) for r in records)
        ),
        "mean_action_norm": float(mean(float(r.get("mean_action_norm", 0.0)) for r in records)),
    }


def write_eval_report(
    report_path: Path,
    run_name: str,
    robot_name: str,
    checkpoint_path: str,
    seed: int,
    episodes: int,
    records: list[dict],
) -> None:
    summary = summary_stats(records)
    hist = failure_histogram(records)

    lines = [
        f"# Eval Report: {run_name}",
        "",
        "## Setup",
        f"- Robot: `{robot_name}`",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Seed: `{seed}`",
        f"- Requested episodes: `{episodes}`",
        f"- Completed episodes: `{int(summary['num_episodes'])}`",
        "",
        "## Metrics",
        f"- Success rate: `{summary['success_rate']:.3f}`",
        f"- Mean episode length: `{summary['mean_episode_length']:.2f}`",
        f"- Mean episode return: `{summary['mean_episode_return']:.3f}`",
        f"- Mean final orientation error (rad): `{summary['mean_final_orientation_error']:.3f}`",
        f"- Mean action norm: `{summary['mean_action_norm']:.3f}`",
        "",
        "## Failure Histogram",
    ]

    if hist:
        for key in sorted(hist):
            lines.append(f"- {key}: {hist[key]}")
    else:
        lines.append("- No completed episodes logged.")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_eval_records(eval_jsonl: Path) -> list[dict]:
    return load_jsonl(eval_jsonl)
