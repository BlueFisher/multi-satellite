"""Paired inference-only communication ablations for SharedMessageMATD3."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


CONDITIONS = (
    "full_communication",
    "sparse_communication",
    "no_communication",
    "shuffle_message",
)
METRICS = ("capture_success", "team_return", "episode_length")
BATCH_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def validate_batch_id(value: str) -> str:
    if (
        not isinstance(value, str)
        or not BATCH_ID_PATTERN.fullmatch(value)
        or value in {".", ".."}
        or ".." in value
    ):
        raise ValueError(
            "batch ID must use only letters, digits, dot, underscore, and "
            "hyphen, without path traversal"
        )
    return value


def resolve_batch_root(output_root: Path, batch_id: str) -> Path:
    return output_root / validate_batch_id(batch_id)


def validate_condition_metadata(actual: dict, expected: dict) -> None:
    if actual != expected:
        raise ValueError(
            f"condition metadata mismatch: expected={expected}, actual={actual}"
        )


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{remaining_seconds:02d}s"
    return f"{minutes}m{remaining_seconds:02d}s"


def format_startup(
    model_path: Path, output_dir: Path, episode_count: int, condition_count: int
) -> str:
    return (
        f"通信消融实验 | 模型={model_path} | 输出={output_dir} | "
        f"每条件回合={episode_count} | 条件={condition_count} | "
        f"总回合={episode_count * condition_count}"
    )


def format_cell_start(index: int, total: int, condition: str) -> str:
    return f"[{index}/{total}] 开始 | condition={condition}"


def format_cell_complete(
    index: int,
    total: int,
    episodes: int,
    cell_seconds: float,
    total_seconds: float,
    eta_seconds: float,
) -> str:
    return (
        f"[{index}/{total}] 完成 | episodes={episodes} | "
        f"本次={format_duration(cell_seconds)} | "
        f"累计={format_duration(total_seconds)} | ETA={format_duration(eta_seconds)}"
    )


@dataclass(frozen=True)
class ExperimentConfig:
    model_path: Path
    episodes: int = 40
    conditions: tuple[str, ...] = CONDITIONS
    max_episode_steps: int = 1000

    def validate(self) -> "ExperimentConfig":
        if not self.model_path.exists():
            raise FileNotFoundError(f"model path does not exist: {self.model_path}")
        if isinstance(self.episodes, bool) or self.episodes <= 0:
            raise ValueError("episodes must be a positive integer")
        invalid = sorted(set(self.conditions) - set(CONDITIONS))
        if invalid or not self.conditions:
            raise ValueError(f"invalid communication ablation conditions: {invalid}")
        return self


def build_inference_command(
    config: ExperimentConfig, condition: str, data_root: Path
) -> list[str]:
    config.validate()
    if condition not in config.conditions:
        raise ValueError(f"condition {condition!r} is not enabled")
    return [
        sys.executable,
        "src/main.py",
        "--just_eval", "True",
        "--load_model", "True",
        "--render", "False",
        "--env_name", "SatelliteEnv3v1Hvt",
        "--alg_name=SharedMessageMATD3",
        "--load_path", str(config.model_path),
        "--obs_add_distance", "False",
        "--on_plane", "False",
        "--limit_accel", "True",
        "--communication_ablation", condition,
        "--eval_episode", str(config.episodes),
        "--max_episode_steps", str(config.max_episode_steps),
        "--eval_data_dir", str(data_root),
        "--eval_run_name", condition,
    ]


def build_condition_paths(
    output_root: Path, condition: str
) -> tuple[Path, Path]:
    """Return the artifact search root and evaluator data root."""

    return output_root / condition, output_root


def stream_subprocess(
    command: Sequence[str],
    condition: str,
    log_handle,
    emit=None,
) -> int:
    """Stream merged child output to the terminal and the condition log."""

    if emit is None:
        emit = lambda message: print(message, flush=True)
    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if process.stdout is None:
        raise RuntimeError("child process stdout pipe was not created")
    for raw_line in process.stdout:
        log_handle.write(raw_line)
        log_handle.flush()
        emit(f"[{condition}] {raw_line.rstrip(chr(13) + chr(10))}")
    process.stdout.close()
    return process.wait()


def validate_episode_artifact_sequence(
    paths: Sequence[Path], episodes: int
) -> list[Path]:
    ordered = sorted(paths, key=lambda path: path.name)
    expected = [f"episode_{index:04d}.npz" for index in range(episodes)]
    actual = [path.name for path in ordered]
    if actual != expected:
        raise ValueError(
            f"expected exact episode artifacts {expected[0]}..{expected[-1]}, "
            f"got {actual}"
        )
    return ordered


def discover_episode_artifacts(root: Path, episodes: int) -> list[Path]:
    matches = list(root.rglob("episode_*.npz")) if root.exists() else []
    return validate_episode_artifact_sequence(matches, episodes)


def _bool(info: dict, *names: str) -> bool:
    return any(bool(info.get(name, False)) for name in names)


def parse_episode_artifact(path: Path, episode_index: int, condition: str) -> dict:
    with np.load(path, allow_pickle=False) as data:
        rewards = np.asarray(data["rewards"], dtype=float)
        terminations = np.asarray(data["terminations"], dtype=bool)
        truncations = np.asarray(data["truncations"], dtype=bool)
        info = json.loads(str(data["episode_info_json"].item()))
    return parse_episode_arrays(
        rewards, terminations, truncations, info, episode_index, condition, str(path.resolve())
    )


def parse_episode_arrays(
    rewards: np.ndarray,
    terminations: np.ndarray,
    truncations: np.ndarray,
    info: dict,
    episode_index: int,
    condition: str,
    artifact: str,
) -> dict:
    capture = _bool(info, "capture_reward_claimed")
    hvt_unsafe = _bool(info, "hvt_unsafe", "is_hvt_breach")
    collision = _bool(info, "collided")
    outside = _bool(info, "out_of_boundary", "out_satellite")
    truncated = bool(truncations[-1]) if len(truncations) else False
    if capture:
        outcome = "capture_success"
    elif hvt_unsafe:
        outcome = "hvt_unsafe"
    elif collision:
        outcome = "collision"
    elif outside:
        outcome = "outside"
    elif truncated or not (bool(terminations[-1]) if len(terminations) else False):
        outcome = "truncated"
    else:
        outcome = "other"
    return {
        "condition": condition,
        "episode_index": int(episode_index),
        "capture_success": capture,
        "hvt_unsafe": hvt_unsafe,
        "collision": collision,
        "outside": outside,
        "truncated": truncated,
        "outcome": outcome,
        "team_return": float(rewards.sum()),
        "episode_length": int(rewards.shape[0]),
        "artifact": artifact,
    }


def validate_complete_pairs(rows: Sequence[dict], episodes: int, conditions: Sequence[str]) -> None:
    counts: dict[tuple[int, str], int] = {}
    for row in rows:
        key = (int(row["episode_index"]), str(row["condition"]))
        counts[key] = counts.get(key, 0) + 1
    expected = {(episode_index, condition) for episode_index in range(episodes) for condition in conditions}
    actual = set(counts)
    duplicates = sorted(key for key, count in counts.items() if count != 1)
    if actual != expected or duplicates:
        raise ValueError(
            f"incomplete paired experiment: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}, duplicates={duplicates}"
        )


def summarize_conditions(rows: Sequence[dict]) -> list[dict]:
    result = []
    for condition in sorted({str(row["condition"]) for row in rows}):
        group = [row for row in rows if row["condition"] == condition]
        result.append({
            "condition": condition,
            "episodes": len(group),
            "capture_success_count": sum(bool(row["capture_success"]) for row in group),
            "capture_success_rate": float(np.mean([row["capture_success"] for row in group])),
            "mean_team_return": float(np.mean([row["team_return"] for row in group])),
            "mean_episode_length": float(np.mean([row["episode_length"] for row in group])),
        })
    return result


def compute_paired_effects(
    rows: Sequence[dict], baseline: str = "sparse_communication", bootstrap_samples: int = 10000,
    analysis_seed: int = 20260802,
) -> list[dict]:
    by_cell = {(int(row["episode_index"]), str(row["condition"])): row for row in rows}
    episode_indices = sorted({int(row["episode_index"]) for row in rows})
    conditions = sorted({str(row["condition"]) for row in rows if row["condition"] != baseline})
    rng = np.random.default_rng(analysis_seed)
    results = []
    for condition in conditions:
        for metric in METRICS:
            diffs = np.asarray([
                float(by_cell[(episode_index, condition)][metric]) - float(by_cell[(episode_index, baseline)][metric])
                for episode_index in episode_indices
            ])
            indices = rng.integers(0, len(diffs), size=(bootstrap_samples, len(diffs)))
            boot = diffs[indices].mean(axis=1)
            results.append({
                "condition": condition,
                "baseline": baseline,
                "metric": metric,
                "effect": float(diffs.mean()),
                "standard_error": float(diffs.std(ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else 0.0,
                "ci95_low": float(np.percentile(boot, 2.5)),
                "ci95_high": float(np.percentile(boot, 97.5)),
                "paired_episodes": len(diffs),
            })
    return results


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def condition_metadata(config: ExperimentConfig, condition: str) -> dict:
    return {
        "model_path": str(config.model_path.resolve()),
        "episodes": config.episodes,
        "condition": condition,
        "inference_only": True,
    }


def load_condition_rows(
    batch_root: Path, config: ExperimentConfig, condition: str
) -> list[dict]:
    condition_root = batch_root / condition
    metadata_path = condition_root / "condition_metadata.json"
    if not metadata_path.exists():
        raise ValueError(f"missing condition metadata: {metadata_path}")
    actual = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_condition_metadata(actual, condition_metadata(config, condition))
    artifacts = discover_episode_artifacts(condition_root, config.episodes)
    return [
        parse_episode_artifact(path, episode_index, condition)
        for episode_index, path in enumerate(artifacts)
    ]


def aggregate_batch(batch_root: Path, config: ExperimentConfig) -> list[dict]:
    rows = []
    for condition in CONDITIONS:
        rows.extend(load_condition_rows(batch_root, config, condition))
    validate_complete_pairs(rows, config.episodes, CONDITIONS)
    _write_csv(batch_root / "episodes.csv", rows)
    _write_csv(batch_root / "summary.csv", summarize_conditions(rows))
    _write_csv(
        batch_root / "paired_effects.csv", compute_paired_effects(rows)
    )
    _write_json(
        batch_root / "metadata.json",
        {
            "model_path": str(config.model_path.resolve()),
            "episodes": config.episodes,
            "conditions": list(CONDITIONS),
            "inference_only": True,
        },
    )
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("output/models/SatelliteEnv3v1HvtNoPlaneLimitAccel"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/communication_ablation_3v1"))
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--episodes", type=int, default=40)
    condition_group = parser.add_mutually_exclusive_group()
    condition_group.add_argument("--condition", choices=CONDITIONS)
    condition_group.add_argument("--conditions", nargs="+", choices=CONDITIONS)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    validate_batch_id(args.batch_id)
    if args.aggregate_only and (args.condition or args.conditions):
        raise ValueError("--aggregate-only cannot be combined with condition options")
    selected_conditions = (
        (args.condition,)
        if args.condition
        else tuple(args.conditions) if args.conditions else CONDITIONS
    )
    config = ExperimentConfig(
        args.model_path, args.episodes, selected_conditions
    ).validate()
    batch_root = resolve_batch_root(args.output_dir, args.batch_id)
    batch_root.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        aggregate_config = ExperimentConfig(
            args.model_path, args.episodes, CONDITIONS
        ).validate()
        rows = aggregate_batch(batch_root, aggregate_config)
        print(
            f"批次汇总完成 | batch={args.batch_id} | episodes={len(rows)} | "
            f"output={batch_root}",
            flush=True,
        )
        return 0
    total_conditions = len(config.conditions)
    experiment_started = time.monotonic()
    completed_durations: list[float] = []
    print(
        format_startup(
            config.model_path.resolve(),
            batch_root.resolve(),
            config.episodes,
            len(config.conditions),
        ),
        flush=True,
    )
    failures = []
    for condition_index, condition in enumerate(config.conditions, start=1):
        print(
            format_cell_start(condition_index, total_conditions, condition),
            flush=True,
        )
        condition_root, eval_data_root = build_condition_paths(
            batch_root, condition
        )
        command = build_inference_command(
            config, condition, eval_data_root
        )
        if args.dry_run:
            print(subprocess.list2cmdline(command), flush=True)
            continue
        condition_root.mkdir(parents=True, exist_ok=True)
        expected_metadata = condition_metadata(config, condition)
        condition_metadata_path = condition_root / "condition_metadata.json"
        if condition_metadata_path.exists():
            validate_condition_metadata(
                json.loads(condition_metadata_path.read_text(encoding="utf-8")),
                expected_metadata,
            )
        else:
            _write_json(condition_metadata_path, expected_metadata)
        log_path = condition_root / "inference.log"
        existing_artifacts = list(condition_root.rglob("episode_*.npz"))
        if existing_artifacts:
            try:
                artifacts = validate_episode_artifact_sequence(
                    existing_artifacts, config.episodes
                )
            except ValueError as exc:
                failures.append({
                    "condition": condition,
                    "error": f"incomplete existing condition output: {exc}",
                    "log": str(log_path),
                })
                print(
                    f"[{condition_index}/{total_conditions}] 失败 | "
                    f"已有结果不完整 | error={exc}",
                    flush=True,
                )
                continue
            print(
                f"[{condition_index}/{total_conditions}] 跳过 | "
                f"已存在完整结果 | episodes={len(artifacts)}",
                flush=True,
            )
            continue
        condition_started = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log:
            returncode = stream_subprocess(command, condition, log)
        if returncode:
            failures.append({
                "condition": condition,
                "returncode": returncode,
                "log": str(log_path),
            })
            print(
                f"[{condition_index}/{total_conditions}] 失败 | "
                f"returncode={returncode} | log={log_path}",
                flush=True,
            )
            continue
        try:
            artifacts = discover_episode_artifacts(
                condition_root, config.episodes
            )
            condition_elapsed = time.monotonic() - condition_started
            completed_durations.append(condition_elapsed)
            eta = float(np.mean(completed_durations)) * (
                total_conditions - condition_index
            )
            print(
                format_cell_complete(
                    condition_index,
                    total_conditions,
                    episodes=len(artifacts),
                    cell_seconds=condition_elapsed,
                    total_seconds=time.monotonic() - experiment_started,
                    eta_seconds=eta,
                ),
                flush=True,
            )
        except Exception as exc:
            failures.append({
                "condition": condition,
                "returncode": 0,
                "error": str(exc),
                "log": str(log_path),
            })
            print(
                f"[{condition_index}/{total_conditions}] 失败 | "
                f"error={exc} | log={log_path}",
                flush=True,
            )
    if args.dry_run:
        return 0
    for condition in config.conditions:
        condition_root = batch_root / condition
        condition_failures = [
            failure for failure in failures if failure["condition"] == condition
        ]
        _write_json(condition_root / "failures.json", condition_failures)
    if failures:
        return 1
    if set(config.conditions) == set(CONDITIONS):
        aggregate_config = ExperimentConfig(
            config.model_path, config.episodes, CONDITIONS
        )
        aggregate_batch(batch_root, aggregate_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
