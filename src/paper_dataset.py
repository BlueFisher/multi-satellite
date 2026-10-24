"""Build a read-only, self-contained snapshot of evaluation experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import statistics
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RunSource:
    run_id: str
    experiment_type: str
    source_dir: Path
    env_name: str
    batch_id: str = ""
    condition_id: str = ""
    model_id: str = ""
    scan_name: str = ""
    overrides: dict[str, Any] = field(default_factory=dict)
    status: str = "succeeded"
    started_at: str = ""
    finished_at: str = ""


@dataclass(frozen=True)
class QualityIssue:
    severity: str
    run_id: str
    episode: int | None
    file: str
    check_name: str
    expected: str
    actual: str
    message: str


@dataclass
class EpisodeRecord:
    run_id: str
    experiment_type: str
    episode: int
    npz_path: Path
    summary_values: dict[str, str]
    metadata: dict[str, Any]
    episode_info: dict[str, Any]
    steps: int
    duration_s: float
    success: bool
    termination_reason: str
    episode_total_reward: float
    global_min_distance_km: float
    satellite_count: int
    agent_count: int
    observation_dim: int
    action_dim: int
    issues: list[QualityIssue]

    @property
    def validation_status(self) -> str:
        return "error" if any(issue.severity == "error" for issue in self.issues) else "valid"


@dataclass(frozen=True)
class SnapshotSources:
    baseline_2v2: Path
    baseline_3v1: Path
    target_root: Path
    batch_manifest: Path
    sweep_config: Path
    logs_dir: Path


@dataclass(frozen=True)
class BuildReport:
    output_dir: Path
    run_count: int
    episode_count: int
    error_count: int
    warning_count: int


def discover_baseline_runs(root: Path) -> list[RunSource]:
    """Return baseline leaf directories that contain summaries and episodes."""
    root = root.resolve()
    runs: list[RunSource] = []
    for summary_path in sorted(root.rglob("summary.csv")):
        run_dir = summary_path.parent
        if not any(run_dir.glob("episode_*.npz")):
            continue
        relative = run_dir.relative_to(root)
        stable_suffix = "__".join((root.name, *relative.parts))
        runs.append(
            RunSource(
                run_id=f"baseline__{stable_suffix}",
                experiment_type="baseline",
                source_dir=run_dir,
                env_name=root.name,
            )
        )
    return runs


def discover_target_runs(data_root: Path, manifest_path: Path) -> list[RunSource]:
    """Return successful target-test conditions in manifest order."""
    data_root = data_root.resolve()
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    batch_id = str(manifest["batch_id"])
    seen: set[str] = set()
    runs: list[RunSource] = []
    for condition in manifest["conditions"]:
        condition_id = str(condition["condition_id"])
        if condition_id in seen:
            raise ValueError(f"duplicate condition_id: {condition_id}")
        seen.add(condition_id)
        if condition.get("status") != "succeeded":
            raise ValueError(f"target condition did not succeed: {condition_id}")
        source_dir = Path(condition.get("output_dir") or data_root / condition_id).resolve()
        try:
            source_dir.relative_to(data_root)
        except ValueError as exc:
            raise ValueError(
                f"target output is outside data root: {source_dir}"
            ) from exc
        if not (source_dir / "summary.csv").is_file():
            raise FileNotFoundError(source_dir / "summary.csv")
        if not any(source_dir.glob("episode_*.npz")):
            raise FileNotFoundError(f"no episode NPZ in {source_dir}")
        runs.append(
            RunSource(
                run_id=f"target__{batch_id}__{condition_id}",
                experiment_type="target_test",
                source_dir=source_dir,
                env_name=str(condition["environment"]),
                batch_id=batch_id,
                condition_id=condition_id,
                model_id=str(condition.get("model_id", "")),
                scan_name=str(condition.get("scan_name", "")),
                overrides=dict(condition.get("overrides", {})),
                status=str(condition["status"]),
                started_at=str(condition.get("started_at", "")),
                finished_at=str(condition.get("finished_at", "")),
            )
        )
    return runs


REQUIRED_NPZ_FIELDS = {
    "step",
    "time_s",
    "satellite_names",
    "agent_names",
    "state_eci",
    "state_lvlh",
    "orbital_elements",
    "distance_matrix_km",
    "observations",
    "next_observations",
    "actions",
    "rewards",
    "cumulative_rewards",
    "terminations",
    "truncations",
    "episode_info_json",
    "metadata_json",
}


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _issue(
    issues: list[QualityIssue],
    run: RunSource,
    episode: int | None,
    path: Path,
    check_name: str,
    expected: Any,
    actual: Any,
    message: str,
    severity: str = "error",
) -> None:
    issues.append(
        QualityIssue(
            severity=severity,
            run_id=run.run_id,
            episode=episode,
            file=str(path),
            check_name=check_name,
            expected=str(expected),
            actual=str(actual),
            message=message,
        )
    )


def validate_episode(
    npz_path: Path, summary_row: dict[str, str], run: RunSource
) -> EpisodeRecord:
    """Validate one transition-aligned evaluation episode."""
    issues: list[QualityIssue] = []
    episode = int(summary_row["episode"])
    with np.load(npz_path, allow_pickle=False) as data:
        missing = sorted(REQUIRED_NPZ_FIELDS.difference(data.files))
        if missing:
            raise ValueError(f"missing required NPZ fields in {npz_path}: {missing}")
        arrays = {name: data[name] for name in data.files}

    try:
        metadata = json.loads(str(arrays["metadata_json"]))
        episode_info = json.loads(str(arrays["episode_info_json"]))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON metadata in {npz_path}: {exc}") from exc

    steps_array = arrays["step"]
    t_steps = int(len(steps_array))
    if not np.array_equal(steps_array, np.arange(t_steps)):
        _issue(
            issues,
            run,
            episode,
            npz_path,
            "step_sequence",
            f"0..{max(t_steps - 1, 0)}",
            steps_array.tolist(),
            "step must be consecutive and zero-based",
        )

    state_fields = ("time_s", "state_eci", "state_lvlh", "orbital_elements", "distance_matrix_km")
    for name in state_fields:
        actual = len(arrays[name])
        if actual != t_steps + 1:
            _issue(
                issues,
                run,
                episode,
                npz_path,
                "time_axis_length" if name == "time_s" else f"{name}_length",
                t_steps + 1,
                actual,
                f"{name} must include the initial state",
            )

    transition_fields = (
        "observations",
        "next_observations",
        "actions",
        "rewards",
        "cumulative_rewards",
        "terminations",
        "truncations",
    )
    for name in transition_fields:
        actual = len(arrays[name])
        if actual != t_steps:
            _issue(
                issues,
                run,
                episode,
                npz_path,
                f"{name}_length",
                t_steps,
                actual,
                f"{name} must align with transitions",
            )

    satellite_count = int(len(arrays["satellite_names"]))
    agent_count = int(len(arrays["agent_names"]))
    for name in ("state_eci", "state_lvlh", "orbital_elements"):
        shape = arrays[name].shape
        expected_tail = (satellite_count, 6)
        if len(shape) != 3 or shape[1:] != expected_tail:
            _issue(
                issues,
                run,
                episode,
                npz_path,
                f"{name}_shape",
                f"[T+1,{satellite_count},6]",
                shape,
                f"{name} satellite and field axes do not match metadata axes",
            )

    distance = arrays["distance_matrix_km"]
    expected_distance_shape = (t_steps + 1, satellite_count, satellite_count)
    if distance.shape != expected_distance_shape:
        _issue(
            issues,
            run,
            episode,
            npz_path,
            "distance_shape",
            expected_distance_shape,
            distance.shape,
            "distance matrix axes must match satellite_names",
        )
    else:
        if not np.allclose(distance, np.swapaxes(distance, 1, 2), atol=1e-8, rtol=0):
            _issue(
                issues,
                run,
                episode,
                npz_path,
                "distance_symmetry",
                "symmetric within 1e-8 km",
                "asymmetric",
                "distance matrix must be symmetric",
            )
        diagonal = np.diagonal(distance, axis1=1, axis2=2)
        if not np.allclose(diagonal, 0.0, atol=1e-8, rtol=0):
            _issue(
                issues,
                run,
                episode,
                npz_path,
                "distance_diagonal",
                "zero within 1e-8 km",
                float(np.max(np.abs(diagonal))),
                "self distance must be zero",
            )

    for name in ("observations", "next_observations", "actions", "rewards", "cumulative_rewards"):
        shape = arrays[name].shape
        if len(shape) < 2 or shape[1] != agent_count:
            _issue(
                issues,
                run,
                episode,
                npz_path,
                f"{name}_agent_axis",
                agent_count,
                shape,
                f"{name} agent axis must match agent_names",
            )

    for name, value in arrays.items():
        if np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
            _issue(
                issues,
                run,
                episode,
                npz_path,
                "finite_values",
                "all numeric values finite",
                name,
                f"{name} contains NaN or infinity",
            )

    rewards = arrays["rewards"]
    total_reward = float(np.sum(rewards))
    summary_total = float(summary_row["episode_total_reward"])
    if not np.isclose(total_reward, summary_total, atol=1e-8, rtol=1e-8):
        _issue(
            issues,
            run,
            episode,
            npz_path,
            "summary_total_reward",
            total_reward,
            summary_total,
            "summary total reward must equal the sum of per-agent rewards",
        )
    summary_steps = int(float(summary_row["steps"]))
    if summary_steps != t_steps:
        _issue(
            issues,
            run,
            episode,
            npz_path,
            "summary_steps",
            t_steps,
            summary_steps,
            "summary steps must equal transition count",
        )

    duration_s = float(arrays["time_s"][-1] - arrays["time_s"][0]) if len(arrays["time_s"]) else 0.0
    observation_dim = int(arrays["observations"].shape[-1]) if arrays["observations"].ndim >= 3 else 0
    action_dim = int(arrays["actions"].shape[-1]) if arrays["actions"].ndim >= 3 else 0
    return EpisodeRecord(
        run_id=run.run_id,
        experiment_type=run.experiment_type,
        episode=episode,
        npz_path=npz_path,
        summary_values=dict(summary_row),
        metadata=metadata,
        episode_info=episode_info,
        steps=t_steps,
        duration_s=duration_s,
        success=_as_bool(summary_row.get("success", False)),
        termination_reason=str(summary_row.get("termination_reason", "")),
        episode_total_reward=total_reward,
        global_min_distance_km=float(summary_row.get("global_min_distance_km", "nan")),
        satellite_count=satellite_count,
        agent_count=agent_count,
        observation_dim=observation_dim,
        action_dim=action_dim,
        issues=issues,
    )


def load_run(run: RunSource) -> tuple[list[EpisodeRecord], list[QualityIssue]]:
    """Load a run only when summary rows and NPZ files align one-to-one."""
    summary_path = run.source_dir / "summary.csv"
    with summary_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_npz: dict[str, dict[str, str]] = {}
    issues: list[QualityIssue] = []
    for row in rows:
        filename = str(row.get("npz_file", ""))
        if not filename or filename in by_npz:
            _issue(
                issues,
                run,
                int(row["episode"]) if row.get("episode", "").isdigit() else None,
                summary_path,
                "summary_npz_alignment",
                "unique non-empty npz_file",
                filename,
                "summary rows must identify unique NPZ files",
            )
            continue
        by_npz[filename] = row

    npz_files = {path.name: path for path in sorted(run.source_dir.glob("episode_*.npz"))}
    if set(by_npz) != set(npz_files):
        _issue(
            issues,
            run,
            None,
            summary_path,
            "summary_npz_alignment",
            sorted(npz_files),
            sorted(by_npz),
            "summary.csv and episode NPZ files must match one-to-one",
        )
        return [], issues

    episodes = [validate_episode(npz_files[name], by_npz[name], run) for name in sorted(npz_files)]
    for episode_record in episodes:
        issues.extend(episode_record.issues)
    return episodes, issues


def _group_values(episode: EpisodeRecord) -> tuple[Any, ...]:
    metadata = episode.metadata
    run_overrides = metadata.get("sweep_overrides", {})
    return (
        episode.experiment_type,
        str(metadata.get("env_name", "")),
        str(metadata.get("alg_name", "")),
        bool(metadata.get("on_plane", False)),
        bool(metadata.get("limit_accel", False)),
        str(metadata.get("model_id", "")),
        str(metadata.get("scan_name", "")),
        json.dumps(run_overrides, ensure_ascii=False, sort_keys=True),
    )


def _describe(values: list[float], prefix: str) -> dict[str, object]:
    return {
        f"{prefix}_mean": statistics.fmean(values),
        f"{prefix}_stdev": statistics.stdev(values) if len(values) > 1 else "",
        f"{prefix}_min": min(values),
        f"{prefix}_median": statistics.median(values),
        f"{prefix}_max": max(values),
    }


def summarize_configurations(
    episodes: list[EpisodeRecord],
) -> list[dict[str, object]]:
    grouped: dict[tuple[Any, ...], list[EpisodeRecord]] = {}
    for episode in episodes:
        grouped.setdefault(_group_values(episode), []).append(episode)
    rows: list[dict[str, object]] = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        (
            experiment_type,
            env_name,
            alg_name,
            on_plane,
            limit_accel,
            model_id,
            scan_name,
            scan_values_json,
        ) = key
        row: dict[str, object] = {
            "experiment_type": experiment_type,
            "env_name": env_name,
            "alg_name": alg_name,
            "on_plane": on_plane,
            "limit_accel": limit_accel,
            "model_id": model_id,
            "scan_name": scan_name,
            "scan_values_json": scan_values_json,
            "run_count": len({episode.run_id for episode in group}),
            "episode_count": len(group),
            "success_count": sum(episode.success for episode in group),
            "success_rate": sum(episode.success for episode in group) / len(group),
        }
        metrics = {
            "steps": [float(episode.steps) for episode in group],
            "duration_s": [episode.duration_s for episode in group],
            "episode_total_reward": [episode.episode_total_reward for episode in group],
            "global_min_distance_km": [episode.global_min_distance_km for episode in group],
        }
        for name, values in metrics.items():
            row.update(_describe(values, name))
        rows.append(row)
    return rows


def summarize_terminations(
    episodes: list[EpisodeRecord],
) -> list[dict[str, object]]:
    totals: dict[tuple[Any, ...], int] = {}
    counts: dict[tuple[Any, ...], int] = {}
    for episode in episodes:
        metadata = episode.metadata
        group = (
            episode.experiment_type,
            str(metadata.get("env_name", "")),
            str(metadata.get("alg_name", "")),
            bool(metadata.get("on_plane", False)),
            bool(metadata.get("limit_accel", False)),
        )
        totals[group] = totals.get(group, 0) + 1
        key = (*group, episode.termination_reason)
        counts[key] = counts.get(key, 0) + 1
    rows = []
    for key, count in sorted(counts.items(), key=lambda item: tuple(map(str, item[0]))):
        group = key[:-1]
        rows.append(
            {
                "experiment_type": key[0],
                "env_name": key[1],
                "alg_name": key[2],
                "on_plane": key[3],
                "limit_accel": key[4],
                "termination_reason": key[5],
                "episode_count": count,
                "proportion": count / totals[group],
            }
        )
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            normalized = {
                name: (
                    json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else str(value).lower()
                    if isinstance(value, bool)
                    else value
                )
                for name, value in row.items()
            }
            writer.writerow(normalized)


def _copy_verified(source: Path, destination: Path) -> tuple[str, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256_file(source)
    shutil.copy2(source, destination)
    destination_hash = sha256_file(destination)
    if source_hash != destination_hash:
        raise OSError(f"SHA-256 mismatch after copying {source} to {destination}")
    return source_hash, destination_hash


def _episode_row(episode: EpisodeRecord, snapshot_npz: Path) -> dict[str, object]:
    metadata = episode.metadata
    row: dict[str, object] = {
        "run_id": episode.run_id,
        "experiment_type": episode.experiment_type,
        "episode": episode.episode,
        "npz_path": snapshot_npz.as_posix(),
        "env_name": metadata.get("env_name", ""),
        "alg_name": metadata.get("alg_name", ""),
        "model_path": metadata.get("model_path", ""),
        "seed": metadata.get("seed", ""),
        "update_interval_s": metadata.get("update_interval_s", ""),
        "on_plane": metadata.get("on_plane", ""),
        "limit_accel": metadata.get("limit_accel", ""),
        "steps": episode.steps,
        "duration_s": episode.duration_s,
        "success": episode.success,
        "termination_reason": episode.termination_reason,
        "episode_total_reward": episode.episode_total_reward,
        "global_min_distance_km": episode.global_min_distance_km,
        "satellite_count": episode.satellite_count,
        "agent_count": episode.agent_count,
        "observation_dim": episode.observation_dim,
        "action_dim": episode.action_dim,
        "validation_status": episode.validation_status,
        "quality_issue_count": len(episode.issues),
    }
    for name, value in episode.summary_values.items():
        row.setdefault(name, value)
    return row


def _field_dictionary(table_columns: dict[str, list[str]]) -> list[dict[str, str]]:
    unit_by_suffix = {
        "_km": "km",
        "_km_s": "km/s",
        "_s": "s",
        "_rate": "1",
        "_count": "count",
        "_deg": "deg",
    }
    rows = []
    for table_name, columns in table_columns.items():
        for column in columns:
            unit = next((value for suffix, value in unit_by_suffix.items() if column.endswith(suffix)), "")
            rows.append(
                {
                    "table_name": table_name,
                    "field_name": column,
                    "description": column.replace("_", " "),
                    "data_type": "mixed" if table_name == "episodes" else "scalar",
                    "unit": unit,
                    "source": "NPZ metadata or summary.csv" if table_name in {"runs", "episodes"} else "derived",
                    "missing_rule": "blank means not applicable or unavailable",
                }
            )
    return rows


def build_snapshot(
    sources: SnapshotSources,
    output_dir: Path,
    *,
    staging_root: Path | None = None,
    workbook_command: list[str] | None = None,
) -> BuildReport:
    """Build and atomically publish a verified, self-contained snapshot."""
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    staging_parent = (staging_root or output_dir.parent).resolve()
    staging_parent.mkdir(parents=True, exist_ok=True)
    building_dir = staging_parent / f".{output_dir.name}.building"
    if building_dir.exists():
        raise FileExistsError(building_dir)
    building_dir.mkdir(parents=True)

    runs = discover_baseline_runs(sources.baseline_2v2)
    runs.extend(discover_baseline_runs(sources.baseline_3v1))
    runs.extend(discover_target_runs(sources.target_root, sources.batch_manifest))
    if len({run.run_id for run in runs}) != len(runs):
        raise ValueError("run_id values are not unique")

    all_episodes: list[EpisodeRecord] = []
    all_issues: list[QualityIssue] = []
    run_rows: list[dict[str, object]] = []
    copied_sources: dict[str, tuple[Path, str, str]] = {}

    baseline_roots = {
        "SatelliteEnv2v2": sources.baseline_2v2.resolve(),
        "SatelliteEnv3v1Hvt": sources.baseline_3v1.resolve(),
    }
    for run in runs:
        episodes, issues = load_run(run)
        all_episodes.extend(episodes)
        all_issues.extend(issues)
        if run.experiment_type == "baseline":
            relative_run = run.source_dir.resolve().relative_to(baseline_roots[run.env_name])
            relative_snapshot_dir = Path("raw") / "baseline" / run.env_name / relative_run
        else:
            relative_snapshot_dir = Path("raw") / "target_tests" / run.condition_id

        source_files = [run.source_dir / "summary.csv", *sorted(run.source_dir.glob("episode_*.npz"))]
        for source_file in source_files:
            relative_destination = relative_snapshot_dir / source_file.name
            source_hash, destination_hash = _copy_verified(
                source_file, building_dir / relative_destination
            )
            copied_sources[relative_destination.as_posix()] = (
                source_file.resolve(),
                source_hash,
                destination_hash,
            )
        for episode in episodes:
            episode.metadata = {
                **episode.metadata,
                "model_id": run.model_id,
                "scan_name": run.scan_name,
                "sweep_overrides": run.overrides,
            }
        first_metadata = episodes[0].metadata if episodes else {}
        run_rows.append(
            {
                "run_id": run.run_id,
                "experiment_type": run.experiment_type,
                "source_path": str(run.source_dir.resolve()),
                "snapshot_path": relative_snapshot_dir.as_posix(),
                "batch_id": run.batch_id,
                "condition_id": run.condition_id,
                "model_id": run.model_id,
                "scan_name": run.scan_name,
                "scan_values_json": run.overrides,
                "env_name": first_metadata.get("env_name", run.env_name),
                "alg_name": first_metadata.get("alg_name", ""),
                "model_path": first_metadata.get("model_path", ""),
                "on_plane": first_metadata.get("on_plane", ""),
                "limit_accel": first_metadata.get("limit_accel", ""),
                "seed": first_metadata.get("seed", ""),
                "update_interval_s": first_metadata.get("update_interval_s", ""),
                "episode_count": len(episodes),
                "status": run.status,
                "started_at": run.started_at,
                "finished_at": run.finished_at,
            }
        )

    provenance = {
        Path("provenance/batch_manifest.json"): sources.batch_manifest,
        Path("provenance/target_sweeps.yaml"): sources.sweep_config,
    }
    for log_path in sorted(sources.logs_dir.glob("*.log")):
        provenance[Path("provenance/logs") / log_path.name] = log_path
    for relative_destination, source_file in provenance.items():
        source_hash, destination_hash = _copy_verified(
            source_file, building_dir / relative_destination
        )
        copied_sources[relative_destination.as_posix()] = (
            source_file.resolve(),
            source_hash,
            destination_hash,
        )

    error_count = sum(issue.severity == "error" for issue in all_issues)
    warning_count = sum(issue.severity == "warning" for issue in all_issues)
    if error_count:
        raise RuntimeError(f"snapshot validation found {error_count} severe issues")

    episode_rows: list[dict[str, object]] = []
    for run, run_row in zip(runs, run_rows):
        relative_snapshot_dir = Path(str(run_row["snapshot_path"]))
        for episode in (item for item in all_episodes if item.run_id == run.run_id):
            episode_rows.append(
                _episode_row(episode, relative_snapshot_dir / episode.npz_path.name)
            )
    config_rows = summarize_configurations(all_episodes)
    termination_rows = summarize_terminations(all_episodes)

    run_columns = list(run_rows[0]) if run_rows else []
    episode_columns = list(dict.fromkeys(name for row in episode_rows for name in row))
    config_columns = list(config_rows[0]) if config_rows else []
    termination_columns = list(termination_rows[0]) if termination_rows else []
    issue_columns = [
        "severity",
        "run_id",
        "episode",
        "file",
        "check_name",
        "expected",
        "actual",
        "message",
    ]
    tables_dir = building_dir / "tables"
    _write_csv(tables_dir / "runs.csv", run_rows, run_columns)
    _write_csv(tables_dir / "episodes.csv", episode_rows, episode_columns)
    _write_csv(tables_dir / "configuration_summary.csv", config_rows, config_columns)
    _write_csv(tables_dir / "termination_summary.csv", termination_rows, termination_columns)
    _write_csv(
        tables_dir / "quality_issues.csv",
        [issue.__dict__ for issue in all_issues],
        issue_columns,
    )
    dictionary_rows = _field_dictionary(
        {
            "runs": run_columns,
            "episodes": episode_columns,
            "configuration_summary": config_columns,
            "termination_summary": termination_columns,
            "quality_issues": issue_columns,
        }
    )
    dictionary_columns = list(dictionary_rows[0]) if dictionary_rows else []
    _write_csv(tables_dir / "field_dictionary.csv", dictionary_rows, dictionary_columns)

    readme = f"""# 论文实验数据集

生成时间：{datetime.now(timezone.utc).isoformat()}

本快照包含 {len(runs)} 个运行、{len(all_episodes)} 个回合。数据来源为两个基准环境目录和打靶批次 {sources.target_root}。

明确排除 `output/eval_data/target_sweeps`、`src/eval_internal`、`img`、模型和训练数据。原始 NPZ 与 summary.csv 仅复制，未修改；复制结果通过 SHA-256 校验。

使用 `numpy.load(path, allow_pickle=False)` 读取 NPZ。完整状态序列长度为 T+1，动作、观测和奖励序列长度为 T。`tables/episodes.csv` 是回合级主索引，`tables/runs.csv` 是运行级索引，统计表均可回溯至回合级表。
"""
    (building_dir / "README.md").write_text(readme, encoding="utf-8")

    if workbook_command:
        render_dir = staging_parent / f".{output_dir.name}.renders"
        command = [
            part.format(dataset=str(building_dir), render=str(render_dir))
            for part in workbook_command
        ]
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "workbook builder failed with exit code "
                f"{result.returncode}: {result.stderr.strip()}"
            )

    manifest_columns = [
        "snapshot_path",
        "role",
        "size_bytes",
        "modified_at",
        "source_path",
        "source_sha256",
        "snapshot_sha256",
    ]
    manifest_rows: list[dict[str, object]] = []
    for snapshot_file in sorted(path for path in building_dir.rglob("*") if path.is_file()):
        relative = snapshot_file.relative_to(building_dir).as_posix()
        if relative == "tables/file_manifest.csv":
            continue
        source_info = copied_sources.get(relative)
        role = "raw" if relative.startswith("raw/") else "provenance" if relative.startswith("provenance/") else "derived"
        manifest_rows.append(
            {
                "snapshot_path": relative,
                "role": role,
                "size_bytes": snapshot_file.stat().st_size,
                "modified_at": datetime.fromtimestamp(
                    snapshot_file.stat().st_mtime, timezone.utc
                ).isoformat(),
                "source_path": str(source_info[0]) if source_info else "",
                "source_sha256": source_info[1] if source_info else "",
                "snapshot_sha256": source_info[2] if source_info else sha256_file(snapshot_file),
            }
        )
    _write_csv(tables_dir / "file_manifest.csv", manifest_rows, manifest_columns)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    building_dir.replace(output_dir)
    return BuildReport(
        output_dir=output_dir,
        run_count=len(runs),
        episode_count=len(all_episodes),
        error_count=error_count,
        warning_count=warning_count,
    )
