"""Three-versus-one single-factor inference sweep orchestration."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SAFE_NAME = re.compile(r"[A-Za-z0-9_.-]+")
FIXED_MODEL_ARGUMENTS = {
    "env_name": "SatelliteEnv3v1Hvt",
    "alg_name": "SharedMessageMATD3",
    "load_path": "output/models/SatelliteEnv3v1HvtNoPlaneLimitAccel",
    "obs_add_distance": False,
    "on_plane": False,
    "limit_accel": True,
}
EXPECTED_FACTORS = {
    "capture_duration": "capture_sustain_steps",
    "attacker_accel": "attacker_max_accel",
    "collision_threshold": "collision_threshold",
    "capture_radius": "capture_distance_threshold_km",
    "hvt_safe_distance": "hvt_safe_distance",
    "outside_threshold": "outside_threshold",
    "attacker_initial_distance": "attacker_initial_rho_range_km",
}


@dataclass(frozen=True)
class ModelSpec:
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class FactorSpec:
    name: str
    parameter: str
    values: tuple[Any, ...]


@dataclass(frozen=True)
class Condition:
    model: ModelSpec
    factor_name: str
    condition_name: str
    overrides: Mapping[str, Any]


def load_catalog(path: Path) -> tuple[ModelSpec, tuple[FactorSpec, ...]]:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("3v1 single-factor catalog must be a mapping")
    if raw.get("model") != FIXED_MODEL_ARGUMENTS:
        raise ValueError("catalog model must match the fixed 3v1 SharedMessageMATD3 model")
    model_path = REPOSITORY_ROOT / FIXED_MODEL_ARGUMENTS["load_path"]
    if not model_path.exists():
        raise FileNotFoundError(f"model path does not exist: {model_path}")

    raw_factors = raw.get("factors")
    if not isinstance(raw_factors, dict) or list(raw_factors) != list(EXPECTED_FACTORS):
        raise ValueError("catalog must contain the seven ordered 3v1 factors")
    factors = []
    for name, expected_parameter in EXPECTED_FACTORS.items():
        factor = raw_factors[name]
        if not isinstance(factor, dict) or factor.get("parameter") != expected_parameter:
            raise ValueError(f"factor {name!r} has an invalid parameter")
        raw_values = factor.get("values")
        if not isinstance(raw_values, list) or not raw_values:
            raise ValueError(f"factor {name!r} must have values")
        values = tuple(_validate_value(name, value) for value in raw_values)
        if len({_value_identity(value) for value in values}) != len(values):
            raise ValueError(f"factor {name!r} contains duplicate values")
        factors.append(FactorSpec(name, expected_parameter, values))
    return ModelSpec(dict(FIXED_MODEL_ARGUMENTS)), tuple(factors)


def _validate_value(factor_name: str, value: Any) -> Any:
    if factor_name == "attacker_initial_distance":
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError("attacker initial distance must contain two endpoints")
        minimum, maximum = value
        if not _finite_number(minimum) or not _finite_number(maximum):
            raise ValueError("attacker initial distance endpoints must be finite")
        if minimum < 0 or minimum > maximum:
            raise ValueError("attacker initial distance must be non-negative and ordered")
        return minimum, maximum
    if not _finite_number(value):
        raise ValueError(f"factor {factor_name!r} values must be finite numbers")
    if value < 0 or (factor_name == "capture_duration" and value <= 0):
        raise ValueError(f"factor {factor_name!r} values are outside their domain")
    if factor_name == "capture_duration" and not float(value).is_integer():
        raise ValueError("capture duration values must be integers")
    return int(value) if factor_name == "capture_duration" else value


def _finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _value_identity(value: Any) -> tuple[Any, ...]:
    return tuple(value) if isinstance(value, tuple) else (value,)


def generate_conditions(
    model: ModelSpec, factors: Sequence[FactorSpec]
) -> list[Condition]:
    conditions = []
    for factor in factors:
        for value in factor.values:
            conditions.append(
                Condition(
                    model=model,
                    factor_name=factor.name,
                    condition_name=_condition_name(factor.name, value),
                    overrides={factor.parameter: value},
                )
            )
    return conditions


def _condition_name(factor_name: str, value: Any) -> str:
    prefixes = {
        "capture_duration": "steps",
        "attacker_accel": "accel",
        "collision_threshold": "collision",
        "capture_radius": "radius",
        "hvt_safe_distance": "distance",
        "outside_threshold": "distance",
    }
    if factor_name == "attacker_initial_distance":
        minimum, maximum = value
        return f"rho_{_id_value(minimum)}_{_id_value(maximum)}"
    return f"{prefixes[factor_name]}_{_id_value(value)}"


def _id_value(value: Any) -> str:
    return format(value, "g").replace("-", "m").replace(".", "p")


def _cli_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    return format(value, "g") if isinstance(value, float) else str(value)


def condition_data_root(batch_dir: Path, condition: Condition) -> Path:
    return Path(batch_dir) / "data" / condition.factor_name / condition.condition_name


def build_inference_command(
    condition: Condition,
    episodes: int,
    batch_dir: Path,
    python_executable: Sequence[str],
    seed: int,
) -> list[str]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not python_executable:
        raise ValueError("python executable tokens must not be empty")
    if dict(condition.model.arguments) != FIXED_MODEL_ARGUMENTS:
        raise ValueError("condition does not use the fixed 3v1 model")
    if SAFE_NAME.fullmatch(condition.factor_name) is None or SAFE_NAME.fullmatch(condition.condition_name) is None:
        raise ValueError("factor and condition names must be path safe")
    if len(condition.overrides) != 1:
        raise ValueError("each condition must override exactly one factor")

    command = list(python_executable) + [
        "src/main.py", "--just_eval", "True", "--load_model", "True"
    ]
    for key, value in condition.model.arguments.items():
        command.extend((f"--{key}", _cli_value(value)))
    command.extend(("--eval_episode", str(int(episodes)), "--seed", str(int(seed))))

    parameter, value = next(iter(condition.overrides.items()))
    if parameter == "attacker_initial_rho_range_km":
        minimum, maximum = value
        command.extend(("--attacker_initial_rho_min_km", _cli_value(minimum)))
        command.extend(("--attacker_initial_rho_max_km", _cli_value(maximum)))
    elif parameter in EXPECTED_FACTORS.values():
        command.extend((f"--{parameter}", _cli_value(value)))
    else:
        raise ValueError(f"unsupported factor parameter: {parameter}")

    data_dir = Path(batch_dir) / "data" / condition.factor_name
    command.extend(("--eval_data_dir", str(data_dir)))
    command.extend(("--eval_run_name", condition.condition_name))
    return command


def create_unique_batch_dir(output_root: Path, batch_id: str | None = None) -> Path:
    output_root = Path(output_root)
    stem = batch_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = output_root / stem
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = output_root / f"{stem}-{suffix:02d}"
    candidate.mkdir(parents=True)
    return candidate


def create_manifest(
    batch_dir: Path,
    conditions: Sequence[Condition],
    commands: Sequence[Sequence[str]],
    config_path: Path,
) -> Path:
    if len(conditions) != len(commands):
        raise ValueError("conditions and commands must have equal length")
    batch_dir = Path(batch_dir)
    entries = []
    for condition, command in zip(conditions, commands):
        entries.append({
            "factor_name": condition.factor_name,
            "condition_name": condition.condition_name,
            "overrides": _json_value(dict(condition.overrides)),
            "status": "pending",
            "command": list(command),
            "requested_output_dir": str(condition_data_root(batch_dir, condition)),
            "log_path": str(batch_dir / "logs" / condition.factor_name / f"{condition.condition_name}.log"),
        })
    manifest = {
        "batch_id": batch_dir.name,
        "config_path": str(Path(config_path).resolve()),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "conditions": entries,
    }
    path = batch_dir / "manifest.json"
    _write_json_atomic(path, manifest)
    return path


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return list(value)
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def run_manifest(
    manifest_path: Path,
    *,
    dry_run: bool = False,
    continue_on_error: bool = False,
    resume: bool = False,
    process_runner=subprocess.run,
) -> int:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if dry_run:
        return 0
    failed = False
    entries = manifest["conditions"]
    total = len(entries)
    for index, entry in enumerate(entries, start=1):
        summary_path = entry.get("summary_csv")
        if resume and entry.get("status") == "succeeded" and summary_path and Path(summary_path).is_file():
            print(
                f"[{index}/{total}] 跳过已完成 | 因素: {entry['factor_name']} | "
                f"条件: {entry['condition_name']}",
                flush=True,
            )
            continue
        output_root = Path(entry["requested_output_dir"])
        existing = set(output_root.rglob("summary.csv")) if output_root.exists() else set()
        log_path = Path(entry["log_path"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"[{index}/{total}] 开始 | 因素: {entry['factor_name']} | "
            f"条件: {entry['condition_name']} | 覆盖: "
            f"{json.dumps(entry['overrides'], ensure_ascii=False)}",
            flush=True,
        )
        print(f"日志: {log_path}", flush=True)
        entry["status"] = "running"
        entry["started_at"] = datetime.now().isoformat(timespec="seconds")
        _write_json_atomic(manifest_path, manifest)
        started = time.monotonic()
        with log_path.open("w", encoding="utf-8") as log_stream:
            result = process_runner(
                entry["command"],
                cwd=REPOSITORY_ROOT,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                text=True,
            )
        current = set(output_root.rglob("summary.csv")) if output_root.exists() else set()
        new_summaries = sorted(current - existing)
        entry["finished_at"] = datetime.now().isoformat(timespec="seconds")
        entry["duration_seconds"] = round(time.monotonic() - started, 3)
        entry["exit_code"] = int(result.returncode)
        if result.returncode == 0 and len(new_summaries) == 1:
            entry["status"] = "succeeded"
            entry["summary_csv"] = str(new_summaries[0])
            entry["output_dir"] = str(new_summaries[0].parent)
            print(
                f"[{index}/{total}] 状态: 成功 | 耗时: "
                f"{entry['duration_seconds']:.3f}s",
                flush=True,
            )
            print(f"summary.csv: {new_summaries[0]}", flush=True)
        else:
            failed = True
            entry["status"] = "failed"
            entry["error"] = (
                f"exit code {result.returncode}; expected one new summary.csv, "
                f"found {len(new_summaries)}"
            )
            print(
                f"[{index}/{total}] 状态: 失败 | 退出码: {result.returncode} | "
                f"耗时: {entry['duration_seconds']:.3f}s",
                flush=True,
            )
            print(f"日志: {log_path}", flush=True)
        _write_json_atomic(manifest_path, manifest)
        if failed and not continue_on_error:
            break
    return 1 if failed else 0
