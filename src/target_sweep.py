"""多条件打靶推断的实验目录、条件生成与安全命令构建。"""

from __future__ import annotations

import json
import math
import os
import queue
import re
import subprocess
import threading
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


SUPPORTED_ENVIRONMENTS = {"SatelliteEnv2v2", "SatelliteEnv3v1Hvt"}
SUPPORTED_SCANS = {"position", "fine_position", "accel", "collision", "duration"}
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODEL_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
MODEL_ARGUMENT_ALLOWLIST = {
    "env_name",
    "alg_name",
    "load_path",
    "obs_add_last_action",
    "obs_add_distance",
    "on_plane",
    "limit_accel",
}
CONDITION_KEY_ALIASES = {
    "target_sector_angle_deg": "a",
    "target_sector_radius_km": "r",
    "attacker_max_accel": "accel",
    "collision_threshold": "collision",
    "intercept_time": "steps",
    "capture_sustain_steps": "steps",
}


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    environment: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class PositionGrid:
    angles_deg: tuple[float | int, ...]
    radii_km: tuple[float | int, ...]


@dataclass(frozen=True)
class ScanConfig:
    position: Mapping[str, PositionGrid]
    fine_position: Mapping[str, PositionGrid]
    accel: tuple[float | int, ...]
    collision: tuple[float | int, ...]
    duration: tuple[int, ...]


@dataclass(frozen=True)
class Condition:
    condition_id: str
    model: ModelSpec
    scan_name: str
    overrides: Mapping[str, Any]


def load_catalog(path: Path) -> tuple[dict[str, ModelSpec], ScanConfig]:
    """读取并校验六模型目录和扫描参数。"""

    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("target sweep catalog must be a mapping")

    raw_models = raw.get("models")
    if not isinstance(raw_models, dict) or not raw_models:
        raise ValueError("target sweep catalog contains no models")

    models: dict[str, ModelSpec] = {}
    for model_id, model_data in raw_models.items():
        if (
            not isinstance(model_id, str)
            or MODEL_ID_PATTERN.fullmatch(model_id) is None
            or ".." in model_id
        ):
            raise ValueError(
                f"model ID {model_id!r} must be a path-safe identifier"
            )
        if not isinstance(model_data, dict):
            raise ValueError(f"model {model_id!r} must be a mapping")
        environment = model_data.get("environment")
        arguments = model_data.get("arguments")
        if environment not in SUPPORTED_ENVIRONMENTS:
            raise ValueError(f"model {model_id!r} has unsupported environment {environment!r}")
        if not isinstance(arguments, dict):
            raise ValueError(f"model {model_id!r} arguments must be a mapping")
        unsupported_arguments = set(arguments) - MODEL_ARGUMENT_ALLOWLIST
        if unsupported_arguments:
            raise ValueError(
                f"model {model_id!r} has unsupported model argument(s): "
                f"{sorted(unsupported_arguments)}"
            )
        if arguments.get("env_name") != environment:
            raise ValueError(f"model {model_id!r} env_name does not match environment")
        for required in ("alg_name", "load_path", "on_plane", "limit_accel"):
            if required not in arguments:
                raise ValueError(f"model {model_id!r} is missing argument {required!r}")
        model_path = Path(str(arguments["load_path"]))
        resolved_model_path = (
            model_path if model_path.is_absolute() else REPOSITORY_ROOT / model_path
        )
        if not resolved_model_path.exists():
            raise FileNotFoundError(
                f"model path for {model_id!r} does not exist: {resolved_model_path}"
            )
        models[str(model_id)] = ModelSpec(
            model_id=str(model_id),
            environment=str(environment),
            arguments=dict(arguments),
        )

    raw_scans = raw.get("scans")
    if not isinstance(raw_scans, dict):
        raise ValueError("target sweep catalog contains no scans")
    scan_config = ScanConfig(
        position=_load_position_grids(raw_scans.get("position"), "position"),
        fine_position=_load_position_grids(
            raw_scans.get("fine_position"), "fine_position"
        ),
        accel=_load_nonnegative_values(raw_scans.get("accel"), "accel"),
        collision=_load_nonnegative_values(
            raw_scans.get("collision"), "collision"
        ),
        duration=_load_duration_values(raw_scans.get("duration")),
    )
    return models, scan_config


def _load_position_grids(raw: Any, scan_name: str) -> dict[str, PositionGrid]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"scan {scan_name!r} must contain environment grids")
    grids: dict[str, PositionGrid] = {}
    for environment, grid in raw.items():
        if environment not in SUPPORTED_ENVIRONMENTS:
            raise ValueError(
                f"scan {scan_name!r} has unsupported environment {environment!r}"
            )
        if not isinstance(grid, dict):
            raise ValueError(f"scan {scan_name!r} grid must be a mapping")
        angles = _load_values(grid.get("angles_deg"), f"{scan_name}.angles_deg")
        radii = _load_values(grid.get("radii_km"), f"{scan_name}.radii_km")
        if any(not math.isfinite(value) or not 0 < value <= 360 for value in angles):
            raise ValueError(
                f"{scan_name} angles must be finite and within (0, 360]"
            )
        if any(not math.isfinite(value) or value < 20 for value in radii):
            raise ValueError(
                f"{scan_name} radii must be finite and at least 20 km"
            )
        grids[environment] = PositionGrid(angles, radii)
    return grids


def _load_values(raw: Any, name: str) -> tuple[float | int, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"scan values for {name!r} must be a non-empty list")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw
    ):
        raise ValueError(f"scan values for {name!r} must be numeric")
    if len(set(raw)) != len(raw):
        raise ValueError(f"duplicate scan values for {name}")
    return tuple(raw)


def _load_nonnegative_values(raw: Any, name: str) -> tuple[float | int, ...]:
    values = _load_values(raw, name)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError(f"{name} values must be finite and non-negative")
    return values


def _load_duration_values(raw: Any) -> tuple[int, ...]:
    values = _load_values(raw, "duration")
    if any(
        not math.isfinite(value) or value <= 0 or not float(value).is_integer()
        for value in values
    ):
        raise ValueError("duration values must be finite positive integers")
    return tuple(int(value) for value in values)


def generate_conditions(
    models: Sequence[ModelSpec], scans: Sequence[str], scan_config: ScanConfig
) -> list[Condition]:
    """按模型和扫描顺序生成确定性的实验条件。"""

    unknown = [scan for scan in scans if scan not in SUPPORTED_SCANS]
    if unknown:
        raise ValueError(f"unknown scans: {unknown}")
    conditions: list[Condition] = []
    condition_ids: set[str] = set()
    for model in models:
        for scan_name in scans:
            for overrides in _generate_overrides(model, scan_name, scan_config):
                condition_id = _condition_id(model.model_id, scan_name, overrides)
                if condition_id in condition_ids:
                    raise ValueError(f"duplicate generated condition ID: {condition_id}")
                condition_ids.add(condition_id)
                conditions.append(
                    Condition(condition_id, model, scan_name, dict(overrides))
                )
    return conditions


def _generate_overrides(
    model: ModelSpec, scan_name: str, scan_config: ScanConfig
) -> list[dict[str, Any]]:
    if scan_name in {"position", "fine_position"}:
        grids = getattr(scan_config, scan_name)
        if model.environment not in grids:
            raise ValueError(
                f"{scan_name} is not configured for {model.environment}"
            )
        grid = grids[model.environment]
        return [
            {
                "target_sector_angle_deg": angle,
                "target_sector_radius_km": radius,
            }
            for angle in grid.angles_deg
            for radius in grid.radii_km
        ]
    if scan_name == "accel":
        return [{"attacker_max_accel": value} for value in scan_config.accel]
    if scan_name == "collision":
        return [{"collision_threshold": value} for value in scan_config.collision]
    if scan_name == "duration":
        parameter = (
            "intercept_time"
            if model.environment == "SatelliteEnv2v2"
            else "capture_sustain_steps"
        )
        return [{parameter: value} for value in scan_config.duration]
    raise ValueError(f"unknown scan: {scan_name}")


def _condition_id(
    model_id: str, scan_name: str, overrides: Mapping[str, Any]
) -> str:
    values = "__".join(
        f"{CONDITION_KEY_ALIASES[key]}-{_format_id_value(value)}"
        for key, value in overrides.items()
    )
    return f"{model_id}__{scan_name}__{values}"


def _format_id_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        value = format(value, "g")
    return str(value).replace("-", "m").replace(".", "p")


def _format_cli_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return format(value, "g")
    return str(value)


def build_inference_command(
    condition: Condition,
    episodes: int,
    output_root: Path,
    python_executable: str,
) -> list[str]:
    """构建只能进入正式纯推断入口的参数列表。"""

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if not python_executable:
        raise ValueError("python executable must not be empty")

    command = [
        str(python_executable),
        "src/main.py",
        "--just_eval",
        "True",
        "--load_model",
        "True",
    ]
    for key, value in condition.model.arguments.items():
        command.extend((f"--{key}", _format_cli_value(value)))
    command.extend(("--eval_episode", str(int(episodes))))
    for key, value in condition.overrides.items():
        command.extend((f"--{key}", _format_cli_value(value)))
    data_root = Path(output_root)
    command.extend(
        (
            "--eval_data_dir",
            str(data_root),
            "--eval_run_name",
            condition.condition_id,
        )
    )
    _validate_inference_command(command)
    return command


def _validate_inference_command(command: Sequence[str]) -> None:
    if len(command) < 6 or command[1] != "src/main.py":
        raise ValueError("target sweep commands must execute src/main.py")
    pairs = dict(zip(command[2::2], command[3::2]))
    if pairs.get("--just_eval") != "True" or pairs.get("--load_model") != "True":
        raise ValueError("target sweep commands must be inference-only")


def create_manifest(
    batch_dir: Path,
    conditions: Sequence[Condition],
    commands: Sequence[Sequence[str]],
    config_path: Path,
) -> dict[str, Any]:
    """为尚未执行的条件构建完整、可恢复的批次清单。"""

    if len(conditions) != len(commands):
        raise ValueError("conditions and commands must have equal length")
    batch_dir = Path(batch_dir)
    entries = []
    for condition, raw_command in zip(conditions, commands):
        command = [str(part) for part in raw_command]
        _validate_inference_command(command)
        data_root = Path(_command_option(command, "--eval_data_dir"))
        run_name = _command_option(command, "--eval_run_name")
        entries.append(
            {
                "condition_id": condition.condition_id,
                "model_id": condition.model.model_id,
                "environment": condition.model.environment,
                "scan_name": condition.scan_name,
                "overrides": dict(condition.overrides),
                "status": "pending",
                "command": command,
                "data_root": str(data_root),
                "requested_output_dir": str(data_root / run_name),
                "log_path": str(batch_dir / "logs" / f"{condition.condition_id}.log"),
                "attempts": 0,
            }
        )
    return {
        "batch_id": batch_dir.name,
        "created_at": _now(),
        "config_path": str(Path(config_path)),
        "conditions": entries,
    }


def create_unique_batch_dir(root: Path, timestamp: str) -> Path:
    """原子创建不覆盖历史批次的时间戳目录。"""

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for suffix in range(10000):
        name = timestamp if suffix == 0 else f"{timestamp}-{suffix:02d}"
        candidate = root / name
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"cannot allocate unique batch directory under {root}")


def _command_option(command: Sequence[str], option: str) -> str:
    try:
        index = command.index(option)
        return str(command[index + 1])
    except (ValueError, IndexError) as exc:
        raise ValueError(f"command is missing {option}") from exc


def write_manifest_atomic(path: Path, manifest: Mapping[str, Any]) -> None:
    """原子写入 JSON 清单，避免中断留下半个文件。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(5):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        temporary.unlink(missing_ok=True)


def select_resume_entries(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """返回尚未成功的条件；成功数据不会被重复生成。"""

    conditions = manifest.get("conditions")
    if not isinstance(conditions, list):
        raise ValueError("manifest conditions must be a list")
    return [entry for entry in conditions if entry.get("status") != "succeeded"]


def run_process_with_live_output(
    command,
    *,
    cwd,
    stdout,
    stderr=subprocess.STDOUT,
    heartbeat_context: str = "",
    heartbeat_interval_s: float = 30.0,
) -> subprocess.CompletedProcess:
    """实时转发子进程输出，并在静默期输出存活心跳。"""

    if heartbeat_interval_s <= 0:
        raise ValueError("heartbeat interval must be positive")

    child_environment = os.environ.copy()
    child_environment["PYTHONUNBUFFERED"] = "1"
    child_environment["PYTHONIOENCODING"] = "utf-8"
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=stderr,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=child_environment,
    )
    if process.stdout is None:
        raise RuntimeError("child process stdout pipe was not created")

    output_queue: queue.Queue = queue.Queue()
    stream_ended = object()

    def read_output() -> None:
        try:
            for line in process.stdout:
                output_queue.put(line)
        finally:
            process.stdout.close()
            output_queue.put(stream_ended)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    started_at = time.monotonic()
    last_output = started_at
    last_heartbeat = started_at

    def emit_output(line: str) -> None:
        stdout.write(line)
        stdout.flush()
        print(line, end="", flush=True)

    def drain_pending_output() -> None:
        while True:
            try:
                pending = output_queue.get_nowait()
            except queue.Empty:
                return
            if pending is stream_ended:
                continue
            with suppress(Exception, KeyboardInterrupt):
                emit_output(pending)

    def shutdown_child() -> None:
        with suppress(OSError):
            if process.poll() is None:
                process.terminate()
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            with suppress(OSError):
                process.kill()
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=5.0)
        reader.join(timeout=1.0)
        drain_pending_output()

    try:
        while True:
            now = time.monotonic()
            remaining = max(0.01, heartbeat_interval_s - (now - last_heartbeat))
            try:
                item = output_queue.get(timeout=min(0.25, remaining))
            except queue.Empty:
                now = time.monotonic()
                if now - last_heartbeat >= heartbeat_interval_s:
                    context = f" | 条件 {heartbeat_context}" if heartbeat_context else ""
                    elapsed = now - started_at
                    silent = now - last_output
                    print(
                        f"仍在运行{context} | 累计耗时 {elapsed:.1f}s | "
                        f"最近输出 {silent:.1f}s 前",
                        flush=True,
                    )
                    last_heartbeat = now
                continue

            if item is stream_ended:
                break
            emit_output(item)
            last_output = time.monotonic()
            last_heartbeat = last_output
    except BaseException:
        shutdown_child()
        raise

    return subprocess.CompletedProcess(command, process.wait())


def run_manifest(
    manifest_path: Path,
    continue_on_error: bool = False,
    dry_run: bool = False,
    process_runner=None,
) -> int:
    """顺序执行清单；仅退出码为零且生成新 summary.csv 才算成功。"""

    manifest_path = Path(manifest_path)
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    entries = select_resume_entries(manifest)

    for entry in entries:
        command = [str(part) for part in entry["command"]]
        _validate_inference_command(command)
        if _command_option(command, "--eval_data_dir") != str(entry["data_root"]):
            raise ValueError(
                f"data root mismatch for condition {entry.get('condition_id')}"
            )
    if dry_run:
        return 0

    use_live_runner = process_runner is None
    selected_runner = (
        run_process_with_live_output if use_live_runner else process_runner
    )
    any_failure = False
    repository_root = Path(__file__).resolve().parents[1]
    for index, entry in enumerate(entries, start=1):
        condition_id = str(entry["condition_id"])
        command = [str(part) for part in entry["command"]]
        data_root = Path(entry["data_root"])
        old_summaries = set(data_root.rglob("summary.csv")) if data_root.exists() else set()
        log_path = Path(entry["log_path"])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        condition_started = time.monotonic()

        entry.update(
            {
                "status": "running",
                "started_at": _now(),
                "finished_at": None,
                "exit_code": None,
                "error": None,
                "attempts": int(entry.get("attempts", 0)) + 1,
            }
        )
        write_manifest_atomic(manifest_path, manifest)
        print(f"[{index}/{len(entries)}] 开始条件: {condition_id}", flush=True)
        print(f"  模型: {entry.get('model_id', '')}", flush=True)
        print(
            "  扫描参数: "
            + json.dumps(entry.get("overrides", {}), ensure_ascii=False),
            flush=True,
        )
        print(f"  日志: {log_path}", flush=True)

        try:
            with log_path.open("a", encoding="utf-8", newline="\n") as log_handle:
                log_handle.write(
                    f"\n===== attempt {entry['attempts']} started {entry['started_at']} =====\n"
                )
                log_handle.flush()
                runner_arguments = {
                    "cwd": repository_root,
                    "stdout": log_handle,
                    "stderr": subprocess.STDOUT,
                }
                if use_live_runner:
                    runner_arguments["heartbeat_context"] = condition_id
                result = selected_runner(command, **runner_arguments)
            return_code = int(result.returncode)
            entry["exit_code"] = return_code
            new_summaries = (
                set(data_root.rglob("summary.csv")) - old_summaries
                if data_root.exists()
                else set()
            )
            if return_code == 0 and len(new_summaries) == 1:
                summary = next(iter(new_summaries))
                entry["status"] = "succeeded"
                entry["summary_csv"] = str(summary)
                entry["output_dir"] = str(summary.parent)
            else:
                entry["status"] = "failed"
                if return_code != 0:
                    entry["error"] = f"inference process exited with code {return_code}"
                else:
                    entry["error"] = (
                        "expected exactly one new summary.csv, "
                        f"found {len(new_summaries)}"
                    )
        except KeyboardInterrupt:
            write_manifest_atomic(manifest_path, manifest)
            raise
        except Exception as exc:
            entry["status"] = "failed"
            entry["error"] = f"runner error: {exc}"
        finally:
            if entry["status"] != "running":
                entry["finished_at"] = _now()
                write_manifest_atomic(manifest_path, manifest)

        condition_elapsed = _format_elapsed(time.monotonic() - condition_started)
        if entry["status"] == "succeeded":
            print(
                f"[{index}/{len(entries)}] 条件成功: {condition_id} | "
                f"退出码: {entry.get('exit_code')} | 总耗时 {condition_elapsed}",
                flush=True,
            )
            print(f"  summary.csv: {entry['summary_csv']}", flush=True)
        else:
            print(
                f"[{index}/{len(entries)}] 条件失败: {condition_id} | "
                f"退出码: {entry.get('exit_code')} | 总耗时 {condition_elapsed}",
                flush=True,
            )
            print(f"  错误: {entry.get('error')}", flush=True)
            print(f"  日志: {log_path}", flush=True)

        if entry["status"] == "failed":
            any_failure = True
            if not continue_on_error:
                break

    return 1 if any_failure else 0


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _format_elapsed(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{remaining_seconds:02d}s"
    return f"{minutes}m{remaining_seconds:02d}s"
