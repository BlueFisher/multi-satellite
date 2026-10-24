"""推断回合数据的采集、派生与持久化。"""

from __future__ import annotations

import csv
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit

from environment.visualization.utils import eci2lvlh


COMMUNICATION_DIAGNOSTIC_FIELDS = (
    "communication_edges",
    "message_kl",
    "message_reconstruction_error",
    "message_mean",
    "message_std",
    "message_l2_norm",
    "message_min",
    "message_max",
)


@dataclass(frozen=True)
class PhysicalSnapshot:
    """一个仿真时刻的统一物理状态。"""

    names: tuple[str, ...]
    state_eci: np.ndarray
    state_lvlh: np.ndarray
    orbital_elements: np.ndarray
    distance_matrix_km: np.ndarray


def to_jsonable(value: Any) -> Any:
    """递归转换 NumPy 等对象，使其可由标准 JSON 编码器处理。"""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, set):
        return [to_jsonable(item) for item in sorted(value, key=str)]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    value_type = type(value)
    return {"__type__": f"{value_type.__module__}.{value_type.__qualname__}"}


def derive_physical_state(
    eci_by_name: Mapping[str, np.ndarray], reference_name: str
) -> PhysicalSnapshot:
    """从全部卫星的 ECI 状态计算 LVLH、六根数和距离矩阵。"""

    names = tuple(eci_by_name)
    if not names:
        raise ValueError("ECI snapshot contains no satellites")
    if reference_name not in eci_by_name:
        raise KeyError(
            f"Reference satellite {reference_name!r} is missing from ECI snapshot"
        )

    state_eci = np.stack(
        [np.asarray(eci_by_name[name], dtype=float) for name in names]
    )
    if state_eci.shape[1:] != (6,):
        raise ValueError(f"Expected satellite states shaped (6,), got {state_eci.shape}")

    reference_state = np.asarray(eci_by_name[reference_name], dtype=float)
    state_lvlh = np.stack(
        [eci2lvlh(state, reference_state) for state in state_eci]
    )
    # 避免参考星因浮点运算留下极小的非零量。
    state_lvlh[names.index(reference_name)] = 0.0

    orbital_elements = []
    for state in state_eci:
        orbit = Orbit.from_vectors(
            Earth,
            state[:3] * u.km,
            state[3:] * u.km / u.s,
        )
        # 兼容项目实际环境中的 poliastro 0.7 以及 requirements 中的
        # 0.17；旧版没有 Orbit.classical()，但两版均公开以下属性。
        orbital_elements.append(
            [
                orbit.a.to_value(u.km),
                float(getattr(orbit.ecc, "value", orbit.ecc)),
                orbit.inc.to_value(u.deg),
                orbit.raan.to_value(u.deg),
                orbit.argp.to_value(u.deg),
                orbit.nu.to_value(u.deg),
            ]
        )

    position_delta = state_eci[:, None, :3] - state_eci[None, :, :3]
    distance_matrix = np.linalg.norm(position_delta, axis=-1)

    return PhysicalSnapshot(
        names=names,
        state_eci=state_eci,
        state_lvlh=state_lvlh,
        orbital_elements=np.asarray(orbital_elements, dtype=float),
        distance_matrix_km=distance_matrix,
    )


def classify_outcome(
    info: Mapping[str, Any], reached_step_limit: bool, truncated: bool = False
) -> tuple[bool, str]:
    """将 2v2/3v1 的任务信息映射为统一结果。"""

    if info.get("all_intercept_claimed"):
        return True, "intercept_success"
    if info.get("capture_reward_claimed"):
        return True, "capture_success"
    if info.get("collided"):
        return False, "collision"
    if info.get("is_hvt_breach"):
        return False, "hvt_breach"
    if info.get("hvt_unsafe"):
        return False, "hvt_unsafe"
    if info.get("out_of_boundary") or info.get("out_satellite"):
        return False, "out_of_boundary"
    if reached_step_limit or truncated:
        return False, "timeout"
    return False, "terminated"


class EpisodeDataRecorder:
    """在纯推断过程中保存逐回合分析数据。"""

    def __init__(
        self,
        output_dir: str | os.PathLike[str],
        env_name: str,
        alg_name: str,
        model_path: str,
        seed: int,
        update_interval: float,
        on_plane: bool,
        limit_accel: bool,
        scenario_parameters: Mapping[str, Any] | None = None,
        baseline_name: str | None = None,
        inference_batch: str | None = None,
        arguments: Mapping[str, Any] | None = None,
        environment_parameters: Mapping[str, Any] | None = None,
        communication_diagnostics_required: bool = False,
        message_dim: int | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.env_name = env_name
        self.alg_name = alg_name
        self.model_path = str(model_path)
        self.seed = int(seed)
        self.update_interval = float(update_interval)
        self.on_plane = bool(on_plane)
        self.limit_accel = bool(limit_accel)
        self.scenario_parameters = to_jsonable(dict(scenario_parameters or {}))
        self.baseline_name = baseline_name
        self.inference_batch = inference_batch
        self.arguments = to_jsonable(dict(arguments or {}))
        self.environment_parameters = to_jsonable(
            dict(environment_parameters or {})
        )
        self.communication_diagnostics_required = bool(
            communication_diagnostics_required
        )
        if self.communication_diagnostics_required and (
            self.env_name != "SatelliteEnv3v1Hvt"
            or self.alg_name != "SharedMessageMATD3"
            or self.on_plane is not False
            or self.limit_accel is not True
        ):
            raise ValueError(
                "communication_diagnostics_required is supported only when "
                "env_name='SatelliteEnv3v1Hvt', "
                "alg_name='SharedMessageMATD3', on_plane=False, and "
                "limit_accel=True; received "
                f"env_name={self.env_name!r}, alg_name={self.alg_name!r}, "
                f"on_plane={self.on_plane!r}, limit_accel={self.limit_accel!r}"
            )
        if self.communication_diagnostics_required and (
            isinstance(message_dim, bool)
            or not isinstance(message_dim, int)
            or message_dim <= 0
        ):
            raise ValueError(
                "message_dim must be a positive integer when communication "
                "diagnostics are required"
            )
        self.message_dim = message_dim
        self._active = False
        self._csv_fieldnames: list[str] | None = None

    def begin_episode(
        self,
        episode_index: int,
        env: Any,
        observations: Mapping[str, np.ndarray],
    ) -> None:
        """开始一个回合并记录 reset 后的初始物理状态。"""

        if self._active:
            raise RuntimeError("Cannot begin a new episode before finishing the current one")

        self._episode_index = int(episode_index)
        self._agent_names = tuple(observations)
        self._key_distance_pairs = self._resolve_key_distance_pairs(env)
        self._state_eci: list[np.ndarray] = []
        self._state_lvlh: list[np.ndarray] = []
        self._orbital_elements: list[np.ndarray] = []
        self._distance_matrices: list[np.ndarray] = []
        self._observations: list[np.ndarray] = []
        self._next_observations: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._rewards: list[np.ndarray] = []
        self._terminations: list[bool] = []
        self._truncations: list[bool] = []
        if self.communication_diagnostics_required:
            self._communication_diagnostics = {
                field: [] for field in COMMUNICATION_DIAGNOSTIC_FIELDS
            }
        self._latest_info: Mapping[str, Any] = {}
        self._active = True
        self._append_snapshot(env)

    def record_transition(
        self,
        observations: Mapping[str, np.ndarray],
        actions: Mapping[str, np.ndarray],
        rewards: Mapping[str, float] | list[float] | np.ndarray,
        next_observations: Mapping[str, np.ndarray],
        termination: bool,
        truncation: bool,
        info: Mapping[str, Any],
        env: Any,
        communication_diagnostics: Mapping[str, np.ndarray] | None = None,
    ) -> None:
        """记录一次动作、奖励及动作后的物理状态。"""

        if not self._active:
            raise RuntimeError("No active inference episode")

        validated_diagnostics = self._validate_communication_diagnostics(
            communication_diagnostics
        )
        observation_array = self._stack_agents(observations, "observations")
        next_observation_array = self._stack_agents(
            next_observations, "next_observations"
        )
        action_array = self._stack_agents(actions, "actions")
        if isinstance(rewards, Mapping):
            reward_array = np.asarray(
                [rewards[name] for name in self._agent_names], dtype=float
            )
        else:
            reward_array = np.asarray(rewards, dtype=float).reshape(-1)
        if reward_array.shape != (len(self._agent_names),):
            raise ValueError(
                "Reward count does not match controlled agents: "
                f"{reward_array.shape} != {(len(self._agent_names),)}"
            )
        termination_value = bool(termination)
        truncation_value = bool(truncation)
        info_snapshot = dict(info)
        reference_name, physical_snapshot = self._collect_snapshot(env)

        self._observations.append(observation_array)
        self._next_observations.append(next_observation_array)
        self._actions.append(action_array)
        self._rewards.append(reward_array)
        if validated_diagnostics is not None:
            for field, value in validated_diagnostics.items():
                self._communication_diagnostics[field].append(value)
        self._terminations.append(termination_value)
        self._truncations.append(truncation_value)
        self._latest_info = info_snapshot
        self._append_collected_snapshot(reference_name, physical_snapshot)

    def build_episode_arrays(
        self, final_info: Mapping[str, Any] | None = None
    ) -> dict[str, np.ndarray]:
        """构建即将写入 NPZ 的无对象数组。"""

        if not self._active:
            raise RuntimeError("No active inference episode")

        self._validate_episode_alignment()
        transition_count = len(self._rewards)
        reward_array = self._stack_transition_list(
            self._rewards, (len(self._agent_names),), float
        )
        observations = self._stack_transition_list(
            self._observations,
            (len(self._agent_names), 0),
            float,
        )
        next_observations = self._stack_transition_list(
            self._next_observations,
            (len(self._agent_names), 0),
            float,
        )
        actions = self._stack_transition_list(
            self._actions,
            (len(self._agent_names), 0),
            float,
        )
        info = self._latest_info if final_info is None else final_info
        metadata = {
            "episode": self._episode_index,
            "env_name": self.env_name,
            "alg_name": self.alg_name,
            "model_path": self.model_path,
            "seed": self.seed,
            "update_interval_s": self.update_interval,
            "on_plane": self.on_plane,
            "limit_accel": self.limit_accel,
            "baseline_name": self.baseline_name,
            "inference_batch": self.inference_batch,
            "output_dir": str(self.output_dir.resolve()),
            "arguments": self.arguments,
            "environment_parameters": self.environment_parameters,
            "scenario_parameters": self.scenario_parameters,
            "state_fields": ["x", "y", "z", "vx", "vy", "vz"],
            "state_units": ["km", "km", "km", "km/s", "km/s", "km/s"],
            "orbital_element_fields": [
                "a_km",
                "ecc",
                "inc_deg",
                "raan_deg",
                "argp_deg",
                "nu_deg",
            ],
            "reference_satellite": self._reference_name,
            "array_axes": {
                "state_eci": ["time", "satellite", "state_component"],
                "state_lvlh": ["time", "satellite", "state_component"],
                "orbital_elements": [
                    "time",
                    "satellite",
                    "orbital_element",
                ],
                "distance_matrix_km": [
                    "time",
                    "satellite_from",
                    "satellite_to",
                ],
                "observations": [
                    "transition",
                    "agent",
                    "observation_component",
                ],
                "next_observations": [
                    "transition",
                    "agent",
                    "observation_component",
                ],
                "actions": ["transition", "agent", "action_component"],
                "rewards": ["transition", "agent"],
                "cumulative_rewards": ["transition", "agent"],
                "terminations": ["transition"],
                "truncations": ["transition"],
            },
        }
        arrays = {
            "step": np.arange(transition_count, dtype=np.int64),
            "time_s": np.arange(transition_count + 1, dtype=float)
            * self.update_interval,
            "satellite_names": np.asarray(self._satellite_names),
            "agent_names": np.asarray(self._agent_names),
            "state_lvlh": np.stack(self._state_lvlh),
            "state_eci": np.stack(self._state_eci),
            "orbital_elements": np.stack(self._orbital_elements),
            "distance_matrix_km": np.stack(self._distance_matrices),
            "observations": observations,
            "next_observations": next_observations,
            "actions": actions,
            "rewards": reward_array,
            "cumulative_rewards": np.cumsum(reward_array, axis=0),
            "terminations": np.asarray(self._terminations, dtype=bool),
            "truncations": np.asarray(self._truncations, dtype=bool),
            "episode_info_json": np.asarray(
                json.dumps(to_jsonable(info), ensure_ascii=False)
            ),
        }
        if self.communication_diagnostics_required:
            metadata.update(
                {
                    "communication_diagnostic_fields": list(
                        COMMUNICATION_DIAGNOSTIC_FIELDS
                    ),
                    "communication_edge_axes": [
                        "receiver_agent",
                        "sender_agent",
                    ],
                    "communication_edges_include_self": True,
                    "message_statistics_source": "post_gate_sampled_message",
                    "message_std_ddof": 0,
                    "message_dim": self.message_dim,
                    "message_kl_reduction": (
                        "sum over message dimensions; no reduction over time or agents"
                    ),
                    "message_reconstruction_reduction": (
                        "mean over action dimensions; no reduction over time or agents"
                    ),
                }
            )
            arrays.update(
                {
                    field: self._stack_transition_list(
                        self._communication_diagnostics[field],
                        (
                            (len(self._agent_names), len(self._agent_names))
                            if field == "communication_edges"
                            else (len(self._agent_names),)
                        ),
                        bool if field == "communication_edges" else float,
                    )
                    for field in COMMUNICATION_DIAGNOSTIC_FIELDS
                }
            )
        arrays["metadata_json"] = np.asarray(
            json.dumps(to_jsonable(metadata), ensure_ascii=False)
        )
        return arrays

    def finish_episode(
        self,
        final_info: Mapping[str, Any],
        reached_step_limit: bool = False,
    ) -> Path:
        """原子写入 NPZ，并将回合汇总追加到 CSV。"""

        if not self._active:
            raise RuntimeError("No active inference episode")

        target = self.output_dir / f"episode_{self._episode_index:04d}.npz"
        temporary = self.output_dir / (
            f".{target.stem}.{uuid.uuid4().hex}.tmp.npz"
        )
        try:
            arrays = self.build_episode_arrays(final_info)
            np.savez_compressed(temporary, **arrays)
            os.replace(temporary, target)
            self._append_summary(target, arrays, final_info, reached_step_limit)
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"Failed to save inference episode {self._episode_index} to {target}"
            ) from exc
        finally:
            self._active = False
        return target

    def _stack_agents(
        self, values: Mapping[str, np.ndarray], field_name: str
    ) -> np.ndarray:
        missing = [name for name in self._agent_names if name not in values]
        if missing:
            raise KeyError(f"{field_name} missing agents: {missing}")
        return np.stack(
            [np.asarray(values[name], dtype=float).reshape(-1) for name in self._agent_names]
        )

    def _validate_communication_diagnostics(
        self, diagnostics: Mapping[str, np.ndarray] | None
    ) -> dict[str, np.ndarray] | None:
        """校验严格通信诊断，确保一个 transition 只会追加完整的一组数据。"""

        if not self.communication_diagnostics_required:
            return None
        if diagnostics is None:
            raise ValueError(
                "communication_diagnostics is required when strict communication "
                "diagnostics are enabled"
            )

        actual_fields = set(diagnostics)
        required_fields = set(COMMUNICATION_DIAGNOSTIC_FIELDS)
        missing = [
            field for field in COMMUNICATION_DIAGNOSTIC_FIELDS if field not in actual_fields
        ]
        unexpected = sorted(actual_fields - required_fields)
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing fields: {missing}")
            if unexpected:
                details.append(f"unexpected fields: {unexpected}")
            raise ValueError(
                "communication_diagnostics must contain exactly the required fields; "
                + "; ".join(details)
            )

        agent_count = len(self._agent_names)
        validated: dict[str, np.ndarray] = {}
        for field in COMMUNICATION_DIAGNOSTIC_FIELDS:
            expected_shape = (
                (agent_count, agent_count)
                if field == "communication_edges"
                else (agent_count,)
            )
            dtype = bool if field == "communication_edges" else float
            value = np.array(diagnostics[field], dtype=dtype, copy=True)
            if value.shape != expected_shape:
                raise ValueError(
                    f"{field} expected shape {expected_shape}, got {value.shape}"
                )
            validated[field] = value
        return validated

    @staticmethod
    def _stack_transition_list(
        values: list[np.ndarray], empty_shape: tuple[int, ...], dtype: Any
    ) -> np.ndarray:
        if values:
            return np.stack(values).astype(dtype, copy=False)
        return np.empty((0,) + empty_shape, dtype=dtype)

    def _validate_episode_alignment(self) -> None:
        transition_count = len(self._rewards)
        transition_lengths = {
            "observations": len(self._observations),
            "next_observations": len(self._next_observations),
            "actions": len(self._actions),
            "rewards": transition_count,
            "terminations": len(self._terminations),
            "truncations": len(self._truncations),
        }
        if self.communication_diagnostics_required:
            transition_lengths.update(
                {
                    field: len(self._communication_diagnostics[field])
                    for field in COMMUNICATION_DIAGNOSTIC_FIELDS
                }
            )

        snapshot_count = transition_count + 1
        snapshot_lengths = {
            "state_eci": len(self._state_eci),
            "state_lvlh": len(self._state_lvlh),
            "orbital_elements": len(self._orbital_elements),
            "distance_matrix_km": len(self._distance_matrices),
        }
        mismatches = [
            f"{field}={length}, expected {transition_count}"
            for field, length in transition_lengths.items()
            if length != transition_count
        ]
        mismatches.extend(
            f"{field}={length}, expected {snapshot_count}"
            for field, length in snapshot_lengths.items()
            if length != snapshot_count
        )
        if mismatches:
            raise RuntimeError(
                "Episode alignment invariant violated: " + "; ".join(mismatches)
            )

    def _append_snapshot(self, env: Any) -> None:
        reference_name, snapshot = self._collect_snapshot(env)
        self._append_collected_snapshot(reference_name, snapshot)

    def _collect_snapshot(self, env: Any) -> tuple[str, PhysicalSnapshot]:
        try:
            eci_by_name = env.simulator.get_all_satellite_states_eci()
            reference_name = getattr(env, "hvt_name", None)
            if reference_name not in eci_by_name:
                reference_name = getattr(
                    env.simulator, "origin_satellite_name", None
                )
            if reference_name not in eci_by_name:
                raise RuntimeError("Cannot resolve HVT/origin reference satellite")
            snapshot = derive_physical_state(eci_by_name, reference_name)
            if self._state_eci and snapshot.names != self._satellite_names:
                raise RuntimeError("Satellite order changed during inference episode")
            return reference_name, snapshot
        except Exception as exc:
            raise RuntimeError(
                f"Failed to collect physical state for episode {self._episode_index}"
            ) from exc

    def _append_collected_snapshot(
        self, reference_name: str, snapshot: PhysicalSnapshot
    ) -> None:
        self._reference_name = reference_name
        self._satellite_names = snapshot.names
        self._state_eci.append(snapshot.state_eci)
        self._state_lvlh.append(snapshot.state_lvlh)
        self._orbital_elements.append(snapshot.orbital_elements)
        self._distance_matrices.append(snapshot.distance_matrix_km)

    @staticmethod
    def _resolve_key_distance_pairs(env: Any) -> tuple[tuple[str, str], ...]:
        pairs: list[tuple[str, str]] = []
        interceptors = list(getattr(env, "interceptor_names", []))
        attackers = list(getattr(env, "attacker_names", []))
        if interceptors and attackers:
            pairs.extend((left, right) for left in interceptors for right in attackers)
        else:
            attacker = getattr(env, "attacker_name", None)
            if attacker:
                pairs.extend((name, attacker) for name in getattr(env, "agent_names", []))
                attackers = [attacker]

        hvt = getattr(env, "hvt_name", None)
        if hvt:
            pairs.extend((attacker, hvt) for attacker in attackers)

        # 保持确定顺序，同时移除重复项和无效自距离。
        return tuple(dict.fromkeys(pair for pair in pairs if pair[0] != pair[1]))

    def _append_summary(
        self,
        npz_path: Path,
        arrays: Mapping[str, np.ndarray],
        final_info: Mapping[str, Any],
        reached_step_limit: bool,
    ) -> None:
        rewards = arrays["rewards"]
        truncated = bool(arrays["truncations"][-1]) if len(rewards) else False
        success, reason = classify_outcome(
            final_info, reached_step_limit, truncated=truncated
        )
        row: dict[str, Any] = {
            "episode": self._episode_index,
            "npz_file": npz_path.name,
            "steps": len(rewards),
            "duration_s": len(rewards) * self.update_interval,
            "success": success,
            "termination_reason": reason,
            "episode_total_reward": float(rewards.sum()),
            "global_min_distance_km": self._global_min_distance(
                arrays["distance_matrix_km"]
            ),
        }
        for index, name in enumerate(self._agent_names):
            total = float(rewards[:, index].sum()) if len(rewards) else 0.0
            mean = float(rewards[:, index].mean()) if len(rewards) else 0.0
            row[f"reward_total__{name}"] = total
            row[f"reward_mean__{name}"] = mean

        name_to_index = {
            name: index for index, name in enumerate(self._satellite_names)
        }
        distances = arrays["distance_matrix_km"]
        for left, right in self._key_distance_pairs:
            if left not in name_to_index or right not in name_to_index:
                continue
            values = distances[:, name_to_index[left], name_to_index[right]]
            row[f"distance_min_km__{left}__{right}"] = float(values.min())
            row[f"distance_final_km__{left}__{right}"] = float(values[-1])

        csv_path = self.output_dir / "summary.csv"
        fieldnames = list(row)
        if self._csv_fieldnames is None:
            if csv_path.exists() and csv_path.stat().st_size:
                with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                    self._csv_fieldnames = next(csv.reader(handle))
            else:
                self._csv_fieldnames = fieldnames
        if fieldnames != self._csv_fieldnames:
            raise RuntimeError("Summary CSV columns changed within one inference run")

        write_header = not csv_path.exists() or csv_path.stat().st_size == 0
        with csv_path.open("a", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._csv_fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _global_min_distance(distance_matrices: np.ndarray) -> float:
        satellite_count = distance_matrices.shape[-1]
        if satellite_count < 2:
            return float("nan")
        mask = ~np.eye(satellite_count, dtype=bool)
        return float(distance_matrices[:, mask].min())
