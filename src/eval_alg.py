import os
import time
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np

from Evaluation import EpisodeDataRecorder, Evaluator
from Evaluation.draw import convert_to_lvlh, plot_lvlh_trajectories
from utils.Logger import Logger
from utils.Logger.LoggerCallback import BaseLoggerCallback
from utils.Logger.LoggerConfig import LoggerMode


EVAL_ENVIRONMENT_PARAMETER_KEYS = (
    "action_scale",
    "max_episode_steps",
    "obs_add_last_action",
    "obs_add_distance",
    "intercept_time",
    "capture_sustain_steps",
    "intercept_distance_threshold_km",
    "capture_distance_threshold_km",
    "win_max_angle",
    "hvt_safe_distance",
    "outside_threshold",
    "collision_threshold",
    "target_sector_angle_deg",
    "target_sector_radius_km",
    "attacker_max_accel",
)


def format_duration(seconds: float) -> str:
    """将耗时格式化为适合终端进度显示的紧凑文本。"""

    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{remaining_seconds:02d}s"
    return f"{minutes}m{remaining_seconds:02d}s"


def requires_communication_diagnostics(cfg: Namespace) -> bool:
    """仅为指定的 SharedMessage 推断基准启用严格通信诊断。"""

    return (
        cfg.env_name == "SatelliteEnv3v1Hvt"
        and cfg.alg_name == "SharedMessageMATD3"
        and cfg.on_plane is False
        and cfg.limit_accel is True
    )


def derive_baseline_name(
    env_name: str,
    on_plane: bool,
    limit_accel: bool,
    run_name: str | None = None,
) -> str:
    """返回不依赖算法和观测开关的基准场景名称。"""

    if run_name:
        return str(run_name)
    return (
        f"{env_name}_on_plane_{str(bool(on_plane)).lower()}"
        f"_limit_accel_{str(bool(limit_accel)).lower()}"
    )


def collect_environment_parameters(cfg: Namespace, env) -> dict:
    """收集配置与已构造环境中实际存在的分析参数。"""

    parameters = {}
    for key in EVAL_ENVIRONMENT_PARAMETER_KEYS:
        if hasattr(cfg, key):
            parameters[key] = getattr(cfg, key)
        elif hasattr(env, key):
            parameters[key] = getattr(env, key)
    return parameters


def allocate_eval_output_dir(
    output_root,
    env_name: str,
    on_plane: bool,
    limit_accel: bool,
    timestamp: str,
    run_name: str | None = None,
) -> Path:
    """原子创建不覆盖历史推断数据的运行目录。"""

    baseline_name = derive_baseline_name(
        env_name,
        on_plane,
        limit_accel,
        run_name=run_name,
    )
    base_dir = Path(output_root) / baseline_name
    directory_name = timestamp
    base_dir.mkdir(parents=True, exist_ok=True)
    for suffix in range(10000):
        name = (
            directory_name
            if suffix == 0
            else f"{directory_name}-{suffix:02d}"
        )
        candidate = base_dir / name
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"cannot allocate unique evaluation directory under {base_dir}")


def generate_image(traj_dict: Dict, ref: str, output_dir: str, eci=False) -> None:
    """
    生成LVLH轨迹图像。

    参数：
        traj_dict (Dict): 包含轨迹数据的字典，键为卫星名称，值为轨迹数据。
        ref (str): 参考卫星的名称，用于转换为LVLH坐标系。
        output_dir (str): 输出图像的保存目录。
        eci (bool, 可选): 是否使用ECI坐标系，默认为False。

    示例：
        >>> traj_dict = {
        ...     "sat1": [[0, 0, 0], [1, 1, 1]],
        ...     "sat2": [[0, 0, 0], [-1, -1, -1]]
        ... }
        >>> generate_image(traj_dict, ref="sat1", output_dir="./output")
    """
    # 转换为LVLH坐标系
    lvlh_dict = convert_to_lvlh(traj_dict, ref) if eci else traj_dict
    # 检查轨迹范围，自动调整视图大小
    max_range = 0
    for key, traj in lvlh_dict.items():
        lvlh_dict[key] = np.array(traj)
        max_val = np.max(np.abs(lvlh_dict[key][:, :2]))
        if max_val > max_range:
            max_range = max_val

    # 设置视图大小为最大值的1.5倍，至少0.1km
    view_size = max(0.1, max_range * 1.5)
    print(f"Auto-adjusted view size: ±{view_size:.2f} km")

    # 生成LVLH轨迹图像
    plot_lvlh_trajectories(lvlh_dict, view_size=view_size, output_dir=output_dir)


def eval_func(
    eval: Evaluator,
    cfg: Namespace,
    logger: Logger,
    logger_callback: BaseLoggerCallback = None,
):
    """
    执行评估过程。

    参数：
        eval (Evaluator): 评估器实例，用于执行评估。
        cfg (Namespace): 配置对象，包含评估相关参数。
        logger (Logger): 日志记录器实例。
        logger_callback (BaseLoggerCallback, 可选): 日志回调对象，默认为None。

    示例：
        >>> from argparse import Namespace
        >>> cfg = Namespace(eval_episode=10, max_episode_steps=1000, render=False)
        >>> eval_func(eval, cfg, logger)
    """
    logger.set_mode(LoggerMode.EVAL)
    recorder = None
    if cfg.just_eval:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        baseline_name = derive_baseline_name(
            cfg.env_name,
            cfg.on_plane,
            cfg.limit_accel,
            run_name=getattr(cfg, "eval_run_name", None),
        )
        output_root = getattr(
            cfg, "eval_data_dir", os.path.join("output", "eval_data")
        )
        output_dir = allocate_eval_output_dir(
            output_root,
            cfg.env_name,
            cfg.on_plane,
            cfg.limit_accel,
            timestamp,
            run_name=getattr(cfg, "eval_run_name", None),
        )
        environment_parameters = collect_environment_parameters(
            cfg, eval.env.unwrapped
        )
        communication_diagnostics_required = requires_communication_diagnostics(cfg)

        recorder = EpisodeDataRecorder(
            output_dir=output_dir,
            env_name=cfg.env_name,
            alg_name=cfg.alg_name,
            model_path=getattr(cfg, "load_path", ""),
            seed=cfg.seed,
            update_interval=eval.env.unwrapped.simulator.update_interval,
            on_plane=cfg.on_plane,
            limit_accel=cfg.limit_accel,
            scenario_parameters=environment_parameters,
            baseline_name=baseline_name,
            inference_batch=output_dir.name,
            arguments=vars(cfg),
            environment_parameters=environment_parameters,
            communication_diagnostics_required=communication_diagnostics_required,
            message_dim=(
                cfg.hidden_size if communication_diagnostics_required else None
            ),
        )
        logger.info(f"推断回合数据输出路径: {output_dir}")

    show_progress = bool(cfg.just_eval)
    evaluation_started = time.monotonic() if show_progress else None
    for ep in range(cfg.eval_episode):
        episode_started = time.monotonic() if show_progress else None
        episode_rewards, info = eval.eval_episode(
            cfg.max_episode_steps,
            render=cfg.render,
            recorder=recorder,
            episode_index=ep,
        )
        if show_progress:
            episode_elapsed = time.monotonic() - episode_started
            evaluation_elapsed = time.monotonic() - evaluation_started
            actual_steps = max(
                (len(rewards) for rewards in episode_rewards.values()), default=0
            )
            print(
                f"推断回合 {ep + 1}/{cfg.eval_episode} | 步数 {actual_steps} | "
                f"本回合耗时 {format_duration(episode_elapsed)} | "
                f"累计耗时 {format_duration(evaluation_elapsed)}",
                flush=True,
            )

        logger.add_episode_rewards(episode_rewards=episode_rewards)
        logger.step()

        if logger_callback is not None:
            logger_callback.after_eval_episode(episode_rewards, info)

        if cfg.just_eval and hasattr(cfg, "generate_image") and cfg.generate_image:
            generate_image(
                info["trail"],
                cfg.origin_satellite,
                os.path.join(cfg.image_dir, str(ep)),
            )

    if logger_callback is not None:
        logger_callback.on_eval_log_interval(cfg.eval_episode)

    logger.set_mode(LoggerMode.TRAIN)
