import json
import os
import time
import atexit
from copy import deepcopy
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import gymnasium as gym

warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")

from utils.Logger.Logger import Logger
from utils.Inference.InferenceEngine import InferenceEngine

import numpy as np
import torch
from environment import SimulatorType, register_env
from environment.SatelliteEnvBase import SatelliteEnvBase
from environment.visualization.FUnitySimulator import FUnitySimulator
from utils.utils_func import (
    load_args,
    update_args,
    load_config,
    is_linux,
)


def _load_remote_config(
    path: str = "../PythonSatellite/save_data/config/remote_inference.yaml",
):
    ret = vars(load_config(path))
    assert ret is not None
    return ret


class EnvironmentManager:
    """
    环境管理器（适配配置化加载）：整合FUnitySimulator配置+配置驱动的环境创建
    """

    def __init__(self, env_kwargs: dict, sync_interval: int = 1):
        self.sync_interval = sync_interval
        self.env = None
        self.simulator = None
        self.logger_callback = None
        self.agent_names: List[str] = []
        self.num_agents = 0
        self.n_actions: List[int] = []
        self.n_obs: List[int] = []
        self.env_kwargs = env_kwargs

        # 补充模拟器关键配置（第一段核心）
        self.env_kwargs["start_on_init"] = False

        self._init_environment()

    def _init_environment(self):
        """初始化gym环境 + FUnitySimulator配置"""
        self.env = gym.make(**self.env_kwargs)

        # 提取环境维度信息
        obs_space = self.env.observation_space
        action_space = self.env.action_space
        self.agent_names = list(obs_space.keys())
        self.num_agents = len(self.agent_names)
        self.n_actions = [action_space[name].shape[0] for name in self.agent_names]
        self.n_obs = [obs_space[name].shape[0] for name in self.agent_names]

        # 初始化并配置FUnitySimulator
        unwrapped_env = self.env.unwrapped
        assert isinstance(unwrapped_env, SatelliteEnvBase) and hasattr(
            unwrapped_env, "simulator"
        )
        self.simulator = unwrapped_env.simulator
        assert isinstance(self.simulator, FUnitySimulator)

        # Simulator核心配置
        self.simulator.set_keep_position(self.simulator.agent_names())
        self.simulator.set_sync_interval(self.sync_interval)

        # 区分己方/敌方智能体
        attacker = list(
            set(self.simulator.agent_names())
            - set(self.agent_names)
            - {getattr(self.env.unwrapped, "hvt_name", "")}
        )
        self.simulator.set_satellite_name(self.agent_names, attacker)

    def set_sync_interval(self, new_interval: float):
        """更新同步间隔"""
        self.sync_interval = new_interval
        if self.simulator:
            self.simulator.set_sync_interval(new_interval)
        logger.info(f"同步间隔已更新为: {new_interval}s")

    def set_keep_position(self, agents: List[str]) -> None:
        """设置保持位置的智能体"""
        self.simulator.set_keep_position(agents)

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict]:
        return self.env.reset()

    def step(
        self, actions: Dict[str, np.ndarray]
    ) -> Tuple[Dict, Dict, bool, bool, Dict]:
        return self.env.step(actions)

    def render(self) -> None:
        self.env.render()

    def close(self) -> None:
        self.env.close()


class SatelliteController:
    """主控制器：配置化算法加载 + 模拟器回调 + 时序控制"""

    def __init__(self, env_manager: EnvironmentManager, algorithms_args: Dict):
        self.env_manager = env_manager

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_connected = False
        self.ready = False
        self.running = True

        remote_config = _load_remote_config()

        assert "local_inference" in remote_config
        local_inference = remote_config.get("local_inference", True)
        del remote_config["local_inference"]

        self.inference = InferenceEngine(
            algorithms_args,
            logger=logger,
            remote_config=remote_config,
            local_inference=local_inference,
        )
        # 注册模拟器回调
        self._register_simulator_callbacks()

    def _register_simulator_callbacks(self):
        """注册模拟器事件回调"""
        sim = self.env_manager.simulator

        assert isinstance(sim, FUnitySimulator)

        sim.on_connect(lambda: logger.info("Unity已连接, 尝试同步数据"))
        sim.on_sync_data(self.on_sync_data)
        sim.on_task_run(self.task_run)
        sim.on_task_stop(self.task_stop)
        sim.on_task_quit(self.task_quit)
        sim.on_sync_interval_change(self.sync_time_change)
        sim.on_server_start(lambda: logger.info("ZMQ服务启动，等待Unity连接"))
        sim.start()

    def on_sync_data(self):
        """模拟器连接回调"""
        self.is_connected = True
        self.ready = False
        self.env_manager.set_keep_position(self.env_manager.simulator.agent_names())
        logger.info(
            f"数据同步完成 - {self.env_manager.simulator.agent_names()}, 等待启动指令"
        )

    def task_run(self, task_type: str):
        """任务启动回调"""
        if not self.is_connected:
            logger.error("未连接到模拟器，无法启动任务")
            return

        self.ready = True
        self.env_manager.set_keep_position([])
        logger.info(f"任务启动 - 类型: {task_type}, 控制方式: 配置化算法")

    def task_stop(self):
        """任务停止回调"""
        self.ready = False
        self.env_manager.set_keep_position(self.env_manager.agent_names)
        logger.info("任务停止")

    def task_quit(self):
        """任务退出回调"""
        if self.is_connected:
            self.is_connected = False
            self.ready = False
            self.env_manager.set_keep_position(self.env_manager.agent_names)
            logger.info("任务已退出 - 等待新连接")

    def sync_time_change(self, time_str: str):
        """同步间隔修改回调"""
        try:
            new_sync_interval = float(time_str)
            self.env_manager.set_sync_interval(new_sync_interval)
        except ValueError:
            logger.info(f"无效的同步间隔值: {time_str}，请提供数字")

    def shutdown(self):
        """关闭资源"""
        self.running = False
        logger.info("系统正在关闭...")

    # ---------- 核心推理方法 ----------
    def _get_alg_actions(
        self, observations: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        if not self.ready:
            return {
                name: np.zeros(self.env_manager.n_actions[i])
                for i, name in enumerate(observations.keys())
            }

        agent_names = list(observations.keys())
        # 构造算法所需的 obs 格式：[[obs_agent0], [obs_agent1], ...]
        obs_list = [[observations[name]] for name in agent_names]

        with torch.no_grad():
            choose_ret = self.inference.choose_action(
                agent_list=agent_names,
                obs=obs_list,
                eval=True,
            )
            actions_tensor = choose_ret["action"]  # (num_agents, action_dim)

            actions = {
                name: actions_tensor[idx].reshape(-1)
                for idx, name in enumerate(agent_names)
            }
        return actions

    def run_main_loop(self, max_episodes: int) -> List[dict]:
        """主循环"""
        current_episode = 0
        max_episodes_per_conn = max_episodes
        break_on_win = True
        finish = False
        save_data = []
        simulator = self.env_manager.simulator

        try:
            while self.running:
                # 等待模拟器连接
                while not self.is_connected and self.running:
                    time.sleep(0.1)

                if not self.running or finish:
                    break

                # 单连接下的轮次控制
                while self.is_connected and current_episode < max_episodes_per_conn:
                    observations, info = self.env_manager.reset()
                    episode_step = 0
                    logger.info(
                        f"开始第 {current_episode + 1} 轮 (就绪状态: {self.ready})"
                    )

                    actions = {
                        name: np.zeros(3) for name in self.env_manager.agent_names
                    }
                    last_loop_time = time.time()
                    start_time = time.time()

                    # 单轮交互
                    while self.is_connected:
                        # 保存状态数据
                        save_data_item = simulator.get_formatted_state_data()
                        timestamp = (
                            start_time + episode_step * simulator.update_interval
                        )
                        save_data.append(
                            {
                                "episode": current_episode,
                                "step": episode_step,
                                "timestamp": round(timestamp, 6),
                                "data": save_data_item,
                            }
                        )

                        # 帧率控制
                        current_sync_interval = self.env_manager.sync_interval
                        if current_sync_interval > 0:
                            current_time = time.time()
                            sleep_time = current_sync_interval - (
                                current_time - last_loop_time
                            )
                            if sleep_time > 0:
                                time.sleep(sleep_time)
                            last_loop_time = time.time()

                        # 生成动作
                        if self.ready:
                            actions = self._get_alg_actions(observations)

                        # 环境步进
                        next_obs, rewards, done, truncated, info = (
                            self.env_manager.step(actions)
                        )
                        observations = next_obs
                        self.env_manager.render()

                        # 终止条件判断
                        if self.ready:
                            episode_step += 1
                            max_steps = self.env_manager.env_kwargs[
                                "m_max_episode_steps"
                            ]
                            if episode_step >= max_steps or (
                                break_on_win
                                and (
                                    info.get("capture_reward_claimed")
                                    or info.get("all_intercept_claimed")
                                )
                            ):
                                break

                    # 日志回调
                    if self.is_connected:
                        current_episode += 1
                        if current_episode >= max_episodes_per_conn:
                            finish = True

        except KeyboardInterrupt:
            logger.warning("用户中断程序")
        finally:
            if self.env_manager.logger_callback:
                self.env_manager.logger_callback.on_eval_log_interval(None)
            self.env_manager.close()
            self.shutdown()
            logger.info("系统已关闭，资源已释放")

        return save_data


def before_exit():
    """退出钩子"""
    os.system("pause")


def main() -> None:
    """主函数"""
    atexit.register(before_exit)
    register_env()  # 注册自定义环境

    # ====================== 配置化加载 ======================
    # 1. 加载默认配置
    this_file_path = Path(__file__)
    python_root_path = this_file_path.parent.parent

    default_config_path = python_root_path / "configs" / "default_config.yaml"

    args = load_config(str(default_config_path.absolute()))
    # 2. 第一次读取命令行参数
    cmd_args_first_pass = load_args(deepcopy(args))

    # 3. 读取算法配置
    # alg_manager = get_alg_manager(alg_name=cmd_args_first_pass.alg_name)
    real_alg_name = cmd_args_first_pass.alg_name

    log_env_name = (
        cmd_args_first_pass.env_name
        if not hasattr(cmd_args_first_pass, "special_env_name")
        else cmd_args_first_pass.special_env_name
    )

    log_alg_name = (
        cmd_args_first_pass.alg_name  # unity 会设置 special_alg_name
        if not hasattr(cmd_args_first_pass, "special_alg_name")
        else cmd_args_first_pass.special_alg_name
    )

    args = update_args(
        args,
        os.path.join(
            python_root_path,
            "configs",
            "alg",
            f"{cmd_args_first_pass.alg_name}.yaml",
        ),
    )

    special_env_args = Namespace()
    special_env_args = update_args(
        special_env_args,
        os.path.join(
            python_root_path,
            "configs",
            "env_config",
            f"{cmd_args_first_pass.env_name}.yaml",
            # 在这个脚本中，Unity一定会传入 env_name
            # 这个env_name指的是最原始的环境比如SatelliteEnv3v1Hvt
            # unity会读取环境的特殊配置，然后传入到args
            # 比如 SatelliteEnv3v1NoPlane 中 OnPlane参数由Unity传入设置为False
        ),
    )

    # 移除冗余参数
    for key in ["task_type", "env_name"]:
        if key in special_env_args.__dict__:
            del special_env_args.__dict__[key]

    # 5. 第二次读取命令行参数（最高优先级）, 这里也会覆盖一些算法特殊的配置参数
    args = load_args(args)

    # 6. 合并命令行参数到环境配置
    # 这里处理的是Unity传入的参数
    # 比如 SatelliteEnv3v1NoPlane中 OnPlane参数由Unity传入设置为False
    for key in special_env_args.__dict__:
        if hasattr(args, key):
            setattr(special_env_args, key, getattr(args, key))

    # 7. 构建环境参数
    env_kwargs = {
        "id": args.env_name,
        "render_mode": None,  # Unity 一定会渲染
        "update_interval": args.update_interval,
        "action_scale": args.action_scale,
        "collision_threshold": args.collision_threshold,
        "agent_heuristic": args.agent_heuristic,
        "opponent_heuristic": args.opponent_heuristic,
        "m_max_episode_steps": args.max_episode_steps,
        "on_plane": args.on_plane,
        "simulator": SimulatorType.FUnity,
        "config_name": args.config_name,
    }

    env_kwargs.update(vars(special_env_args))

    # ====================== 环境/模拟器管理 ======================
    # 初始化环境管理器
    env_manager = EnvironmentManager(env_kwargs=env_kwargs, sync_interval=1)

    # ====================== 算法初始化 ======================
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 设置全局状态
    if not hasattr(args, "use_global_state"):
        args.use_global_state = False
    global_state_shape = (
        0
        if not args.use_global_state
        else env_manager.env.unwrapped.global_state_space.shape[0]
    )

    # 更新算法所需参数
    args.name_agents = env_manager.agent_names
    args.num_agents = env_manager.num_agents
    args.n_actions = env_manager.n_actions
    args.n_obs = env_manager.n_obs
    args.device = "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"

    algorithms_args = {
        "alg_name": real_alg_name,
        "name_agents": args.name_agents,
        "obs_shapes": args.n_obs,
        "global_state_shape": global_state_shape,
        "action_shapes": args.n_actions,
        "args": args,
        "action_continuous": args.continuous,
    }

    start_info()

    mode_name = args.load_path.split("\\")[-1].split("-")[0]
    logger.info(f"开始评估模型")
    logger.info(f"使用环境: {log_env_name}")
    logger.info(f"使用算法: {log_alg_name}")
    logger.info(f"使用模型: {mode_name}")
    logger.info(f"推演轮次: {args.max_episodes}")

    # ====================== 控制器 + 主循环 ======================
    # 初始化控制器
    controller = SatelliteController(
        env_manager=env_manager, algorithms_args=algorithms_args
    )

    # 运行主循环
    data = controller.run_main_loop(args.max_episodes)

    # 保存推演数据
    save_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "save_data"
    )
    os.makedirs(save_data_dir, exist_ok=True)
    save_path = os.path.join(save_data_dir, "data.json")
    with open(save_path, "w") as f:
        json.dump(data, f)
    logger.info(f"推演数据已保存于 {os.path.normpath(save_path)}")


def start_info():
    print("欢迎使用星群智能协同一体化虚实结合训练与演示系统")


if __name__ == "__main__":
    start_info()
    logger = Logger()
    main()
