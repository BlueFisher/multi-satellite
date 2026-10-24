import json
import os
import time
from argparse import Namespace
from typing import Dict, List, Tuple

import numpy as np
import torch

from algorithms.Network.Actor.DeterministicActor import DeterministicActor
from algorithms.Network.Actor.IntentPredictActor import IntentDeterministicActor
from environment import SimulatorType
from environment.SatelliteEnv3v1Hvt import SatelliteEnv3v1Hvt
from environment.SatelliteEnvBase import SatelliteEnvBase
from plot import plot_end
from utils.utils_func import load_args

internal_model_path = {
    "3v1": "../PythonSatellite/output/models/SatelliteEnv3v1HvtNoPlane/MATD3_Intent/Intent-20260122-114445/actor_critic.pth"
}


class EnvironmentManager:
    """
    管理仿真环境的类。

    Attributes:
        task_type (str): 任务类型，例如 "3v1" 或 "2v2"。
        env (SatelliteEnvBase): 当前的仿真环境实例。
        simulator: 本地仿真器实例。
        logger_callback: 日志回调实例。
        agent_names (List[str]): 代理名称列表。
        num_agents (int): 代理数量。
        n_actions (List[int]): 每个代理的动作维度列表。
        n_obs (List[int]): 每个代理的观测维度列表。
        env_kwargs (dict): 环境初始化参数。
    """

    def __init__(
        self,
        task_type: str,
        update_interval: int,
        action_scale: float,
        collision_threshold: float,
        m_max_episode_steps: int,
        args: Namespace,
    ):
        self.task_type = task_type
        self.env = None
        self.simulator = None
        self.logger_callback = None
        self.agent_names: List[str] = []
        self.num_agents = 0
        self.n_actions: List[int] = []
        self.n_obs: List[int] = []

        self.env_kwargs = {
            "render_mode": None,
            "update_interval": update_interval,
            "action_scale": action_scale * update_interval,
            "collision_threshold": 0.005,
            "agent_heuristic": False,
            "opponent_heuristic": True,
            "m_max_episode_steps": 2000,
            "on_plane": False,
            "simulator": SimulatorType.Local,
            "start_on_init": True,
        }
        if hasattr(args, "config_name"):
            self.env_kwargs.update({"config_name": args.config_name})

        self._init_environment()

    def _init_environment(self):
        """
        根据任务类型初始化对应环境及组件。

        Raises:
            ValueError: 如果任务类型无效。

        Example:
            >>> manager = EnvironmentManager("3v1", 10, 1.0, 0.005, 100, args)
            >>> manager._init_environment()
        """
        if self.task_type == "3v1":
            from utils.Logger.LoggerCallback import Satellite3v1HvtLoggerCallback

            self.env_kwargs.update({"use_diff_obs": True, "obs_add_distance": True})
            self.env = SatelliteEnv3v1Hvt(**self.env_kwargs)
            # Logger 可能需要适配，这里保留实例化
            self.logger_callback = Satellite3v1HvtLoggerCallback(
                None, Namespace(eval_episode=1, just_eval=True, debug_mode=True)
            )
        elif self.task_type == "2v2":
            from utils.Logger.LoggerCallback import Satellite2v2LoggerCallback
            from environment.SatelliteEnv2v2 import SatelliteEnv2v2

            self.env_kwargs.update({"intercept_time": 300})
            self.env = SatelliteEnv2v2(**self.env_kwargs)
            self.logger_callback = Satellite2v2LoggerCallback(
                None, Namespace(eval_episode=1, just_eval=True, debug_mode=True)
            )
        else:
            raise ValueError(f"无效的任务类型: {self.task_type}")

        # 提取环境维度信息
        obs_space = self.env.observation_space
        action_space = self.env.action_space
        self.agent_names = list(obs_space.keys())
        self.num_agents = len(self.agent_names)
        self.n_actions = [action_space[name].shape[0] for name in self.agent_names]
        self.n_obs = [obs_space[name].shape[0] for name in self.agent_names]

        # 获取本地仿真器实例
        unwrapped_env = self.env.unwrapped
        assert isinstance(unwrapped_env, SatelliteEnvBase) and hasattr(
            unwrapped_env, "simulator"
        )
        self.simulator = unwrapped_env.simulator

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        重置环境。

        Returns:
            Tuple[Dict[str, np.ndarray], Dict]: 包含观测值和信息的元组。

        Example:
            >>> observations, info = manager.reset()
        """
        return self.env.reset()

    def step(
        self, actions: Dict[str, np.ndarray]
    ) -> Tuple[Dict, Dict, bool, bool, Dict]:
        """
        执行一步动作。

        Args:
            actions (Dict[str, np.ndarray]): 每个代理的动作字典。

        Returns:
            Tuple[Dict, Dict, bool, bool, Dict]: 包含下一个观测值、奖励、是否完成、是否截断和额外信息的元组。

        Example:
            >>> next_obs, rewards, done, truncated, info = manager.step(actions)
        """
        return self.env.step(actions)

    def close(self) -> None:
        """
        关闭环境。

        Example:
            >>> manager.close()
        """
        self.env.close()


class SatelliteController:
    """
    主控制器类，用于管理卫星仿真。

    Attributes:
        env_manager (EnvironmentManager): 环境管理器实例。
        model_path (str): 模型路径。
        device: 设备类型（CPU 或 GPU）。
        actors (List[DeterministicActor]): 模型的 actor 列表。
        running (bool): 控制器运行状态。
    """

    def __init__(self, env_manager: EnvironmentManager):
        self.env_manager = env_manager

        if not env_manager.task_type in internal_model_path:
            raise NotImplementedError(
                f"Task type {env_manager.task_type} is not supported"
            )

        self.model_path = internal_model_path[env_manager.task_type]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actors: List[IntentDeterministicActor] = []
        self.running = True

        # 初始化时直接加载模型
        if not self._load_local_model():
            raise RuntimeError("模型加载失败，无法继续")

    def _load_local_model(self) -> bool:
        """
        加载本地模型权重。

        Returns:
            bool: 如果加载成功返回 True，否则返回 False。

        Example:
            >>> success = controller._load_local_model()
        """
        try:
            print(f"正在加载模型: {self.model_path}")
            all_model = torch.load(self.model_path, map_location=self.device)
            self.actors = []
            for i in range(self.env_manager.num_agents):
                actor = IntentDeterministicActor(
                    input_dim=self.env_manager.n_obs[i],
                    output_dim=self.env_manager.n_actions[i],
                    hidden_dim=128,
                    intent_predictor_count=self.env_manager.num_agents,
                    intent_predictor_output_dim=self.env_manager.n_actions[i],
                )
                # 适配保存的模型 key 格式
                key = f"actor{i}"
                if key in all_model:
                    actor.load_state_dict(all_model[key])
                else:
                    print(
                        f"警告: 模型文件中未找到 {key}，尝试使用 actor_state_dict_{i} 或其他格式"
                    )
                    # 这里可以根据实际模型结构添加兼容代码

                actor.eval()
                self.actors.append(actor)
            print(f"成功加载 {len(self.actors)} 个本地模型")
            return True
        except Exception as e:
            print(f"本地模型加载失败: {str(e)}")
            self.actors = []
            return False

    def _get_model_actions(
        self, observations: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        根据观测值生成模型动作。

        Args:
            observations (Dict[str, np.ndarray]): 每个代理的观测值字典。

        Returns:
            Dict[str, np.ndarray]: 每个代理的动作字典。

        Example:
            >>> actions = controller._get_model_actions(observations)
        """
        if not self.actors:
            return {
                name: np.zeros(self.env_manager.n_actions[i])
                for i, name in enumerate(observations.keys())
            }

        actions = {}
        with torch.no_grad():
            for i, (name, obs) in enumerate(observations.items()):
                obs_tensor = torch.FloatTensor(obs)
                action = self.actors[i](obs_tensor)
                actions[name] = action.numpy()
        return actions

    def shutdown(self):
        """
        关闭控制器。

        Example:
            >>> controller.shutdown()
        """
        self.running = False
        self.actors = []
        print("系统正在关闭...")

    def run_main_loop(self, max_episodes: int):
        """
        主循环：纯本地环境交互。

        Args:
            max_episodes (int): 最大运行轮数。

        Returns:
            list: 保存的仿真数据。

        Example:
            >>> data = controller.run_main_loop(100)
        """
        current_episode = 0
        save_data = []
        simulator = self.env_manager.simulator

        try:
            print(f"开始本地推演，计划运行 {max_episodes} 轮")

            while self.running and current_episode < max_episodes:
                observations, info = self.env_manager.reset()
                episode_step = 0
                print(f"--> 开始第 {current_episode + 1}/{max_episodes} 轮")

                actions = {name: np.zeros(3) for name in self.env_manager.agent_names}

                # 记录开始时间用于计算模拟时间戳
                sim_time = time.time()
                update_interval = self.env_manager.env_kwargs["update_interval"]

                # 单轮交互循环
                while True:
                    save_data_item = simulator.get_formatted_state_data()
                    save_data.append(
                        {
                            "step": episode_step,
                            "data": save_data_item,
                        }
                    )
                    # 2. 决策
                    actions = self._get_model_actions(observations)

                    # 3. 执行一步
                    next_obs, rewards, done, truncated, info = self.env_manager.step(
                        actions
                    )
                    observations = next_obs

                    # 纯后台运行时不需要 render，如果需要调试可取消注释
                    self.env_manager.env.render()

                    episode_step += 1
                    sim_time += update_interval

                    # 4. 检查结束条件
                    # 检查是否获胜/失败/超时
                    max_steps = self.env_manager.env_kwargs["m_max_episode_steps"]
                    is_done = any(done.values()) if isinstance(done, dict) else done
                    is_truncated = (
                        any(truncated.values())
                        if isinstance(truncated, dict)
                        else truncated
                    )

                    # 额外胜利条件检查
                    win_condition = info.get("capture_reward_claimed") or info.get(
                        "all_intercept_claimed"
                    )

                    if episode_step % 200 == 0:
                        print("推演正在进行中, 请耐心等待..")

                    if (
                        episode_step >= max_steps
                        or is_done
                        or is_truncated
                        or win_condition
                    ):
                        print("done break")
                        break

                current_episode += 1

        except KeyboardInterrupt:
            print("\n用户中断程序")
        except Exception as e:
            print(f"运行出错: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self.env_manager.close()
            self.shutdown()

        return save_data


def main():
    args = load_args(Namespace())

    env_manager = EnvironmentManager(
        task_type=args.task_type,
        update_interval=args.update_interval,
        action_scale=args.action_scale,
        collision_threshold=args.collision_threshold,
        m_max_episode_steps=args.m_max_episode_steps,
        args=args,
    )

    # 路径处理
    save_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "save_data"
    )
    os.makedirs(save_data_dir, exist_ok=True)

    # 控制器初始化与运行
    try:
        controller = SatelliteController(env_manager=env_manager)
        data = controller.run_main_loop(1)

        save_path = os.path.join(save_data_dir, "data.json")
        with open(save_path, "w") as f:
            json.dump(
                reformat_simulation_data(data), f
            )  # indent=4 可选，增加可读性但增加文件大小
        plot_end(data)
        # print(f"推演数据已保存于 {os.path.normpath(save_path)}")

    except Exception as e:
        print(f"主程序错误: {e}")


def reformat_simulation_data(time_series_data):
    """
    将 [ {step:0, data:[sat1, sat2]}, {step:1, data:[sat1, sat2]} ]
    转换为 [ {name:sat1, data:[{step:0,...}, {step:1,...}]}, {name:sat2, ...} ]
    """
    # 使用字典临时存储，key为卫星名称
    grouped_data = {}

    for frame in time_series_data:
        # 获取当前帧的时间信息
        current_step = frame.get("step")

        # 遍历当前帧里的所有卫星
        satellites_in_frame = frame.get("data", [])

        for sat in satellites_in_frame:
            name = sat.get("name")

            # 提取位置和速度 (r 和 v)
            # 注意：确保这里使用的是 list 格式，如果是 numpy 需要 .tolist()
            pos = sat.get("r")
            vel = sat.get("v")

            # 构造单步数据包
            step_data_item = {
                "step": current_step * 2,
                "r": [round(i, 6) for i in pos],
                "v": [round(i, 6) for i in vel],
            }

            # 按名称归类
            if name not in grouped_data:
                grouped_data[name] = []

            grouped_data[name].append(step_data_item)

    final_output = []
    for name, trajectory_list in grouped_data.items():
        final_output.append({"name": name, "data": trajectory_list})

    return final_output


if __name__ == "__main__":
    main()
