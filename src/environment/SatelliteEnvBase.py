import abc
import math
import os
from abc import abstractmethod
from typing import List, Union

import yaml

from .rewards.RewardComputerFactory import RewardComputerFactory
from .utils import distance, project_point_to_plane

try:
    import gymnasium as gym
except ModuleNotFoundError:
    import gym

import numpy as np

from .dynamics.GeneratorFactory import GeneratorType, GeneratorFactory
from .visualization.SimulatorFactory import SimulatorFactory, SimulatorType


def validate_target_sector(angle, radius):
    """校验可选的打靶扇区参数，并规范为浮点数。"""

    if angle is None and radius is None:
        return None
    if angle is None or radius is None:
        raise ValueError("target sector angle and radius must be provided together")

    angle = float(angle)
    radius = float(radius)
    if not math.isfinite(angle) or not 0.0 < angle <= 360.0:
        raise ValueError("target sector angle must be in (0, 360]")
    if not math.isfinite(radius) or radius < 20.0:
        raise ValueError("target sector radius must be at least 20 km")
    return angle, radius


class SatelliteEnvBase(gym.Env):
    metadata = {"render.modes": ["human"], "render_modes": ["human"]}

    def __init__(
        self,
        render_mode=None,
        m_max_episode_steps=1000,
        update_interval: int = 2,  # 模拟一步的时间, 单位为秒，默认10s
        max_trail_length: int = 1000,  # 显示轨迹的最大步长，只有 Local 才有效
        sampling_interval: int = 5,  # 显示轨迹采样的大小，只有 Local 才有效
        action_scale: float = 0.001,
        simulator: SimulatorType = SimulatorType.Local,
        start_on_init: bool = True,
        generator: GeneratorType = GeneratorType.F,
        config_name="",
        collision_threshold=1,
        outside_threshold=50,
        on_plane=True,
        reward_type: str = "",
        limit_accel=False,
    ):
        self.collision_threshold = collision_threshold
        self.outside_threshold = outside_threshold
        self.config_name = config_name
        self.scenario_config = self._get_scenario_config()
        self.max_episode_steps = m_max_episode_steps
        self.render_mode = render_mode
        self.action_scale = action_scale
        self.start_on_init = start_on_init

        position_generator = GeneratorFactory.get_generator(generator)
        self.simulator = SimulatorFactory.get_simulator(simulator, position_generator)

        self.simulator.update_interval = update_interval
        self.simulator.max_trail_length = max_trail_length
        self.simulator.sampling_interval = sampling_interval
        self.simulator.render = True if self.render_mode == "human" else False
        self.simulator.limit_accel = limit_accel

        self.limit_accel = limit_accel
        self.reward_computer = RewardComputerFactory.get_reward_computer(reward_type)
        self.reward_computer.env = self

        self.on_plane = on_plane
        self.plane_normal = None
        self.plane_point = None

        self.current_step = 0
        self.collided = False

        self.satellites = self._setup_satellite()

        if start_on_init:
            self.simulator.start()

        self._setup_observation_space()
        self._setup_action_space()
        self._setup_global_state_space()

        self.agent_names = list(self.observation_space.keys())
        self.num_agents = len(self.agent_names)

        self.out_satellite: List[str] = []
        self.collided_satellite: List[str] = []

    @abc.abstractmethod
    def _setup_observation_space(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _setup_action_space(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _setup_satellite(self) -> List[str]:
        raise NotImplementedError

    def _get_scenario_config(self):
        base_path = os.path.dirname(os.path.abspath(__file__))

        file_name = self.config_name
        if not file_name.endswith(".yaml"):
            file_name += ".yaml"

        config_path = os.path.join(base_path, "configs", file_name)

        # 检查路径是否存在
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        # 检查是否为文件
        if not os.path.isfile(config_path):
            raise ValueError(f"指定路径不是文件: {config_path}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            raise RuntimeError(f"读取配置文件失败: {str(e)}") from e

    def _compute_reward(self) -> dict[str, float]:
        return self.reward_computer.compute_reward()

    @abstractmethod
    def _get_obs(self) -> dict[str, np.ndarray[float, (None, 3)]]:
        raise NotImplementedError

    @abc.abstractmethod
    def _is_done(self):
        raise NotImplementedError

    def _setup_global_state_space(self) -> None:
        pass

    def get_global_state(self) -> Union[np.ndarray, None]:
        return None

    def _is_collided(self):
        ret = set()
        for left in self.satellites:
            for right in self.satellites:
                if left == right:
                    continue
                if (
                    self.simulator.get_distance_to_real(left, right)
                    < self.collision_threshold
                ):
                    ret.add(left)
                    ret.add(right)

        return list(ret)

    def _before_update(self):
        pass

    def _after_update(self):
        pass

    @abc.abstractmethod
    def _get_info(self):
        raise NotImplementedError

    def step(self, action):
        for name, accel in action.items():
            self._add_accel(name, accel)

        self._before_update()
        self.simulator.update()
        obs = self._get_obs()
        self.current_step += 1

        self.collided_satellite = self._is_collided()
        self.collided = len(self.collided_satellite) > 0

        self.out_satellite = self._is_outside()

        self._after_update()

        done = self._is_done()
        reward = self._compute_reward()
        truncations = self.current_step >= self.max_episode_steps
        info = self._get_info()

        reward_list = list(reward.values())
        return obs, reward_list, done, truncations, info

    @abc.abstractmethod
    def _is_outside(self):
        raise NotImplementedError

    def _add_accel(self, name: str, accel: np.ndarray[(3,), float]):
        if self.on_plane:
            assert self.plane_normal is not None
            assert self.plane_point is not None
            accel = project_point_to_plane(accel, self.plane_normal, self.plane_point)

        self.simulator.add_accel(name, accel * self.action_scale)

    def render(self, mode="human"):
        if self.render_mode == "human":
            self.simulator.render_frame()

    def close(self):
        self.simulator.stop()

    def _before_reset(self):
        pass

    def _after_reset(self):
        pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._before_reset()

        self.simulator.clear()
        self._setup_satellite()

        self.out_satellite = self._is_outside()
        if len(self.out_satellite) > 0:
            print(f"Warning!! the {self.out_satellite} is outside at first ")

        self.collided = False
        self.collided_satellite = self._is_collided()
        if len(self.collided_satellite) > 0:
            self.collided = True
            print(f"Warning!! the {self.collided_satellite} is collided at first ")

        self.current_step = 0  # 重置时间步
        self._after_reset()
        return self._get_obs(), self._get_info()

    def get_all_satellite_state(self):
        ret = {}
        for name in self.simulator.agent_names():
            ret[name] = self.simulator.get_satellite_state(name)

        return ret
