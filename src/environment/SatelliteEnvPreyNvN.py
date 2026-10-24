from collections import deque
from typing import List

import gymnasium.spaces as spaces
import numpy as np

from .SatelliteEnvBase import SatelliteEnvBase
from .dynamics.GeneratorFactory import GeneratorType
from .visualization.SimulatorFactory import SimulatorType
from .utils import (
    get_plane_normal,
    distance,
    project_point_to_plane,
    create_scenario,
)


class SatelliteEnvPreyNvN(SatelliteEnvBase):
    """
    支持 NVN (多拦截者 vs 多逃逸者) 的卫星围捕环境。
    """

    def __init__(
        self,
        # -----------环境基础设置------------
        render_mode=None,
        m_max_episode_steps=1000,
        update_interval: int = 2,
        max_trail_length: int = 1000,
        sampling_interval: int = 5,
        action_scale: float = 0.001,
        simulator: SimulatorType = SimulatorType.Local,
        start_on_init: bool = True,
        generator: GeneratorType = GeneratorType.CW,
        config_name="nvn_scenario.yaml",  # 假设有一个适配NVN的配置文件
        outside_threshold=50,
        collision_threshold=1,
        obs_window_size=2,
        action_window_size=1,
        reward_computer_type="RewardComputerPreyNvN",
        # -----------奖励与任务设置------------
        capture_distance_threshold_km: float = 10.0,
        on_plane=True,
        agent_heuristic=False,
        opponent_heuristic=True,  # 是否启用对手的简单避障策略
        use_diff_obs=True,  # 观测是否使用相对坐标
        attacker_max_accel=0.8,  # 逃逸者最大加速度
        *args,
        **kwargs,
    ):
        self.capture_distance_threshold_km = capture_distance_threshold_km
        self.opponent_heuristic = opponent_heuristic
        self.agent_heuristic = agent_heuristic
        self.use_diff_obs = use_diff_obs
        self.attacker_max_accel = attacker_max_accel

        # 状态追踪
        self.captured_evaders = set()  # 记录已被捕获的逃逸者ID
        self.prev_total_distance = 0.0  # 用于计算奖励塑形
        self.obs_window_size = obs_window_size
        self.action_window_size = action_window_size
        # 调用父类构造函数
        super().__init__(
            render_mode=render_mode,
            m_max_episode_steps=m_max_episode_steps,
            update_interval=update_interval,
            max_trail_length=max_trail_length,
            sampling_interval=sampling_interval,
            action_scale=action_scale,
            simulator=simulator,
            start_on_init=start_on_init,
            generator=generator,
            config_name=config_name,
            outside_threshold=outside_threshold,
            collision_threshold=collision_threshold,
            on_plane=on_plane,
            reward_type=reward_computer_type,
        )

        self._setup_obs_action_queue()

    def _setup_satellite(self) -> List[str]:
        """
        初始化卫星，支持 N 个拦截者和 M 个逃逸者
        """
        if self.scenario_config.get("interceptor_prefix"):
            self.interceptor_prefix = self.scenario_config["interceptor_prefix"]
        else:
            self.interceptor_prefix = "Inter"

        if self.scenario_config.get("attacker_prefix"):
            self.attacker_prefix = self.scenario_config["attacker_prefix"]
        else:
            self.attacker_prefix = "Evader"

        # 使用 create_scenario 创建场景，它应该返回创建的数量
        # 假设 create_scenario 逻辑能处理 yaml 中的 num_agents 和 num_attackers
        agent_count, attacker_count = create_scenario(
            self.simulator, self.scenario_config
        )

        # 生成名称列表
        self.interceptor_names = [
            f"{self.interceptor_prefix}_{i}" for i in range(agent_count)
        ]
        self.attacker_names = [
            f"{self.attacker_prefix}_{i}" for i in range(attacker_count)
        ]

        # HVT (High Value Target) 在 NVN 纯围捕中可能不需要，
        # 如果需要保护目标，可以在此添加，这里暂定为纯对抗
        self.hvt_name = self.scenario_config.get("hvt_name", None)

        if self.on_plane and len(self.interceptor_names) >= 3:
            points = [
                self.simulator.get_satellite_position(name)
                for name in self.interceptor_names[:3]
            ]
            self.plane_normal = get_plane_normal(points[0], points[1], points[2])
            self.plane_point = points[0]
        elif self.on_plane:
            # Fallback for < 3 agents
            self.plane_normal = np.array([0, 0, 1])
            self.plane_point = np.array([0, 0, 0])

        # 返回所有需要模拟器追踪的卫星名称
        all_sats = self.interceptor_names + self.attacker_names
        if self.hvt_name:
            all_sats.append(self.hvt_name)
        return all_sats

    def _setup_obs_action_queue(self):
        """
        设置观测和动作的历史队列
        """
        self.old_obs_queue = {
            name: deque(maxlen=self.obs_window_size) for name in self.interceptor_names
        }

        # 计算单步观测维度
        # 自身(6) + 所有逃逸者(6*M) + 其他队友(6*(N-1))
        num_interceptors = len(self.interceptor_names)
        num_attackers = len(self.attacker_names)

        self.obs_per_step_dim = self.satellite_obs_dim * (
            1 + num_attackers + (num_interceptors - 1)
        )

        for _ in range(self.obs_window_size):
            for name in self.interceptor_names:
                self.old_obs_queue[name].append(np.zeros(self.obs_per_step_dim))

        self.old_action_queue = {
            name: deque(maxlen=self.action_window_size)
            for name in self.interceptor_names
        }

        for _ in range(self.action_window_size):
            for name in self.interceptor_names:
                self.old_action_queue[name].append(np.zeros(3))

    def _setup_observation_space(self):
        self.satellite_obs_dim = 6  # Pos(3) + Vel(3)

        num_interceptors = len(self.interceptor_names)
        num_attackers = len(self.attacker_names)

        # 维度 = 历史观测 + 历史动作
        obs_dim = (
            self.satellite_obs_dim
            * (num_interceptors + num_attackers)
            * self.obs_window_size
        ) + (3 * self.action_window_size)

        self.observation_space = spaces.Dict(
            {
                name: spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
                for name in self.interceptor_names
            }
        )
        self.agent_names = self.interceptor_names  # 兼容父类

    def _setup_action_space(self):
        self.action_space = spaces.Dict(
            {
                name: spaces.Box(-1, 1, shape=(3,), dtype=np.float32)
                for name in self.interceptor_names
            }
        )

    def _get_obs(self) -> dict[str, np.ndarray]:
        """
        获取 NVN 场景的观测。
        对于每个智能体，观测包括：
        1. 自身状态
        2. 所有逃逸者的状态 (相对自身)
        3. 所有队友的状态 (相对自身)
        """
        # 1. 获取所有卫星的原始状态
        states = {}
        all_names = self.interceptor_names + self.attacker_names

        for name in all_names:
            raw_state = self.simulator.get_satellite_state(name)
            # 简单的归一化: 位置/50km, 速度保持或微调
            norm_state = raw_state.copy()
            norm_state[:3] /= 50.0
            states[name] = norm_state

        obs_ret = {}

        for agent_name in self.interceptor_names:
            current_obs = []

            # A. 自身状态
            my_state = states[agent_name]
            current_obs.append(my_state)

            # B. 逃逸者状态
            for adv_name in self.attacker_names:
                adv_state = states[adv_name]
                if self.use_diff_obs:
                    # 使用相对状态 (Pos_adv - Pos_me, Vel_adv - Vel_me)
                    rel_state = adv_state - my_state
                    current_obs.append(rel_state)
                else:
                    current_obs.append(adv_state)

            # C. 队友状态
            for other_name in self.interceptor_names:
                if other_name == agent_name:
                    continue
                other_state = states[other_name]
                if self.use_diff_obs:
                    rel_state = other_state - my_state
                    current_obs.append(rel_state)
                else:
                    current_obs.append(other_state)

            # 拼接当前步的特征
            step_obs = np.concatenate(current_obs)

            # 更新队列
            self.old_obs_queue[agent_name].append(step_obs)

            # 展平历史观测 + 历史动作
            flat_obs = np.concatenate(list(self.old_obs_queue[agent_name]))
            flat_actions = np.concatenate(list(self.old_action_queue[agent_name]))

            obs_ret[agent_name] = np.concatenate([flat_obs, flat_actions])

        return obs_ret

    def _before_update(self):
        """
        在物理更新前，计算对手（逃逸者）的控制策略
        """
        if self.opponent_heuristic:
            # 简单的启发式：逃离最近的拦截者
            interceptor_orbits = [
                self.simulator.get_satellite(name) for name in self.interceptor_names
            ]

            for att_name in self.attacker_names:
                # 如果已被捕获，可能停止移动或移除，这里假设继续挣扎
                att_state = self.simulator.get_satellite(att_name)
                accel = self._calculate_evasion_acceleration(
                    att_state, interceptor_orbits
                )
                self._add_accel(att_name, accel)

    def _calculate_evasion_acceleration(
        self, evader_state, interceptor_orbits
    ) -> np.ndarray:
        """
        计算 NVN 中的逃逸策略
        """
        evader_pos = evader_state[:3]

        # 1. 寻找最近的拦截者
        min_dist = float("inf")
        nearest_interceptor_pos = None

        for int_orbit in interceptor_orbits:
            int_pos = int_orbit[:3]
            dist = np.linalg.norm(evader_pos - int_pos)
            if dist < min_dist:
                min_dist = dist
                nearest_interceptor_pos = int_pos

        if nearest_interceptor_pos is None:
            return np.zeros(3)

        # 2. 计算远离向量
        escape_vec = evader_pos - nearest_interceptor_pos
        if min_dist > 1e-6:
            escape_dir = escape_vec / min_dist
        else:
            escape_dir = np.random.randn(3)
            escape_dir /= np.linalg.norm(escape_dir)

        # 3. 简单的势场避障：距离越近，推力越大
        # 在远处时稍微向原点回归以防飞太远（可选）
        origin_pull = -0.05 * (evader_pos / (np.linalg.norm(evader_pos) + 1e-6))

        escape_accel = escape_dir * self.attacker_max_accel

        total_accel = escape_accel

        # 限制范围
        total_accel = np.clip(total_accel, -1.0, 1.0)
        return total_accel

    def _is_outside(self):
        # 检查所有拦截者是否出界
        outsiders = []
        for name in self.interceptor_names:
            pos = self.simulator.get_satellite_position(name)
            if np.linalg.norm(pos) > self.outside_threshold:
                outsiders.append(name)
        return outsiders

    def _is_done(self):
        # 1. 碰撞结束
        if self.collided:
            return True

        # 2. 所有逃逸者被捕获
        if len(self.captured_evaders) >= len(self.attacker_names):
            return True

        # 3. 任何拦截者出界（视任务严格程度而定，这里设为 True）
        if len(self.out_satellite) > 0:
            return True

        return False

    def _get_info(self) -> dict:
        return {
            "captured_count": len(self.captured_evaders),
            "collided": self.collided,
            "current_step": self.current_step,
            "out_satellite": self.out_satellite,
        }

    def _after_reset(self):
        self.captured_evaders = set()
        self.prev_total_distance = 0.0

        # 重置时重新计算一次初始距离和
        dists = np.zeros((len(self.interceptor_names), len(self.attacker_names)))
        for i, int_name in enumerate(self.interceptor_names):
            for j, att_name in enumerate(self.attacker_names):
                pos_i = self.simulator.get_satellite_position(int_name)
                pos_j = self.simulator.get_satellite_position(att_name)
                dists[i, j] = distance(pos_i, pos_j)

        self.prev_total_distance = np.sum(np.min(dists, axis=0))

        self._setup_obs_action_queue()

    def _add_accel(self, name: str, accel: np.ndarray):
        # 兼容父类方法
        if self.on_plane:
            assert self.plane_normal is not None
            assert self.plane_point is not None
            accel = project_point_to_plane(accel, self.plane_normal, self.plane_point)

        if name in self.interceptor_names:
            self.old_action_queue[name].append(accel)

        self.simulator.add_accel(name, accel * self.action_scale)
