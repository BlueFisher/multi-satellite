from collections import deque
from copy import deepcopy
import math
from typing import List

import gymnasium.spaces as spaces
import numpy as np

from .SatelliteEnvBase import SatelliteEnvBase, validate_target_sector
from .dynamics.GeneratorFactory import GeneratorType
from .visualization.SimulatorFactory import SimulatorType
from .utils import (
    # 4颗卫星在同轨道平面，距离逃逸星20-40KM
    get_plane_normal,
    distance,
    project_point_to_plane,
    create_scenario,
)


class SatelliteEnv3v1Hvt(SatelliteEnvBase):
    def __init__(
        self,
        # -----------环境基础设置------------
        render_mode=None,
        m_max_episode_steps=1000,
        update_interval: int = 2,  # 模拟一步的时间, 单位为秒
        max_trail_length: int = 1000,  # 显示轨迹的最大步长，只有Local 才有效
        sampling_interval: int = 5,  # 显示轨迹采样的大小，只有Local 才有效
        action_scale: float = 0.001,
        simulator: SimulatorType = SimulatorType.Local,
        start_on_init: bool = True,
        generator: GeneratorType = GeneratorType.CW,
        config_name="SatelliteEnv3v1Hvt.yaml",
        outside_threshold=50,
        collision_threshold=1,  # 碰撞安全距离（km）
        obs_window_size=2,
        action_window_size=1,
        # -----------奖励设置------------
        capture_distance_threshold_km: float = 15.0,  # 围捕距离
        capture_sustain_steps: int = 400,  # 持续时间步
        # -----------环境特殊设置------------
        reward_computer_type: str = "RewardComputer3v1",
        agent_heuristic=False,
        opponent_heuristic=True,
        on_plane=True,
        hvt_safe_distance=5.0,
        use_diff_obs=True,
        reward_side=False,
        reward_min_accel=False,
        obs_add_distance=False,
        attacker_max_accel=5.0 / 6.0,
        obs_compatible=False,
        limit_accel=False,
        target_sector_angle_deg=None,
        target_sector_radius_km=None,
        attacker_initial_rho_min_km=None,
        attacker_initial_rho_max_km=None,
    ):
        self._target_sector = validate_target_sector(
            target_sector_angle_deg, target_sector_radius_km
        )
        self._attacker_initial_distance = self._normalize_attacker_initial_distance(
            attacker_initial_rho_min_km, attacker_initial_rho_max_km
        )
        self.obs_compatible = obs_compatible
        self.hvt_safe_distance = hvt_safe_distance
        self.hvt_unsafe = False
        self.use_diff_obs = use_diff_obs
        self.reward_side = reward_side
        self.obs_add_distance = obs_add_distance
        self.update_interval = update_interval

        self.capture_distance_threshold_km = capture_distance_threshold_km
        self.capture_sustain_steps = capture_sustain_steps
        self.reward_computer_type = reward_computer_type
        self.agent_heuristic = agent_heuristic
        self.opponent_heuristic = opponent_heuristic

        self.capture = False
        self.capture_sustain_counter = 0
        self.total_capture_sustain_counter = 0
        self.capture_reward_claimed = False

        self.attacker_max_accel = attacker_max_accel
        self.reward_min_accel = reward_min_accel

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
            limit_accel=limit_accel,
        )

        self._setup_obs_action_queue()
        self.obs = {}

    @staticmethod
    def _apply_target_sector(config, target_sector):
        """返回仅覆盖拦截者出生扇区的独立场景配置。"""

        changed = deepcopy(config)
        interceptors = [
            group
            for group in changed.get("satellite_groups", [])
            if group.get("prefix") == "拦截者"
        ]
        if len(interceptors) != 1:
            raise ValueError("target sector requires exactly one interceptor group")

        angle, radius = target_sector
        interceptors[0]["generation_mode"] = "sector_around_axis"
        interceptors[0]["sector_half_angle_deg"] = angle
        interceptors[0]["sector_distance_range_km"] = [20.0, radius]
        return changed

    @staticmethod
    def _normalize_attacker_initial_distance(minimum, maximum):
        if minimum is None and maximum is None:
            return None
        if minimum is None or maximum is None:
            raise ValueError(
                "attacker initial distance endpoints must both be provided"
            )
        return float(minimum), float(maximum)

    @staticmethod
    def _apply_attacker_initial_distance(config, distance_range):
        """Return an isolated scenario config with an attacker rho override."""

        changed = deepcopy(config)
        if distance_range is None:
            return changed

        minimum, maximum = (float(value) for value in distance_range)
        if (
            not math.isfinite(minimum)
            or not math.isfinite(maximum)
            or minimum < 0
            or minimum > maximum
        ):
            raise ValueError(
                "attacker initial distance must be finite, non-negative, and ordered"
            )

        attacker_prefix = changed.get("attacker_prefix", "攻击者")
        attackers = [
            group
            for group in changed.get("satellite_groups", [])
            if group.get("prefix") == attacker_prefix
        ]
        if len(attackers) != 1:
            raise ValueError(
                "attacker initial distance requires exactly one attacker group"
            )
        attackers[0]["polar_rho_range_km"] = [minimum, maximum]
        return changed

    def _get_scenario_config(self):
        config = super()._get_scenario_config()
        if self._target_sector is not None:
            config = self._apply_target_sector(config, self._target_sector)
        return self._apply_attacker_initial_distance(
            config, self._attacker_initial_distance
        )

    def _setup_obs_action_queue(self):
        self.old_obs_queue = {
            name: deque(maxlen=self.obs_window_size) for name in self.agent_names
        }

        obs_queue_dim = self.satellite_obs_dim * (len(self.agent_names) + 1)
        if self.obs_add_distance:
            obs_queue_dim += len(self.agent_names) + 2

        for _ in range(self.obs_window_size):
            [
                self.old_obs_queue[name].append(np.zeros(obs_queue_dim))
                for name in self.agent_names
            ]

        self.old_action_queue = {
            name: deque(maxlen=self.action_window_size) for name in self.agent_names
        }

        for _ in range(self.action_window_size):
            [
                self.old_action_queue[name].append(np.zeros(3))
                for name in self.agent_names
            ]

    def _is_capture_step(self):
        err = self.capture_distance_threshold_km * 0.1
        valid_agents = []

        # 获取attacker位置
        attacker = self.simulator.get_satellite_position(self.attacker_name)

        # 收集所有在距离阈值范围内的agent（最多3个）
        for name in self.agent_names:
            agent = self.simulator.get_satellite_position(name)
            dis = distance(attacker, agent)
            if (
                self.capture_distance_threshold_km - err
                <= dis
                <= self.capture_distance_threshold_km + err
            ):
                valid_agents.append(agent)

        return len(valid_agents) >= 3

    def _is_capture(self):
        if self.simulator.get_distance_to_real(self.attacker_name, self.hvt_name) < 20:
            return False
        return self.capture_sustain_counter >= self.capture_sustain_steps

    def _is_done(self):
        if self.collided:
            return True

        if self.capture_reward_claimed:
            return True

        if len(self.out_satellite) != 0:
            return True

        if self.hvt_unsafe:
            return True

        return False

    def _is_collided(self):
        ret = super()._is_collided()

        if len(ret) > 0:
            return ret

        # 这里需要考虑和 HVT 的碰撞
        ret = []
        for name in self.agent_names:
            if (
                self.simulator.get_distance_to_real(name, self.hvt_name)
                < self.collision_threshold
            ):
                ret.append(name)

        return ret

    def _is_outside(self):
        ret = []
        for name in self.agent_names:
            if (
                self.simulator.get_distance_to_real(name, self.attacker_name)
                > self.outside_threshold
            ):
                ret.append(name)
                continue

        return ret

    def _is_hvt_unsafe(self):
        return (
            self.simulator.get_distance_to_real(self.hvt_name, self.attacker_name)
            < self.hvt_safe_distance
        )

    def _before_update(self):
        # 获取卫星轨道信息
        evader_state = self.simulator.get_satellite(self.attacker_name)
        agent_orbit = [self.simulator.get_satellite(name) for name in self.agent_names]

        # 打印当前位置和速度
        # print(f"逃逸者位置: {evader_orbit.r}, 速度: {evader_orbit.v}")
        if self.opponent_heuristic:
            # 计算逃逸加速度
            accel = self._calculate_evasion_acceleration(evader_state, agent_orbit)
            # 应用加速度
            self._add_accel(self.attacker_name, accel)
        if self.agent_heuristic:
            for name in self.agent_names:
                accel = self._calculate_agent_acceleration(evader_state, name)
                self._add_accel(name, accel)

    def _after_update(self):
        self.hvt_unsafe = self._is_hvt_unsafe()
        last_capture = self.capture
        self.capture = False
        if self._is_capture_step():
            self.capture = True
            self.capture_sustain_counter += 1
            self.total_capture_sustain_counter += 1
            if self._is_capture():
                self.capture_reward_claimed = True
        if last_capture and not self.capture:
            self.capture_sustain_counter = 0

    def _calculate_agent_acceleration(self, evader_state, agent_name) -> np.ndarray:
        """
        计算agent卫星的加速度，确保输出范围在[-1, 1]
        目标是围捕evader并阻止其靠近HVT
        """
        agent = self.simulator.get_satellite(agent_name)
        evader = self.simulator.get_satellite(self.attacker_name)

        # 提取位置和速度信息
        agent_pos = agent[:3]
        agent_vel = agent[3:6]
        evader_pos = evader[:3]
        evader_vel = evader[3:6]
        hvt_pos = np.array([0.0, 0.0, 0.0])  # LVLH坐标系中HVT位于原点

        # 计算关键距离
        dist_to_evader = self.simulator.get_distance_to(agent_name, self.attacker_name)
        evader_dist_to_hvt = self.simulator.get_distance_to(
            self.attacker_name, self.hvt_name
        )

        # 规则1: 拦截evader - 移动到evader前往HVT路径上的前方位置
        if evader_dist_to_hvt < 1e-6:
            evader_dir_to_hvt = np.zeros(3)
        else:
            evader_dir_to_hvt = (
                hvt_pos - evader_pos
            ) / evader_dist_to_hvt  # 指向HVT的方向

        # 计算拦截点
        interception_distance = min(evader_dist_to_hvt * 0.3, 15.0)
        target_pos = evader_pos + evader_dir_to_hvt * interception_distance

        # 计算到拦截点的方向
        pos_diff = target_pos - agent_pos
        dist_to_target = np.linalg.norm(pos_diff)
        if dist_to_target < 1e-6:
            dir_to_target = np.zeros(3)
        else:
            dir_to_target = pos_diff / dist_to_target

        # 调整拦截强度以适应[-1, 1]范围
        interception_strength = 0.1 + (1.0 - min(evader_dist_to_hvt / 100.0, 1.0)) * 0.2
        interception_accel = interception_strength * dir_to_target

        # 规则2: 与其他agent保持围捕队形
        formation_accel = np.zeros(3)
        formation_radius = 15.0  # 围捕时与evader保持的距离
        neighbors = 0

        # 计算每个agent应处的位置，形成均匀包围圈
        for other_agent_name in self.agent_names:
            if other_agent_name == agent_name:
                continue

            other_agent = self.simulator.get_satellite(other_agent_name)
            other_agent_pos = other_agent[:3]

            # 计算期望的相对位置
            num_agents = len(self.agent_names)
            agent_index = self.agent_names.index(agent_name)
            angle = 2 * np.pi * agent_index / num_agents

            # 基于角度计算相对于evader的期望位置
            desired_rel_pos = np.array(
                [
                    formation_radius * np.cos(angle),
                    formation_radius * np.sin(angle),
                    0.0,
                ]
            )
            desired_abs_pos = evader_pos + desired_rel_pos

            # 计算到达期望位置所需的加速度
            pos_error = desired_abs_pos - agent_pos
            formation_strength = 0.08  # 降低强度以避免超出范围
            formation_accel += (
                formation_strength * pos_error / (np.linalg.norm(pos_error) + 1e-6)
            )
            neighbors += 1

        if neighbors > 0:
            formation_accel /= neighbors

        # 规则3: 与其他agent的碰撞规避
        collision_avoidance_accel = np.zeros(3)
        collision_threshold = 10.0

        for other_agent_name in self.agent_names:
            if other_agent_name == agent_name:
                continue

            other_agent = self.simulator.get_satellite(other_agent_name)
            other_agent_pos = other_agent[:3]
            pos_diff = agent_pos - other_agent_pos
            dist_to_other = np.linalg.norm(pos_diff)

            if dist_to_other < collision_threshold and dist_to_other > 1e-6:
                # 调整排斥强度以适应范围
                repulsion_strength = (
                    0.8 * (collision_threshold - dist_to_other) / collision_threshold
                )
                collision_avoidance_accel += repulsion_strength * (
                    pos_diff / dist_to_other
                )

        # 组合各项加速度
        accel = (
            1.0 * collision_avoidance_accel
            + 0.6 * interception_accel
            + 0.3 * formation_accel
        )

        # 先限制整体大小
        max_accel = 1.0  # 明确设置最大加速度为1.0
        current_accel_mag = np.linalg.norm(accel)

        if current_accel_mag > max_accel:
            accel = accel * (max_accel / current_accel_mag)

        # 关键修正：对每个分量进行裁剪，确保严格在[-1, 1]范围内
        accel = np.clip(accel, -1.0, 1.0)

        return accel

    def _calculate_evasion_acceleration(self, evader_state, agent_orbits) -> np.ndarray:
        # 在LVLH坐标系中，HVT位置固定为原点(0,0,0)
        evader_pos = evader_state[:3]  # 提取LVLH位置坐标
        hvt_pos = np.array([0.0, 0.0, 0.0])  # LVLH下HVT永远在原点

        # 计算与HVT的距离和方向（指向HVT即原点）
        dist_to_hvt = self.simulator.get_distance_to_real(
            self.attacker_name, self.hvt_name
        )
        if dist_to_hvt < 1e-6:
            dir_to_hvt = np.zeros(3)
        else:
            dir_to_hvt = (hvt_pos - evader_pos) / dist_to_hvt  # 指向原点的方向

        # 规则1: 靠近HVT（在LVLH下即向原点移动）
        target_dist_to_hvt = 5.0  # 目标距离HVT的距离(km)
        dist_error = dist_to_hvt - target_dist_to_hvt
        approach_strength = 0.12  # 降低强度以适应新范围

        # 根据距离误差计算靠近加速度
        if dist_error > 0:  # 距离大于目标，需要靠近
            approach_accel = approach_strength * dist_error * dir_to_hvt
        else:  # 距离小于目标，轻微远离
            approach_accel = approach_strength * dist_error * 0.2 * dir_to_hvt

        # 规则2: 躲避其他agents
        evade_accel = np.zeros(3)
        min_agent_dist = float("inf")  # 最近的agent距离
        active_agents = 0
        collision_accel = np.zeros(3)
        safety_distance = 10.0  # 碰撞安全距离(km)

        # 遍历所有agent
        for i, name in enumerate(self.agent_names):
            agent_orbit = agent_orbits[i]
            agent_pos = agent_orbit[:3]  # 提取位置部分

            # 计算与agent的距离和远离方向
            pos_diff = evader_pos - agent_pos
            dist_to_agent = np.linalg.norm(pos_diff)
            if dist_to_agent < 1e-6:  # 避免除以零
                dir_from_agent = np.zeros(3)
            else:
                dir_from_agent = pos_diff / dist_to_agent  # 远离agent的方向

            # 更新最近agent距离
            if dist_to_agent < min_agent_dist:
                min_agent_dist = dist_to_agent

            # 计算躲避加速度（30km内的agent都纳入计算）
            if dist_to_agent < 30.0:
                active_agents += 1
                # 距离随距离调整躲避强度（越近强度越大）
                evade_strength = 0.7 / (1.0 + max(0, dist_to_agent - 20.0))
                evade_accel += evade_strength * dir_from_agent

            # 近距离碰撞风险规避（小于安全距离10km）
            if dist_to_agent < safety_distance:
                collision_strength = 0.9 * safety_distance / max(1e-6, dist_to_agent)
                collision_accel += collision_strength * dir_from_agent

        # 归一化常规躲避加速度
        if active_agents > 0:
            evade_accel = evade_accel / active_agents
            # 合并常规躲避和碰撞规避加速度
            total_evade_accel = evade_accel + collision_accel
        else:
            total_evade_accel = np.zeros(3)

        # 根据距离区间确定加速度组合策略
        if dist_to_hvt > 30.0:
            # 1. HVT 30km以外：只考虑靠近HVT
            accel = approach_accel
        else:
            # 2. HVT 30~15km：混合靠近和躲避，根据agent距离调整权重
            if min_agent_dist < 15.0:
                # 4. agent靠近15km内：只考虑躲避
                accel = total_evade_accel
            elif min_agent_dist < 20.0:
                # 3. agent靠近15km内：增加躲避权重
                accel = 0.05 * approach_accel + 0.95 * total_evade_accel
            else:
                # 无近距离agent：基础混合比例
                accel = 0.2 * approach_accel + 0.8 * total_evade_accel

        # 速度增量转换与限制
        max_accel = self.attacker_max_accel  # 最大速度增量限制
        current_accel_mag = np.linalg.norm(accel)
        if current_accel_mag > max_accel:
            accel = accel * (max_accel / current_accel_mag)
        return accel

    def _setup_observation_space(self):
        # 获取所有智能体名称
        all_agent_names = self.simulator.agent_names()

        # 计算攻击者数量（减去HVT，而不是evader）
        num_agents = len(
            [
                name
                for name in all_agent_names
                if name.startswith(self.interceptor_prefix)
            ]
        )

        self.satellite_obs_dim = 6  # 每个卫星的基本观测维度（位置+速度）

        if not self.obs_add_distance:
            # 没有距离信息时的观测维度
            # 每个时间步的观测包括：自身状态(6) + evader状态(6) + 其他智能体状态(6*(num_agents-1))
            obs_per_step = self.satellite_obs_dim * (
                num_agents + 1
            )  # +1 是因为包括evader
            obs_dim = obs_per_step * self.obs_window_size + 3 * self.action_window_size
        else:
            # 有距离信息时的观测维度
            # 每个时间步的观测包括：
            # - 自身状态(6)
            # - evader状态(6)
            # - 其他智能体状态(6*(num_agents-1))
            # - 自身到evader的距离(1)
            # - 自身到原点的距离(1)
            # - 自身到其他智能体的距离(num_agents-1)
            # - evader到原点的距离(1)
            base_obs_dim = self.satellite_obs_dim * (num_agents + 1)  # 基础观测维度
            distance_obs_dim = 2 + (num_agents - 1) + 1  # 距离相关观测维度
            obs_per_step = base_obs_dim + distance_obs_dim
            obs_dim = obs_per_step * self.obs_window_size + 3 * self.action_window_size

        self.observation_space = spaces.Dict(
            {
                name: spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
                for name in all_agent_names
                if name.startswith(self.interceptor_prefix)
            }
        )

    def _setup_action_space(self):
        self.action_space = spaces.Dict(
            {
                name: spaces.Box(-1, 1, shape=(3,), dtype=np.float32)
                for name in self.simulator.agent_names()
                if name.startswith(self.interceptor_prefix)
            }
        )

    def _add_accel(self, name: str, accel: np.ndarray[(3,), np.float32]):
        # if self.on_plane:
        #     assert self.plane_normal is not None
        #     assert self.plane_point is not None
        #     accel = project_point_to_plane(accel, self.plane_normal, self.plane_point)
        # self.simulator.add_accel(name, accel * self.action_scale)

        super()._add_accel(name, accel)

        if name in self.agent_names:
            self.old_action_queue[name].append(accel)

    def _get_obs_compatible(self) -> dict[str, np.ndarray[np.float32, (None, 3)]]:
        # 获取敌方目标
        evader_state = self.simulator.get_satellite_state(self.attacker_name)
        evader_pos = evader_state[:3]  # 提取位置信息
        evader_dis = np.linalg.norm(evader_pos) / 50
        evader_state[:3] = evader_pos / 50

        agent_lvlh = {}
        agent_positions = {}  # 存储归一化前的位置信息用于距离计算

        # 获取所有智能体的状态并存储位置信息
        for name in self.agent_names:
            agent_state = self.simulator.get_satellite_state(name)
            agent_pos = agent_state[:3]
            agent_positions[name] = agent_pos  # 存储归一化前的位置
            agent_state[:3] = agent_pos / 50  # 距离按照 50 km 进行归一化处理
            agent_lvlh[name] = agent_state
            if self.use_diff_obs:  # 使用和目标的位置速度差作为实际观察
                agent_lvlh[name] = agent_lvlh[name] - evader_state

        obs_now = {name: [] for name in self.agent_names}

        for name in self.agent_names:
            # 记录基本观察信息
            obs_now[name].append(agent_lvlh[name])
            obs_now[name].append(evader_state)

            for other in self.agent_names:
                if other == name:
                    continue
                obs_now[name].append(agent_lvlh[other])

            # 只有在 obs_add_distance 为 True 时才添加距离信息
            if self.obs_add_distance:
                # 获取当前智能体的归一化前位置
                agent_pos = agent_positions[name]

                # 1. 当前智能体到evader的距离
                agent_to_evader_dist = np.linalg.norm(agent_pos - evader_pos) / 50

                # 2. 当前智能体到其他所有智能体的距离
                agent_to_other_dists = []
                for other_name in self.agent_names:
                    if other_name != name:
                        other_pos = agent_positions[other_name]
                        dist = np.linalg.norm(agent_pos - other_pos) / 50
                        agent_to_other_dists.append(dist)

                # 添加距离信息到观察中
                # 1. 当前智能体到evader的距离
                obs_now[name].append(np.array([agent_to_evader_dist]))

                # 2. 当前智能体到 原点的距离
                obs_now[name].append([np.linalg.norm(agent_positions[name])])

                # 2. 当前智能体到其他所有智能体的距离
                obs_now[name].append(np.array(agent_to_other_dists))

                # 3. evader到原点的距离
                obs_now[name].append(np.array([evader_dis]))

        # 更新历史观察队列
        [
            self.old_obs_queue[name].append(np.concatenate(obs_now[name]))
            for name in self.agent_names
        ]

        # 构建最终观察信息
        obs_ret = {}
        for name in self.agent_names:
            obs = []
            for elem in self.old_obs_queue[name]:
                obs = np.concatenate([obs, elem])
            for elem in self.old_action_queue[name]:
                obs = np.concatenate([obs, elem])
            obs_ret[name] = obs

        return obs_ret

    def _get_obs(self) -> dict[str, np.ndarray[np.float32, (None, 3)]]:

        if self.obs_compatible:
            return self._get_obs_compatible()

        agent_lvlh = {}
        evader_state = self.simulator.get_satellite_state(self.attacker_name)  # copy
        evader_state[:3] /= self.outside_threshold

        for name in self.agent_names:
            agent_state = self.simulator.get_satellite_state(name)  # copy
            agent_state[:3] /= self.outside_threshold
            agent_lvlh[name] = agent_state
            if self.use_diff_obs:
                agent_lvlh[name] = agent_lvlh[name] - evader_state

        obs_now = {name: [] for name in self.agent_names}

        evader_to_hvt_dis = (
            self.simulator.get_distance_to(self.attacker_name, self.hvt_name)
            / self.outside_threshold
        )

        for name in self.agent_names:
            # 记录基本观察信息
            obs_now[name].append(agent_lvlh[name])
            obs_now[name].append(evader_state)

            for other in self.agent_names:
                if other == name:
                    continue
                obs_now[name].append(agent_lvlh[other])

            # 只有在 obs_add_distance 为 True 时才添加距离信息
            if self.obs_add_distance:
                agent_to_evader_dist = (
                    self.simulator.get_distance_to(name, self.attacker_name)
                    / self.outside_threshold
                )

                # 2. 当前智能体到其他所有智能体的距离
                agent_to_other_dists = []
                for other_name in self.agent_names:
                    if other_name != name:
                        agent_to_other_dists.append(
                            self.simulator.get_distance_to(name, other_name)
                            / self.outside_threshold
                        )

                # 添加距离信息到观察中
                # 1. 当前智能体到evader的距离
                obs_now[name].append(np.array([agent_to_evader_dist]))

                # 2. 当前智能体到原点(hvt)的距离
                obs_now[name].append(
                    np.array(
                        [
                            self.simulator.get_distance_to(name, self.hvt_name)
                            / self.outside_threshold
                        ]
                    )
                )

                # 2. 当前智能体到其他所有智能体的距离
                obs_now[name].append(np.array(agent_to_other_dists))

                # 3. evader到原点的距离
                obs_now[name].append(np.array([evader_to_hvt_dis]))

        # 更新历史观察队列
        [
            self.old_obs_queue[name].append(np.concatenate(obs_now[name]))
            for name in self.agent_names
        ]

        # 构建最终观察信息
        obs_ret = {}
        for name in self.agent_names:
            obs = []
            for elem in self.old_obs_queue[name]:
                obs = np.concatenate([obs, elem])
            for elem in self.old_action_queue[name]:
                obs = np.concatenate([obs, elem])
            obs_ret[name] = obs

        return obs_ret

    def _setup_satellite(self) -> List[str]:
        # self.r_min = self.scenario_config["orbit_min"]
        # self.r_max = self.scenario_config["orbit_max"]
        # self.attacker_name = self.scenario_config["attacker_name"]

        if self.scenario_config.get("attacker_name"):
            self.attacker_name = self.scenario_config["attacker_name"]
        else:
            self.attacker_name = self.scenario_config["attacker_prefix"] + "_0"
        self.interceptor_prefix = self.scenario_config["interceptor_prefix"]
        self.attacker_prefix = self.scenario_config["attacker_prefix"]
        self.hvt_name = self.scenario_config["hvt_name"]

        # create_3v1_hvt_co_orbital_scenario(self.simulator, self.scenario_config)
        agent_count, attacker_count = create_scenario(
            self.simulator, self.scenario_config
        )
        agent_name = [
            self.interceptor_prefix + "_" + str(index) for index in range(agent_count)
        ]
        if self.on_plane:
            points = [
                self.simulator.get_satellite_position(name) for name in agent_name
            ]
            self.plane_normal = get_plane_normal(points[0], points[1], points[2])
            self.plane_point = points[0]
        return [self.attacker_name] + agent_name

    def _get_info(self) -> dict:
        return {
            "total_capture_sustain_counter": self.total_capture_sustain_counter,
            "capture_sustain_counter": self.capture_sustain_counter,
            "capture_reward_claimed": self.capture_reward_claimed,
            "collided": self.collided,
            "current_step": self.current_step,
            "out_satellite": self.out_satellite,
            "hvt_unsafe": self.hvt_unsafe,
        }

    def _after_reset(self):
        self.hvt_unsafe = False
        self.capture_sustain_counter = 0  # 重置奖励计算器
        self.total_capture_sustain_counter = 0
        self.capture_reward_claimed = False
        self.capture = False

        self._setup_obs_action_queue()

        if self.render_mode is not None:
            self.simulator.set_center_satellite(self.hvt_name)
