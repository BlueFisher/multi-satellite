from copy import deepcopy
from typing import Any, List, Dict

import numpy as np
from gymnasium import spaces
from scipy.optimize import linear_sum_assignment

from .SatelliteEnvBase import SatelliteEnvBase, validate_target_sector
from .dynamics.GeneratorFactory import GeneratorType
from .dynamics.OrbitGeneratorCW import CwSolution
from .dynamics.OrbitGeneratorCW import OrbitGeneratorCW
from .rewards.reward.RewardComputer2v2 import RewardComputer2v2
from .utils import (
    distance,
    get_plane_normal,
    create_scenario,
)
from .visualization.SimulatorFactory import SimulatorType


class SatelliteEnv2v2(SatelliteEnvBase):
    def __init__(
        self,
        # -----------环境基础设置------------
        render_mode=None,
        m_max_episode_steps=1000,
        update_interval: int = 10,  # 模拟一步的时间, 单位为秒，默认10s
        max_trail_length: int = 1000,  # 显示轨迹的最大步长，只有Local 才有效
        sampling_interval: int = 5,  # 显示轨迹采样的大小，只有Local 才有效
        action_scale: float = 0.00015,
        simulator: SimulatorType = SimulatorType.Local,
        start_on_init: bool = True,
        generator: GeneratorType = GeneratorType.CW,
        on_plane=True,
        obs_add_last_action=False,
        # -----------2v2任务特定参数------------
        config_name="SatelliteEnv2v2.yaml",
        outside_threshold=50.0,  # 出界阈值（km)
        collision_threshold=1.0,  # 碰撞安全距离（km）
        intercept_distance_threshold_km=10.0,  # 拦截条件: 距离阈值（km）
        hvt_safe_distance=5.0,  # 攻击者-Hvt安全阈值
        intercept_time=20,  # 拦截需要维持时间步
        win_max_angle=15.0,  # 拦截成功最大夹角
        interceptor_max_accel=1.0,  # （规则）拦截者最大机动（无用）
        attacker_max_accel=5.0 / 6.0,  # （规则）攻击者最大机动
        reward_computer_type: str = "RewardComputer2v2",
        # -----------环境特殊设置------------
        agent_heuristic=False,
        opponent_heuristic=False,
        limit_accel=False,
        target_sector_angle_deg=None,
        target_sector_radius_km=None,
    ):
        self._target_sector = validate_target_sector(
            target_sector_angle_deg, target_sector_radius_km
        )
        # 任务状态变量
        self.attacker_prefix: str = ""
        self.interceptor_prefix: str = ""
        self.hvt_name: str = ""
        self.attacker_names: List[str] = []
        self.interceptor_names: List[str] = []
        self.attackers_alive: Dict[str, str] = (
            {}
        )  # 攻击卫星的生存状态（"alive"/"dead"），围捕成功将该攻击卫星设置成dead
        self.assignments: Dict[str, str] = {}  # 当前回合的固定分配（拦截者—攻击者）
        self.interceptor_intercept_duration: Dict[str, int] = {}  # 拦截维持时间

        # 每步更新状态
        self.is_intercept: Dict[str, bool] = {}  # 本步是否拦截
        self.all_intercept_claimed: bool = False  # 是否全部拦截

        # 失败条件标志
        self.is_hvt_breach = (
            False  # self.collided or self.is_hvt_breach or len(self.out_satellite)
        )

        # 任务特定阈值
        self.intercept_distance_threshold_km = intercept_distance_threshold_km
        self.hvt_safe_distance = hvt_safe_distance
        self.intercept_time = intercept_time  # 拦截需维持时间步
        self.win_max_angle = win_max_angle  # 拦截成功最大夹角

        # 环境特殊设置
        self.on_plane = on_plane
        self.agent_heuristic = agent_heuristic
        self.opponent_heuristic = opponent_heuristic

        # 基于规则的攻击者参数
        self.attacker_loiter_dist = 10.0  # 规则攻击者悬停半径（距离HVT）
        self.interceptor_max_accel = interceptor_max_accel  # 默认 1
        self.attacker_max_accel = attacker_max_accel  # 默认 5/6
        self.update_interval = update_interval
        self.obs_add_last_action = obs_add_last_action
        self.last_action = self._reset_last_action()
        super().__init__(
            render_mode=render_mode,
            m_max_episode_steps=m_max_episode_steps,
            update_interval=update_interval,  # 模拟一步的时间, 单位为秒，默认10s
            max_trail_length=max_trail_length,  # 显示轨迹的最大步长，只有Local 才有效
            sampling_interval=sampling_interval,  # 显示轨迹采样的大小，只有Local 才有效
            action_scale=action_scale,
            simulator=simulator,
            start_on_init=start_on_init,
            generator=generator,
            config_name=config_name,
            collision_threshold=collision_threshold,
            outside_threshold=outside_threshold,
            on_plane=on_plane,
            reward_type=reward_computer_type,
            limit_accel=limit_accel,
        )
        self.interceptor_names = self.agent_names

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

    def _get_scenario_config(self):
        config = super()._get_scenario_config()
        if self._target_sector is None:
            return config
        return self._apply_target_sector(config, self._target_sector)

    def _reset_last_action(self):
        return {
            name: np.zeros(
                3,
            )
            for name in self.interceptor_names
        }

    def _setup_satellite(self) -> List[str]:

        self.attacker_prefix = self.scenario_config["attacker_prefix"]
        self.interceptor_prefix = self.scenario_config["interceptor_prefix"]
        self.hvt_name = self.scenario_config["hvt_name"]

        # create_2v2_scenario(self.simulator, self.scenario_config)
        interceptor_count, attacker_counter = create_scenario(
            self.simulator, self.scenario_config
        )

        self.attacker_names = [
            self.attacker_prefix + "_" + str(index) for index in range(attacker_counter)
        ]
        self.interceptor_names = [
            self.interceptor_prefix + "_" + str(index)
            for index in range(interceptor_count)
        ]

        points = [
            self.simulator.get_satellite_position(name)
            for name in self.interceptor_names + self.attacker_names
        ]
        self.plane_normal = get_plane_normal(points[0], points[1], points[2])
        self.plane_point = points[0]
        return self.attacker_names + self.interceptor_names + [self.hvt_name]

    def _setup_observation_space(self):
        self.interceptor_obs_dim = 6  # (x,y,z,vx,vy,vz)
        self.attacker_obs_dim = 7  # (x,y,z,vx,vy,vz,is_alive)
        self.interceptor_num = len(self.interceptor_names)
        self.attacker_num = len(self.attacker_names)

        obs_dim = (  # 当前观测，暂未考虑历史观测  26
            self.interceptor_obs_dim * self.interceptor_num
            + self.attacker_obs_dim * self.attacker_num
        )

        if self.obs_add_last_action:
            obs_dim += 3

        self.observation_space = spaces.Dict(
            {
                name: spaces.Box(
                    low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
                )
                for name in self.interceptor_names
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

    def _add_accel(self, name: str, accel: np.ndarray[(3,), float]):
        super()._add_accel(name, accel)
        self.last_action[name] = accel

    def _after_reset(self):
        # 1. 重置任务状态变量
        self.attackers_alive = {name: "alive" for name in self.attacker_names}
        self.is_intercept = {name: False for name in self.interceptor_names}
        self.all_intercept_claimed = False
        self.is_hvt_breach = False
        self.interceptor_intercept_duration = {             # 记录当前拦截步数
            name: 0 for name in self.interceptor_names
        }
        self.intercept_success_flags = {
            name: False for name in self.interceptor_names
        }
        self.max_interceptor_duration = {                   # 记录单局最高拦截步数
            name: 0 for name in self.interceptor_names
        }
        self.total_interceptor_duration = {                 # 记录单局累计拦截步数
            name: 0 for name in self.interceptor_names
        }
        # 2. 在回合开始时固定分配任务
        self._assign_pairs_at_start()
        # 3. 重置奖励计算机的状态
        self.reward_computer.reset()
        # 4. 设置视角的中心
        self.simulator.set_center_satellite(self.hvt_name)

        self.last_action = self._reset_last_action()
        for name in self.simulator.agent_names():
            self.simulator.set_satellite_enable(name, True)

    def _after_update(self):
        # 1.检查本步的拦截状态，并更新攻击者存活状态
        self._check_interception()
        # 2.动态重分配（拦截目标死亡后重新分配）
        alive_attackers = [a for a, s in self.attackers_alive.items() if s == "alive"]
        if len(alive_attackers) == 1:
            self.reassign_dead_attackers()
        # 3.检查是否所有目标都被拦截
        if all(status == "dead" for status in self.attackers_alive.values()):
            if not self.all_intercept_claimed:
                # print(f"--- Step {self.current_step}: 任务成功! 所有目标已被拦截! ---")
                self.all_intercept_claimed = True
        # 4.更新失败条件
        self.is_hvt_breach = (
            self._check_hvt_breach()
        )  #  _is_collided() 和 _is_outside() 在基类的 step 中已被调用

    def _before_update(self):
        for name in self.interceptor_names:
            self.intercept_success_flags[name] = False
        # ------------ 基于规则的 AI （2v2场景） ------------
        if self.opponent_heuristic:  # 攻击方 AI
            interceptor_states = [
                self.simulator.get_satellite_state(name)
                for name in self.interceptor_names
            ]
            hvt_state = self.simulator.get_satellite_state(self.hvt_name)
            # 先收集所有活着的攻击者状态
            attacker_states_dict = {
                name: self.simulator.get_satellite_state(name)
                for name in self.attacker_names
                if self.attackers_alive.get(name) == "alive"
            }
            for attacker_name in self.attacker_names:
                if self.attackers_alive[attacker_name] == "alive":
                    attacker_state = attacker_states_dict[attacker_name]
                    other_attacker_states = [
                        s for n, s in attacker_states_dict.items() if n != attacker_name
                    ]
                    accel = self._calculate_attacker_acceleration(
                        attacker_state,
                        hvt_state,
                        interceptor_states,
                        other_attacker_states,
                    )
                    self._add_accel(attacker_name, accel)

        if self.agent_heuristic:  # 拦截方 AI
            for interceptor_name in self.interceptor_names:
                interceptor_states = self.simulator.get_satellite_state(
                    interceptor_name
                )
                assigned_attacker_name = self.assignments.get(interceptor_name)
                if (
                    assigned_attacker_name
                    and self.attackers_alive[assigned_attacker_name] == "alive"
                ):
                    attacker_state = self.simulator.get_satellite_state(
                        assigned_attacker_name
                    )
                    accel = self._calculate_interceptor_acceleration(
                        interceptor_name, attacker_state
                    )
                    self._add_accel(interceptor_name, accel)

    def _calculate_interceptor_acceleration(
        self, interceptor_name: str, target_attacker_state: np.ndarray
    ) -> np.ndarray:
        """
        拦截者规则
        - 拦截分配的攻击者
        - 战术站位：落在 A->H 直线上、拦截圈ITP（理想战术点）
        - 环带：以 A为圆心，半径 d_des = intercept_distance_threshold_km，±10% 误差
          * 环带外：快速收敛
          * 环带内：以速度匹配 + 横向消抖为主
        - 软避撞：目标/队友，平方强度
        - 软边界：接近 50km 触发返回力
        """
        max_accel = self.interceptor_max_accel
        d_des = float(self.intercept_distance_threshold_km)  #  = 5.0 km 围捕阈值
        tol = 0.10 * d_des  # ±10%
        lower, upper = d_des - tol, d_des + tol

        boundary_dist = float(self.outside_threshold)  # 出界
        boundary_buffer = 20.0
        collision_avoid_dist = float(self.collision_threshold) + 1.0  # 碰撞避免距离

        # 增益
        kp_pos = 0.25  # 到 ITP 的位置P
        kd_vel = 0.60  # 对 ITP 的速度匹配D
        k_cross = 0.35  # 拉回 A–H 直线的横向位置误差
        k_latv = 0.45  # 消除横向相对速度

        alpha_v = 0.3  # 估算 ITP 速度时 A/H 混合比

        # 状态
        I = self.simulator.get_satellite_state(interceptor_name)
        H = self.simulator.get_satellite_state(self.hvt_name)
        A = target_attacker_state

        rI, vI = I[:3], I[3:]
        rH, vH = H[:3], H[3:]
        rA, vA = A[:3], A[3:]

        # -----高优先级：软边界 + 软避撞-----
        a_high = np.zeros(3)
        # 边界
        vecIH = rH - rI
        dIH = np.linalg.norm(vecIH)
        if dIH > boundary_dist - boundary_buffer:
            u_center = vecIH / (dIH + 1e-9)
            pen = (dIH - (boundary_dist - boundary_buffer)) / boundary_buffer
            a_high += (pen**2) * u_center * max_accel
        # 避开目标
        vecIT = rA - rI
        dIT = np.linalg.norm(vecIT)
        if dIT < collision_avoid_dist and dIT > 1e-6:
            a_high += (
                ((1.0 - dIT / collision_avoid_dist) ** 2)
                * (-(vecIT / (dIT + 1e-9)))
                * max_accel
            )
        # 避开队友
        teammate_name = [n for n in self.interceptor_names if n != interceptor_name][0]
        Tm = self.simulator.get_satellite_state(teammate_name)
        vecITm = Tm[:3] - rI
        dITm = np.linalg.norm(vecITm)
        if dITm < collision_avoid_dist and dITm > 1e-6:
            a_high += (
                ((1.0 - dITm / collision_avoid_dist) ** 2)
                * (-(vecITm / (dITm + 1e-9)))
                * max_accel
            )

        if np.linalg.norm(a_high) > 1e-6:
            nrm = np.linalg.norm(a_high)
            if nrm > max_accel:
                a_high = a_high / nrm * max_accel
            return a_high

        # ----- 核心：ITP + 环带一致控制 -----
        # ITP: A->H 方向上，距 A 为 d_des 的点（若 A–H 很短，则落在 A–H 段内）
        vecAH = rH - rA
        dAH = np.linalg.norm(vecAH)
        if dAH < 1e-6:
            # 退化：直接追 A 并速度匹配
            u_AI = rA - rI
            d_AI = np.linalg.norm(u_AI)
            u_AI = u_AI / (d_AI + 1e-9) if d_AI > 1e-6 else np.zeros(3)
            a_cmd = kp_pos * u_AI * min(d_AI / 10.0, 1.0) + kd_vel * (vA - vI)
            nrm = np.linalg.norm(a_cmd)
            if nrm > max_accel:
                a_cmd = a_cmd / nrm * max_accel
            return a_cmd

        u_AH = vecAH / dAH
        place = (
            d_des if dAH >= d_des else max(0.3 * dAH, 0.5 * dAH)
        )  # 保证 ITP 在 A/H 之间且不过冲
        r_ITP = rA + u_AH * place

        # ITP 速度近似（A/H 投影合成，偏向 A）
        vA_par = np.dot(vA, u_AH) * u_AH
        vH_par = np.dot(vH, u_AH) * u_AH
        v_ITP = vA_par + alpha_v * (vH_par - vA_par)

        # 到 ITP 的误差
        e_pos = r_ITP - rI
        d_pos = np.linalg.norm(e_pos)
        u_pos = e_pos / (d_pos + 1e-9)

        # 与攻击者的实际距离与相对量
        rAI = rI - rA
        v_rel = vI - vA
        rAI_par = np.dot(rAI, u_AH) * u_AH
        rAI_lat = rAI - rAI_par
        v_rel_par = np.dot(v_rel, u_AH) * u_AH
        v_rel_lat = v_rel - v_rel_par

        a_cmd = np.zeros(3)

        if dIT > upper:
            # 距离过大：拉近到 ITP + 匹配其速度 + 消横向相对速度 + 拉回直线
            a_cmd += kp_pos * u_pos * min(d_pos / 15.0, 1.0)
            a_cmd += kd_vel * (v_ITP - vI)
            a_cmd += -k_latv * v_rel_lat
            a_cmd += (
                -k_cross
                * rAI_lat
                / (np.linalg.norm(rAI_lat) + 1e-9)
                * min(np.linalg.norm(rAI_lat) / 10.0, 1.0)
            )
        elif dIT < lower:
            # 距离过小：沿远离 A 的方向打开距离 + 抑制横向
            u_away = -(rA - rI) / (dIT + 1e-9)
            a_cmd += kp_pos * u_away * min((lower - dIT) / 5.0, 1.0)
            a_cmd += -k_latv * v_rel_lat
            a_cmd += (
                -k_cross
                * rAI_lat
                / (np.linalg.norm(rAI_lat) + 1e-9)
                * min(np.linalg.norm(rAI_lat) / 10.0, 1.0)
            )
            a_cmd += 0.3 * kd_vel * (vA - vI)  # 轻匹配
        else:
            # 环带内：主要消相对速度 + 抑制横向 + 轻微位置保持
            a_cmd += kd_vel * (v_ITP - vI)
            a_cmd += -k_latv * v_rel_lat
            a_cmd += (
                -k_cross
                * rAI_lat
                / (np.linalg.norm(rAI_lat) + 1e-9)
                * min(np.linalg.norm(rAI_lat) / 10.0, 1.0)
            )
            a_cmd += 0.2 * kp_pos * u_pos * min(d_pos / 15.0, 1.0)
            # 确保 A–I–H 排列（点积>0），若反向给一点沿 u_AH 的偏置
            if np.dot(rI - rA, rH - rA) < 0:
                a_cmd += 0.2 * u_AH

        nrm = np.linalg.norm(a_cmd)
        if nrm > max_accel:
            a_cmd = a_cmd / nrm * max_accel
        return a_cmd

    def _calculate_attacker_acceleration(
        self,
        attacker_state: np.ndarray,
        hvt_state: np.ndarray,
        interceptor_states: List[np.ndarray],
        other_attacker_states: List[np.ndarray] = None,
    ) -> np.ndarray:
        """
        （）攻击者规则：
        - 远距离区 (> 30km): 迅速接近 HVT
        - 中距离区 (10km~30km): 维持在环带，并进行全套避碰   # 15km
        - 近距离区 (< 10km): 强排斥，避免靠近 HVT
        """
        # === 基础参数 ===
        R_close = 20.0  # 近距离排斥区
        R_far = 30.0  # 进入追击模式的边界
        eps = 1e-9
        attacker_max_accel = self.attacker_max_accel

        rA, vA = attacker_state[:3], attacker_state[3:]
        rH, vH = hvt_state[:3], hvt_state[3:]

        r_rel = rA - rH
        v_rel = vA - vH
        d = float(np.linalg.norm(r_rel))

        # ===基础向量计算===
        # 1. 径向 (Radial)
        u_radial = r_rel / (d + eps)

        # 2. 法向 (Normal) - 轨道角动量方向
        h_vec = np.cross(rH, vH)
        h_norm = np.linalg.norm(h_vec)
        u_normal = np.array([0.0, 0.0, 1.0]) if h_norm < eps else h_vec / h_norm

        # 3. 切向 (Tangent) - 轨道面内垂直于径向
        base_tangent = np.cross(u_normal, u_radial)

        # 辅助向量
        u_to_hvt = -u_radial  # 指向HVT
        u_away_hvt = u_radial  # 远离HVT

        a_cmd = np.zeros(3)

        # === 内部函数：常规避碰 ===
        def repulse(me_pos, other_pos, safe_dist, gain):
            vec = me_pos - other_pos
            dist = float(np.linalg.norm(vec))
            if dist < safe_dist:
                u_away = vec / (dist + eps)
                depth = (safe_dist - dist) / safe_dist
                return (gain * depth**2) * u_away * attacker_max_accel
            return np.zeros(3)

        # --- 模式 1: 远距离区 (> 30km) ---
        if d > R_far:
            kp_far, kd_far = 0.25, 0.6
            a_cmd = kp_far * u_to_hvt + kd_far * (-v_rel)  # P+D 直线追击

            # 远区避碰
            for s in interceptor_states or []:
                a_cmd += repulse(rA, s[:3], 15.0, 0.3)
            # 远区也要避队友，防止碰撞
            for s in other_attacker_states or []:
                if np.allclose(s[:3], rA):
                    continue
                a_cmd += repulse(rA, s[:3], 5.0, 0.8)

        # --- 模式 2: 中距离区 (10-30km) ---
        elif R_close <= d <= R_far:
            # 保持 20km 半径环绕
            target_radius = 20.0
            radial_err = target_radius - d
            kp_mid, kd_mid, kt_mid = 0.08, 0.40, 0.25

            # 分解速度
            v_radial = np.dot(v_rel, u_radial) * u_radial
            v_tan = v_rel - v_radial

            # 径向控制 + 切向阻尼
            a_cmd += kp_mid * radial_err * (-u_radial)
            a_cmd += -kd_mid * v_radial
            a_cmd += -kt_mid * v_tan

            # 全套避碰
            a_cmd += repulse(rA, rH, 15.0, 1.5)
            for s in interceptor_states or []:
                a_cmd += repulse(rA, s[:3], 10.0, 1.2)
            for s in other_attacker_states or []:
                if np.allclose(s[:3], rA):
                    continue
                a_cmd += repulse(rA, s[:3], 10.0, 0.8)

        # --- 模式 3: 近距离区 (< 10km) ---
        elif d < R_close:
            k_emergency = 2.0
            inward_speed = -np.dot(v_rel, u_away_hvt)

            # 强力排斥
            a_cmd = k_emergency * (R_close - d) * u_away_hvt
            if inward_speed > 0:
                a_cmd += 1.5 * inward_speed * u_away_hvt

            # 近距离避碰
            for s in interceptor_states or []:
                a_cmd += repulse(rA, s[:3], 5.0, 1.5)
            for s in other_attacker_states or []:
                if np.allclose(s[:3], rA):
                    continue
                a_cmd += repulse(rA, s[:3], 5.0, 1.0)

        # =========================================================
        #            [全域增强版] 强力正交扩散 (Orthogonal Expansion)
        # =========================================================
        # 核心逻辑：直接计算在“横截面”上的斥力。
        # 对象：包含【队友攻击者】和【敌方拦截者】，只要挡路统统推开。

        separation_threshold = 10.0  # 触发扩散的距离阈值
        separation_gain = 0.1  # 扩散力度系数

        a_separation = np.zeros(3)
        neighbor_count = 0

        # [修改点]：合并队友列表和敌人列表，统一进行扩散计算
        # 注意：这里 s 可能是攻击者状态，也可能是拦截者状态，但前3位都是位置
        all_neighbors = (other_attacker_states or []) + (interceptor_states or [])

        # 只有在中远距离才需要这种战术扩散，近距离优先保命/冲锋
        if d > R_close:
            for s in all_neighbors:
                other_pos = s[:3]

                # 排除自己 (防止自己对自己产生斥力)
                if np.allclose(other_pos, rA):
                    continue

                vec_to_me = rA - other_pos  # 从目标指向我

                # 分解向量到两个轴上 (横截面投影)
                # base_tangent 是之前算好的几何切向 (不含螺旋方向)
                dist_tan = np.dot(vec_to_me, base_tangent)  # 左右距离
                dist_nor = np.dot(vec_to_me, u_normal)  # 上下距离

                # 计算横截面上的 2D 距离 (投影距离)
                planar_dist = np.sqrt(dist_tan**2 + dist_nor**2)

                # 如果在横截面上靠得太近 (视线遮挡/弹道重合)
                if planar_dist < separation_threshold:

                    # 1. 计算推力方向 (仅在 2D 平面上)
                    if planar_dist < 0.1:
                        # 几乎重叠时随机选方向
                        u_push_2d = base_tangent if np.random.rand() > 0.5 else u_normal
                    else:
                        # 正常的推开方向
                        u_push_2d = (
                            dist_tan * base_tangent + dist_nor * u_normal
                        ) / planar_dist

                    # 2. 计算强度 (反比函数，越近推力越爆炸)
                    # 使用 (1/x) - (1/R) 的势场形式
                    force_mag = (1.0 / (planar_dist + 0.5)) - (
                        1.0 / separation_threshold
                    )

                    # [应用 separation_gain]
                    force_mag = (
                        max(0.0, force_mag) * separation_gain * attacker_max_accel
                    )

                    # 叠加力
                    a_separation += force_mag * u_push_2d
                    neighbor_count += 1

            # 应用分离力
            if neighbor_count > 0:
                # 限幅，允许占用最多 80% 的动力 (为了躲避敌人，权限给高一点)
                sep_norm = np.linalg.norm(a_separation)
                max_sep_accel = 0.2 * attacker_max_accel

                if sep_norm > max_sep_accel:
                    a_separation = a_separation / sep_norm * max_sep_accel

                a_cmd += a_separation

        # =========================================================
        #            最终输出
        # =========================================================
        nrm = np.linalg.norm(a_cmd)
        if nrm > attacker_max_accel:
            a_cmd = a_cmd / nrm * attacker_max_accel

        return a_cmd

    def _get_obs(self) -> dict[str, np.ndarray]:
        interceptor_state_lvlh = {}
        for interceptor_name in self.interceptor_names:
            interceptor_state = self.simulator.get_satellite_state(
                interceptor_name
            ).copy()
            interceptor_state[:3] = interceptor_state[:3] / self.outside_threshold
            interceptor_state_lvlh[interceptor_name] = interceptor_state

        attacker_state_lvlh = {}
        for attacker_name in self.attacker_names:
            attacker_state = self.simulator.get_satellite_state(attacker_name)
            attacker_state[:3] = attacker_state[:3] / self.outside_threshold
            attacker_state_lvlh[attacker_name] = attacker_state

        obs_dict = {}
        for interceptor_name in self.interceptor_names:
            teammate_name = [
                name for name in self.interceptor_names if name != interceptor_name
            ][0]
            assigned_attacker_name = self.assignments[interceptor_name]  # 拦截-攻击对
            other_attacker_name = [
                name for name in self.attacker_names if name != assigned_attacker_name
            ][0]

            self_state = interceptor_state_lvlh[interceptor_name]
            teammate_state = interceptor_state_lvlh[teammate_name]
            assigned_attacker_state = attacker_state_lvlh[assigned_attacker_name]
            assigned_attacker_alive = np.array(
                [
                    (
                        1.0
                        if self.attackers_alive[assigned_attacker_name] == "alive"
                        else 0.0
                    )
                ]
            )
            other_attacker_state = attacker_state_lvlh[other_attacker_name]
            other_attacker_alive = np.array(
                [1.0 if self.attackers_alive[other_attacker_name] == "alive" else 0.0]
            )

            obs_vector = np.concatenate(
                [  # 26
                    self_state,
                    teammate_state,
                    assigned_attacker_state,
                    assigned_attacker_alive,
                    other_attacker_state,
                    other_attacker_alive,
                ]
            ).astype(float)

            if self.obs_add_last_action:
                obs_vector = np.concatenate(
                    [obs_vector, self.last_action[interceptor_name]],
                )

            obs_dict[interceptor_name] = obs_vector
        return obs_dict

    def _is_done(self) -> bool:
        # 任务成功
        if self.all_intercept_claimed:
            # print(f"{self.current_step}步:任务成功")
            return True
        # 任务失败
        if self.collided or self.is_hvt_breach or len(self.out_satellite) > 0:
            # print(f"{self.current_step}步：任务失败：is_collision: {self.collided}, is_hvt_breach: {self.is_hvt_breach}, is_out: {self.out_satellite}")
            return True
        return False

    def _is_outside(self) -> List[str]:
        """检查是否有拦截者飞出作战区域（与HVT的距离）"""
        outside_list = []
        hvt_pos = self.simulator.get_satellite_position(self.hvt_name)
        for name in self.interceptor_names:
            inter_pos = self.simulator.get_satellite_position(name)
            if distance(inter_pos, hvt_pos) > self.outside_threshold:
                outside_list.append(name)
        # for name in self.attacker_names:
        #     att_pos = self.simulator.get_satellite_position(name)
        #     if distance(att_pos, hvt_pos) > self.outside_threshold:
        #         print(f"{self.current_step}:攻击者距离hvt:{name}, att: {distance(att_pos, hvt_pos)}")
        #         outside_list.append(name)
        return outside_list

    def _get_info(self) -> dict[str, Any]:

        info = {
            "all_intercept_claimed": self.all_intercept_claimed,
            "is_hvt_breach": self.is_hvt_breach,
            "collided": self.collided,
            "out_of_boundary": self.out_satellite,
            "current_step": self.current_step,
        }

        # for key, value in self.interceptor_intercept_duration.items():
        #     info[str(key) + "interceptor_duration"] = value
        for key, value in getattr(self, 'max_interceptor_duration', {}).items():
            info[f"max_interceptor_duration_{key}"] = value
        for key, value in getattr(self, 'total_interceptor_duration', {}).items():
            info[f"total_interceptor_duration_{key}"] = value
        return info

    # ------------2v2任务特定辅助方法--------------
    def _solve_bounded_constant_accel(
        self,
        Psi_r: np.ndarray,
        r_free: np.ndarray,
        a_max: float,
        rtol: float = 1e-9,
        max_iter: int = 60,
    ):
        """
        约束最小二乘问题求解器：找到一个加速度向量，使得轨迹误差（残差）最小，同时加速度不能超过最大允许值
        ||Psi_r a + r_free||， s.t. ||a|| <= a_max
        返回 (a_opt, residual_norm)。
        """
        # 无约束最小范数解
        a_uncon = -np.linalg.pinv(Psi_r) @ r_free
        norm_uncon = np.linalg.norm(a_uncon)
        if norm_uncon <= a_max + 1e-12:
            res = Psi_r @ a_uncon + r_free
            return a_uncon, np.linalg.norm(res)
        # 需要约束：二分 lambda 解 ||a(lambda)|| = a_max
        AtA = Psi_r.T @ Psi_r
        Atb = Psi_r.T @ r_free
        # 初始上下界：lambda_low=0，lambda_high 逐步扩大直到 ||a(lambda_high)|| <= a_max
        lam_low = 0.0
        lam_high = 1.0

        def a_of_lam(lam):
            return -np.linalg.solve(AtA + lam * np.eye(3), Atb)

        while True:
            a_try = a_of_lam(lam_high)
            if np.linalg.norm(a_try) <= a_max or lam_high > 1e20:
                break
            lam_high *= 10.0
        # 二分
        for _ in range(max_iter):
            lam_mid = 0.5 * (lam_low + lam_high)
            a_mid = a_of_lam(lam_mid)
            n_mid = np.linalg.norm(a_mid)
            if abs(n_mid - a_max) <= rtol * max(1.0, a_max):
                a_opt = a_mid
                break
            if n_mid > a_max:
                lam_low = lam_mid
            else:
                lam_high = lam_mid
        else:
            a_opt = a_mid
        res = Psi_r @ a_opt + r_free
        return a_opt, np.linalg.norm(res)

    def _assign_pairs_at_start(self):
        """
        在回合开始时，根据每个拦截者与每个攻击者的成本计算，使用匈牙利算法进行任务分配。
        成本是基于Clohessy-Wiltshire方程计算的，最小化拦截者的Δv0和可能的时间惩罚。
        """
        # 获取所有卫星的当前状态
        interceptor_states = {
            name: self.simulator.get_satellite_state(name)
            for name in self.interceptor_names
        }
        attacker_states = {
            name: self.simulator.get_satellite_state(name)
            for name in self.attacker_names
        }
        hvt_eci = self.simulator.get_origin_satellite_state()

        # 计算参考轨道的平均角速度 n
        cw = OrbitGeneratorCW()
        cw.set_origin(hvt_eci)
        n = cw.n

        # 设定时间步长网络
        T_grid = [120.0, 240.0, 360.0, 480.0, 600.0]
        T_max = float(max(T_grid))
        w_a, w_t, w_res = 1.0, 0.05, 10.0  # Δv0权重 + 时间权重 + 残差权重
        # 加速度上限
        a_max_interceptor = self.interceptor_max_accel  # 默认
        a_max_attacker = self.attacker_max_accel  # 默认

        # 初始化成本矩阵（行 = 拦截者，列 = 攻击者）
        cost_matrix = np.zeros((len(self.interceptor_names), len(self.attacker_names)))

        # 计算每对拦截者-攻击者的最小成本
        for i, interceptor_name in enumerate(self.interceptor_names):
            interceptor_state = interceptor_states[interceptor_name]
            for j, attacker_name in enumerate(self.attacker_names):
                attacker_state = attacker_states[attacker_name]
                # 获取拦截者和攻击者的状态（相对于HVT）
                r_I, v_I = interceptor_state[:3], interceptor_state[3:]
                r_A, v_A = attacker_state[:3], attacker_state[3:]
                # 计算拦截者和攻击者之间的相对位置和相对速度
                r0 = r_I - r_A
                v0 = v_I - v_A

                best_cost = np.inf

                for T in T_grid:
                    Phi, Psi = CwSolution(n, T).Phi, CwSolution(n, T).Psi
                    Phi_rr, Phi_rv, Psi_r = Phi[0:3, 0:3], Phi[0:3, 3:6], Psi[0:3, :]
                    # 自由飞（零加速度）到 T 的相对位置
                    r_free_T = Phi_rr @ r0 + Phi_rv @ v0
                    # 攻击者：沿 HVT 方向、最大幅值
                    norm_rA = np.linalg.norm(r_A)
                    u_A_to_H = (-r_A / norm_rA) if norm_rA > 1e-12 else np.zeros(3)
                    aA = a_max_attacker * u_A_to_H
                    # 扰动后的“自由漂移”位置（加上攻击者机动的影响）
                    r_free_T_disturbed = r_free_T - Psi_r @ aA
                    # 仍然只对拦截者求解，但面对“更糟”的 r_free
                    a_opt, res_norm = self._solve_bounded_constant_accel(
                        Psi_r, r_free_T_disturbed, a_max_interceptor
                    )
                    # 成本：加速度幅值 + 时间惩罚 + 位置残差（越小越好）
                    J = (
                        w_a * np.linalg.norm(a_opt)
                        + w_t * (T / T_max)
                        + w_res * res_norm
                    )
                    if J < best_cost:
                        best_cost = J

                cost_matrix[i, j] = best_cost
        # 使用匈牙利算法找到最优分配
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        # 更新拦截者与攻击者的分配
        self.assignments = {
            self.interceptor_names[r]: self.attacker_names[c]
            for r, c in zip(row_ind, col_ind)
        }
        # print(f"成本：{cost_matrix}")
        # print(f"任务分配已完成: {self.assignments}")

    def _check_interception(self):
        """检查每个拦截-攻击对是否满足拦截条件，并更新状态。"""
        hvt_pos = self.simulator.get_satellite_position(self.hvt_name)
        max_angle_rad = np.deg2rad(self.win_max_angle)
        cos_min_angle = np.cos(max_angle_rad)  # cos(15°)

        leak_penalty = 5

        for inter_name, att_name in self.assignments.items():
            # 只有当攻击者还活着时才检查拦截
            if self.attackers_alive[att_name] == "alive":
                inter_pos = self.simulator.get_satellite_position(inter_name)
                att_pos = self.simulator.get_satellite_position(att_name)
                # 角度与排列
                vec_ai = inter_pos - att_pos  # A->I 向量
                vec_ah = hvt_pos - att_pos  # A->H 向量
                d_ai = np.linalg.norm(vec_ai)
                d_ah = np.linalg.norm(vec_ah)

                # 距离条件
                between_ok = d_ai < d_ah
                # 夹角条件
                dot = np.dot(vec_ai, vec_ah)
                cos_theta = dot / ((d_ai * d_ah) + 1e-9)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                angle_ok = cos_theta >= cos_min_angle
                # 距离条件
                in_ring = (
                    0.9 * self.intercept_distance_threshold_km
                    <= d_ai
                    <= 1.1 * self.intercept_distance_threshold_km
                )
                # 拦截距离圈 & 夹角<15 & 排列 A-I-HVT
                ok = in_ring and angle_ok and between_ok

                # # 检查是否拦截
                # if ok:
                #     if not self.is_intercept[
                #         inter_name
                #     ]:  # 只有在拦截状态下才增加持续时间
                #         self.interceptor_intercept_duration[inter_name] = (
                #             1  # 开始维持拦截
                #         )
                #         # print(
                #         #     f"第{self.current_step}步开始拦截：{inter_name}拦截{att_name},维持时间：{self.interceptor_intercept_duration[inter_name]}"
                #         # )
                #     else:
                #         self.interceptor_intercept_duration[
                #             inter_name
                #         ] += 1  # 增加维持时间
                #     # print(
                #     #     f"第{self.current_step}步：{inter_name}拦截{att_name},维持时间：{self.interceptor_intercept_duration[inter_name]}"
                #     # )
                #     self.is_intercept[inter_name] = True

                #     if self.interceptor_intercept_duration[inter_name] > self.max_interceptor_duration[inter_name]:
                #         self.max_interceptor_duration[inter_name] = self.interceptor_intercept_duration[inter_name]
                #     self.total_interceptor_duration[inter_name] += 1
                # else:
                #     self.is_intercept[inter_name] = False
                #     self.interceptor_intercept_duration[inter_name] = 0
                if ok:
                    # 无论之前是否在拦截状态，只要这步满足，就在现有基础上 +1
                    self.interceptor_intercept_duration[inter_name] += 1
                    self.is_intercept[inter_name] = True

                    # 记录最大值和总值
                    if self.interceptor_intercept_duration[inter_name] > self.max_interceptor_duration[inter_name]:
                        self.max_interceptor_duration[inter_name] = self.interceptor_intercept_duration[inter_name]
                    self.total_interceptor_duration[inter_name] += 1
                else:
                    self.is_intercept[inter_name] = False
                    # 不直接归零，而是减去 leak_penalty，最低保底为 0
                    self.interceptor_intercept_duration[inter_name] = max(
                        0, self.interceptor_intercept_duration[inter_name] - leak_penalty
                    )

                # 成功拦截的处理：更新攻击者存活状态
                if (
                    self.interceptor_intercept_duration[inter_name]
                    >= self.intercept_time
                ):  # 大于所需拦截维持时间，修改攻击者状态
                    # print(
                    #     f"--- Step {self.current_step}: {inter_name} 成功拦截 {att_name}! ---"
                    # )
                    self.intercept_success_flags[inter_name] = True
                    self.attackers_alive[att_name] = "dead"
                    self._on_attacker_dead(
                        att_name
                    )  # （重分配）清除所有盯着该攻击者的拦截者的状态/计时
            else:
                # 如果目标已经死亡，则该拦截者的拦截状态为False
                self.is_intercept[inter_name] = False

    def _check_hvt_breach(self) -> bool:
        """检查是否有存活的攻击者突破了HVT的安全防线。"""
        hvt_pos = self.simulator.get_satellite_position(self.hvt_name)
        for att_name in self.attacker_names:
            if self.attackers_alive[att_name] == "alive":
                att_pos = self.simulator.get_satellite_position(att_name)
                if distance(att_pos, hvt_pos) < self.hvt_safe_distance:
                    # print(f"!!! 失败: HVT防线被 {att_name} 突破 !!!")
                    return True
        return False

    # -------------------------任务重分配（拦截-攻击对）——暂时不考虑了-----------------------
    def reassign_dead_attackers(self):
        """
        当某个攻击卫星死亡时，重新分配该拦截卫星给一个活攻击卫星。
        """
        alive_attackers = [a for a, s in self.attackers_alive.items() if s == "alive"]
        if len(alive_attackers) != 1:
            # print(f"存活卫星数量：{len(alive_attackers)}, 不为1！")
            return

        only_alive = alive_attackers[0]
        for inter_name, old_att in list(self.assignments.items()):
            if self.attackers_alive.get(old_att) == "dead":
                # self._on_attacker_dead(old_att)     # 攻击者死亡，对配对拦截者清零计时和拦截状态

                if old_att != only_alive:  # 若本就指向唯一存活者则不必重复切换（防抖）
                    self.assignments[inter_name] = only_alive
                    self._on_target_changed(inter_name, old_att, only_alive)

    def _on_target_changed(
        self, inter_name: str, old_att: str | None, new_att: str | None
    ):
        """拦截者 inter_name 的目标从 old_att 切换为 new_att 时，统一做状态重置。"""
        self.interceptor_intercept_duration[inter_name] = 0
        self.is_intercept[inter_name] = False
        # 清掉该拦截者在奖励器里的“上一距离”，避免因目标切换导致的奖励跳变
        if isinstance(self.reward_computer, RewardComputer2v2):
            self.reward_computer.previous_distance.pop(inter_name, None)
        # print(
        #     f"[reassign] {inter_name}: {old_att} -> {new_att}, reset intercept duration/state."
        # )

    def _on_attacker_dead(self, att_name: str):
        """攻击者 att_name 死亡后，所有当前盯它的拦截者都要清零计时与状态。"""
        for inter_name, cur_att in self.assignments.items():
            if cur_att == att_name:
                self.interceptor_intercept_duration[inter_name] = 0
                self.is_intercept[inter_name] = False
                if isinstance(self.reward_computer, RewardComputer2v2):
                    self.reward_computer.previous_distance.pop(inter_name, None)

        self.simulator.set_satellite_enable(att_name, False)
