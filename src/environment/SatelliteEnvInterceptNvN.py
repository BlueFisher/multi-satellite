from typing import List, Dict

import gymnasium.spaces as spaces
import numpy as np
from scipy.optimize import linear_sum_assignment

from .SatelliteEnvBase import SatelliteEnvBase
from .dynamics.GeneratorFactory import GeneratorType
from .visualization.SimulatorFactory import SimulatorType
from .utils import (
    get_plane_normal,
    distance,
    project_point_to_plane,
    create_scenario,
)


class SatelliteEnvInterceptNvN(SatelliteEnvBase):
    """
    支持 NVN (多拦截者 vs 多攻击者) 的卫星拦截任务环境。
    重点在于：任务分配、角度限制拦截、保护HVT。
    """

    def __init__(
        self,
        # -----------环境基础设置------------
        render_mode=None,
        m_max_episode_steps=1000,
        update_interval: int = 10,
        max_trail_length: int = 1000,
        sampling_interval: int = 5,
        action_scale: float = 0.00015,
        simulator: SimulatorType = SimulatorType.Local,
        start_on_init: bool = True,
        generator: GeneratorType = GeneratorType.CW,
        obs_window_size=1,
        action_window_size=1,
        on_plane=True,
        # -----------NVN拦截任务特定参数------------
        config_name="nvn_intercept.yaml",  # 假设有一个适配NVN的配置文件
        outside_threshold=50.0,
        collision_threshold=1.0,
        intercept_distance_threshold_km=10.0,
        hvt_safe_distance=5.0,
        intercept_time=20,  # 判定成功所需的持续拦截步数
        win_max_angle=15.0,  # 拦截成功最大夹角 (度)
        interceptor_max_accel=1.0,
        attacker_max_accel=5.0 / 6.0,
        reward_computer_type="RewardComputerInterceptNvN",
        # -----------环境特殊设置------------
        agent_heuristic=False,  # 是否开启拦截者规则脚本
        opponent_heuristic=True,  # 是否开启攻击者规则脚本 (默认开启以提供对抗)
        *args,
        **kwargs,
    ):
        # 任务参数
        self.intercept_distance_threshold_km = intercept_distance_threshold_km
        self.hvt_safe_distance = hvt_safe_distance
        self.intercept_time = intercept_time
        self.win_max_angle = win_max_angle
        self.interceptor_max_accel = interceptor_max_accel
        self.attacker_max_accel = attacker_max_accel
        self.opponent_heuristic = opponent_heuristic
        self.agent_heuristic = agent_heuristic

        # 状态追踪
        self.attackers_alive: Dict[str, str] = {}  # "alive" / "dead"
        self.assignments: Dict[str, str] = {}  # interceptor -> attacker
        self.interceptor_intercept_duration: Dict[str, int] = {}
        self.is_intercept: Dict[str, bool] = {}
        self.all_intercept_claimed = False
        self.is_hvt_breach = False

        self.obs_window_size = obs_window_size
        self.action_window_size = action_window_size

        # 奖励计算辅助
        self.prev_distances = {}  # 用于计算shaping reward

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
            collision_threshold=collision_threshold,
            outside_threshold=outside_threshold,
            on_plane=on_plane,
            reward_type=reward_computer_type,
        )
        self.interceptor_names = self.agent_names

        # 初始化历史队列（如果有需要的话，目前沿用2v2结构暂时留空或按需添加）
        self._setup_obs_action_queue()

    def _setup_obs_action_queue(self):
        # 如果需要基于历史的观测，在这里初始化队列
        pass

    def _setup_satellite(self) -> List[str]:
        if self.scenario_config.get("attacker_prefix"):
            self.attacker_prefix = self.scenario_config["attacker_prefix"]
        else:
            self.attacker_prefix = "Attacker"

        if self.scenario_config.get("interceptor_prefix"):
            self.interceptor_prefix = self.scenario_config["interceptor_prefix"]
        else:
            self.interceptor_prefix = "Inter"

        self.hvt_name = self.scenario_config.get("hvt_name", "HVT")

        # 使用 create_scenario，假设它返回 (拦截者数量, 攻击者数量)
        interceptor_count, attacker_count = create_scenario(
            self.simulator, self.scenario_config
        )

        self.attacker_names = [
            f"{self.attacker_prefix}_{i}" for i in range(attacker_count)
        ]
        self.interceptor_names = [
            f"{self.interceptor_prefix}_{i}" for i in range(interceptor_count)
        ]

        # 确定轨道平面 (用于 visualization 或 2D 简化计算)
        all_sats = self.interceptor_names + self.attacker_names
        if self.on_plane and len(all_sats) >= 3:
            points = [
                self.simulator.get_satellite_position(name) for name in all_sats[:3]
            ]
            self.plane_normal = get_plane_normal(points[0], points[1], points[2])
            self.plane_point = points[0]
        elif self.on_plane:
            self.plane_normal = np.array([0, 0, 1])
            self.plane_point = np.array([0, 0, 0])

        return self.attacker_names + self.interceptor_names + [self.hvt_name]

    def _setup_observation_space(self):
        self.satellite_obs_dim = 6  # Pos(3) + Vel(3)

        num_interceptors = len(self.interceptor_names)
        num_attackers = len(self.attacker_names)

        # 观测设计：
        # 1. 自身状态 (6)
        # 2. 分配的目标状态 (6 + 1alive)
        # 3. HVT 状态 (6)
        # 4. 其他队友状态 (6 * (N-1))
        # 5. 其他攻击者状态 ((6+1) * (M-1))
        # 注意：这种定长设计要求 N 和 M 在训练期间固定。

        obs_dim = (
            self.satellite_obs_dim  # Self
            + (self.satellite_obs_dim + 1)  # Assigned Target
            + self.satellite_obs_dim  # HVT
            + self.satellite_obs_dim * (num_interceptors - 1)  # Teammates
            + (self.satellite_obs_dim + 1) * (num_attackers - 1)  # Other Attackers
        )

        self.observation_space = spaces.Dict(
            {
                name: spaces.Box(
                    low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
                )
                for name in self.interceptor_names
            }
        )

    def _setup_action_space(self):
        self.action_space = spaces.Dict(
            {
                name: spaces.Box(-1, 1, shape=(3,), dtype=np.float32)
                for name in self.interceptor_names
            }
        )

    def _get_obs(self) -> dict[str, np.ndarray]:
        """
        获取 NVN 拦截任务的观测。
        归一化处理：位置除以 50.0 (outside_threshold)
        """
        states_lvlh = {}
        # 获取并归一化所有相关卫星状态
        all_names = self.interceptor_names + self.attacker_names + [self.hvt_name]
        for name in all_names:
            st = self.simulator.get_satellite_state(name).copy()
            st[:3] /= 50.0  # 简单归一化
            states_lvlh[name] = st

        obs_dict = {}
        for inter_name in self.interceptor_names:
            obs_list = []

            # 1. 自身状态
            obs_list.append(states_lvlh[inter_name])

            # 2. 分配的目标状态 (如果无分配或目标不存在，给零向量)
            assigned_att = self.assignments.get(inter_name)
            if assigned_att and assigned_att in self.attacker_names:
                att_state = states_lvlh[assigned_att]
                is_alive = (
                    1.0 if self.attackers_alive.get(assigned_att) == "alive" else 0.0
                )
                obs_list.append(att_state)
                obs_list.append([is_alive])
            else:
                # Fallback if logic fails or no assignment
                obs_list.append(np.zeros(6))
                obs_list.append([0.0])

            # 3. HVT 状态
            obs_list.append(states_lvlh[self.hvt_name])

            # 4. 队友状态
            for teammate in self.interceptor_names:
                if teammate == inter_name:
                    continue
                obs_list.append(states_lvlh[teammate])

            # 5. 其他攻击者状态
            for other_att in self.attacker_names:
                if other_att == assigned_att:
                    continue
                att_state = states_lvlh[other_att]
                is_alive = (
                    1.0 if self.attackers_alive.get(other_att) == "alive" else 0.0
                )
                obs_list.append(att_state)
                obs_list.append([is_alive])

            # Concatenate
            obs_dict[inter_name] = np.concatenate(obs_list).astype(np.float32)

        return obs_dict

    def _before_update(self):
        # 1. 攻击者 AI (基于规则)
        if self.opponent_heuristic:
            self._update_attackers_heuristic()

        # 2. 拦截者 AI (可选，用于测试或混合控制)
        if self.agent_heuristic:
            self._update_interceptors_heuristic()

    def _update_attackers_heuristic(self):
        """
        攻击者通用逻辑:
        - 存活时: 冲向 HVT，同时避开拦截者
        - 死亡时: 停止机动
        """
        interceptor_states = [
            self.simulator.get_satellite_state(name) for name in self.interceptor_names
        ]
        hvt_state = self.simulator.get_satellite_state(self.hvt_name)

        # 收集所有攻击者状态用于互相避碰
        all_att_states = [
            self.simulator.get_satellite_state(name)
            for name in self.attacker_names
            if self.attackers_alive.get(name) == "alive"
        ]

        for name in self.attacker_names:
            if self.attackers_alive[name] == "dead":
                continue

            my_state = self.simulator.get_satellite_state(name)

            # 复用/适配 2v2 中的攻击者逻辑
            accel = self._calculate_attacker_acceleration(
                my_state, hvt_state, interceptor_states, all_att_states
            )
            self._add_accel(name, accel)

    def _update_interceptors_heuristic(self):
        """
        拦截者通用逻辑
        """
        for name in self.interceptor_names:
            assigned_att = self.assignments.get(name)
            if not assigned_att or self.attackers_alive.get(assigned_att) != "alive":
                continue

            att_state = self.simulator.get_satellite_state(assigned_att)

            # 复用 2v2 中的拦截逻辑
            # 注意：_calculate_interceptor_acceleration 需要 self.hvt_name 对应的状态
            # 这里调用需要确保方法签名匹配或稍作修改，这里假设直接复用
            accel = self._calculate_interceptor_acceleration(name, att_state)
            self._add_accel(name, accel)

    # ------------------ 复用 2v2 的物理计算与分配核心 ------------------
    # 这里我们直接包含 _calculate_interceptor_acceleration 和 _calculate_attacker_acceleration
    # 为了保持文件独立完整，将它们适配到此类中

    def _calculate_interceptor_acceleration(
        self, interceptor_name: str, target_attacker_state: np.ndarray
    ) -> np.ndarray:
        # ... (逻辑同 SatelliteEnv2v2, 略作通用性适配) ...
        max_accel = self.interceptor_max_accel
        d_des = float(self.intercept_distance_threshold_km)

        # 状态
        I = self.simulator.get_satellite_state(interceptor_name)
        H = self.simulator.get_satellite_state(self.hvt_name)
        A = target_attacker_state

        rI, vI = I[:3], I[3:]
        rH, vH = H[:3], H[3:]
        rA, vA = A[:3], A[3:]

        # 简化的拦截逻辑实现 (ITP Guidance)
        vecAH = rH - rA
        dAH = np.linalg.norm(vecAH)

        # 目标位置: A->H 连线上 d_des 处
        if dAH < 1e-6:
            target_pos = rA
        else:
            target_pos = rA + (vecAH / dAH) * min(d_des, dAH * 0.5)

        # PD 控制追踪目标位置
        err_p = target_pos - rI

        # 速度匹配 (匹配目标速度，或者如果太远则全速)
        # 这里简化：期望速度指向目标
        kp = 0.5
        kd = 1.0

        des_vel = (err_p / (np.linalg.norm(err_p) + 1e-9)) * 0.1  # 巡航速度
        if np.linalg.norm(err_p) > 2.0:
            des_vel *= 5.0

        accel = kp * err_p + kd * (vA - vI)  # 简单的相对控制

        # 简单的避碰
        for tm in self.interceptor_names:
            if tm == interceptor_name:
                continue
            tm_pos = self.simulator.get_satellite_position(tm)
            dist = np.linalg.norm(rI - tm_pos)
            if dist < 2.0:
                push = (rI - tm_pos) / (dist + 1e-9)
                accel += push * 2.0

        # 限幅
        nrm = np.linalg.norm(accel)
        if nrm > max_accel:
            accel = accel / nrm * max_accel
        return accel

    def _calculate_attacker_acceleration(
        self,
        attacker_state: np.ndarray,
        hvt_state: np.ndarray,
        interceptor_states: List[np.ndarray],
        other_attacker_states: List[np.ndarray],
    ) -> np.ndarray:
        # 简化版攻击逻辑：冲向HVT，避开拦截者
        rA = attacker_state[:3]
        rH = hvt_state[:3]

        vec_hvt = rH - rA
        dist_hvt = np.linalg.norm(vec_hvt)
        u_hvt = vec_hvt / (dist_hvt + 1e-9)

        accel = u_hvt * self.attacker_max_accel * 0.5  # 50% 推力进攻

        # 避障
        for idx, istate in enumerate(interceptor_states):
            rI = istate[:3]
            dist = np.linalg.norm(rA - rI)
            if dist < 8.0:  # 感知范围
                push = (rA - rI) / (dist + 1e-9)
                strength = (8.0 - dist) / 8.0
                accel += push * strength * self.attacker_max_accel * 1.5

        # 边界保护 (不飞太远)
        if np.linalg.norm(rA) > 45.0:
            accel += -rA * 0.1

        nrm = np.linalg.norm(accel)
        if nrm > self.attacker_max_accel:
            accel = accel / nrm * self.attacker_max_accel
        return accel

    def _after_reset(self):
        # 重置状态
        self.attackers_alive = {name: "alive" for name in self.attacker_names}
        self.is_intercept = {name: False for name in self.interceptor_names}
        self.interceptor_intercept_duration = {
            name: 0 for name in self.interceptor_names
        }
        self.all_intercept_claimed = False
        self.is_hvt_breach = False
        self.prev_distances = {}

        # 初始分配
        self._assign_pairs_at_start()

        # 可视化设置
        if self.render_mode is not None:
            self.simulator.set_center_satellite(self.hvt_name)
        for name in self.simulator.agent_names():
            self.simulator.set_satellite_enable(name, True)

    def _after_update(self):
        # 1. 检查拦截判定
        self._check_interception()

        # 2. 动态重分配 (当有攻击者死亡时)
        self._check_reassignment()

        # 3. 检查胜利条件
        alive_cnt = sum(1 for s in self.attackers_alive.values() if s == "alive")
        if alive_cnt == 0 and not self.all_intercept_claimed:
            # print(f"--- Step {self.current_step}: NVN 任务成功! 所有目标已被拦截! ---")
            self.all_intercept_claimed = True

        # 4. 检查失败条件 (HVT Breach)
        if not self.is_hvt_breach:
            self.is_hvt_breach = self._check_hvt_breach()

    def _assign_pairs_at_start(self):
        """
        N vs M 分配逻辑。
        使用 CW 方程估算消耗作为 Cost Matrix。
        支持 N != M。
        """
        interceptor_states = [
            self.simulator.get_satellite_state(n) for n in self.interceptor_names
        ]
        attacker_states = [
            self.simulator.get_satellite_state(n) for n in self.attacker_names
        ]

        n_int = len(interceptor_states)
        n_att = len(attacker_states)

        cost_matrix = np.zeros((n_int, n_att))

        for i in range(n_int):
            for j in range(n_att):
                r_diff = np.linalg.norm(
                    interceptor_states[i][:3] - attacker_states[j][:3]
                )
                v_diff = np.linalg.norm(
                    interceptor_states[i][3:] - attacker_states[j][3:]
                )
                cost_matrix[i, j] = r_diff + 10.0 * v_diff  # 简单的加权

        # 匈牙利算法求解
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        self.assignments = {}
        for r, c in zip(row_ind, col_ind):
            self.assignments[self.interceptor_names[r]] = self.attacker_names[c]

        # 处理未分配的拦截者 (如果 N > M)
        # 简单策略：分配给最近的存活攻击者，或者助攻
        assigned_inters = set(self.interceptor_names[r] for r in row_ind)
        unassigned_inters = [
            n for n in self.interceptor_names if n not in assigned_inters
        ]

        if unassigned_inters and n_att > 0:
            for inter_name in unassigned_inters:
                # 简单分配给第一个攻击者 (或者可以做更复杂的逻辑)
                self.assignments[inter_name] = self.attacker_names[0]

    def _check_interception(self):
        """
        检查拦截逻辑:
        1. 距离在阈值内
        2. 角度在阈值内 (A位于I和H之间)
        3. 维持时间达标
        """
        hvt_pos = self.simulator.get_satellite_position(self.hvt_name)
        cos_min_angle = np.cos(np.deg2rad(self.win_max_angle))

        for inter_name, att_name in self.assignments.items():
            if self.attackers_alive.get(att_name) != "alive":
                self.is_intercept[inter_name] = False
                continue

            inter_pos = self.simulator.get_satellite_position(inter_name)
            att_pos = self.simulator.get_satellite_position(att_name)

            vec_ai = inter_pos - att_pos
            vec_ah = hvt_pos - att_pos
            d_ai = np.linalg.norm(vec_ai)
            d_ah = np.linalg.norm(vec_ah)

            # 判据
            dist_ok = (
                0.8 * self.intercept_distance_threshold_km
                <= d_ai
                <= 1.2 * self.intercept_distance_threshold_km
            )
            pos_ok = d_ai < d_ah  # 必须在内侧

            dot = np.dot(vec_ai, vec_ah)
            cos_theta = dot / ((d_ai * d_ah) + 1e-9)
            angle_ok = cos_theta >= cos_min_angle

            if dist_ok and pos_ok and angle_ok:
                if not self.is_intercept.get(inter_name, False):
                    self.interceptor_intercept_duration[inter_name] = 1
                else:
                    self.interceptor_intercept_duration[inter_name] += 1
                self.is_intercept[inter_name] = True
            else:
                self.is_intercept[inter_name] = False
                self.interceptor_intercept_duration[inter_name] = 0

            # 结算
            if self.interceptor_intercept_duration[inter_name] >= self.intercept_time:
                # print(
                #     f"--- Step {self.current_step}: {inter_name} 成功拦截 {att_name}! ---"
                # )
                self.attackers_alive[att_name] = "dead"
                self.simulator.set_satellite_enable(att_name, False)  # 视觉上消失或变灰

    def _check_reassignment(self):
        """
        简化的重分配逻辑：
        如果有拦截者当前分配的目标已死，且场上还有活着的攻击者，
        则重新分配给最近的活着攻击者。
        """
        alive_attackers = [a for a, s in self.attackers_alive.items() if s == "alive"]
        if not alive_attackers:
            return

        for inter_name, target in self.assignments.items():
            if self.attackers_alive.get(target) == "dead":
                # 需要重分配
                my_pos = self.simulator.get_satellite_position(inter_name)

                # 找最近的活着的目标
                best_att = None
                min_dist = float("inf")
                for att in alive_attackers:
                    d = distance(my_pos, self.simulator.get_satellite_position(att))
                    if d < min_dist:
                        min_dist = d
                        best_att = att

                if best_att:
                    self.assignments[inter_name] = best_att
                    # 重置状态
                    self.interceptor_intercept_duration[inter_name] = 0
                    self.is_intercept[inter_name] = False
                    self.prev_distances.pop(inter_name, None)
                    # print(f"[Reassign] {inter_name} now targets {best_att}")

    def _check_hvt_breach(self) -> bool:
        hvt_pos = self.simulator.get_satellite_position(self.hvt_name)
        for att_name, status in self.attackers_alive.items():
            if status == "alive":
                d = distance(self.simulator.get_satellite_position(att_name), hvt_pos)
                if d < self.hvt_safe_distance:
                    # print(f"!!! HVT Breach by {att_name} !!!")
                    return True
        return False

    def _is_done(self) -> bool:
        if self.all_intercept_claimed:
            return True
        if self.is_hvt_breach:
            return True
        if self.collided:
            return True
        if len(self.out_satellite) > 0:
            return True
        return False

    def _is_outside(self):
        outsiders = []
        hvt_pos = self.simulator.get_satellite_position(self.hvt_name)
        for name in self.interceptor_names:
            pos = self.simulator.get_satellite_position(name)
            if distance(pos, hvt_pos) > self.outside_threshold:
                outsiders.append(name)
        return outsiders

    def _get_info(self) -> dict:
        return {
            "all_intercept_claimed": self.all_intercept_claimed,
            "is_hvt_breach": self.is_hvt_breach,
            "collided": self.collided,
            "current_step": self.current_step,
            "alive_attackers_count": sum(
                1 for s in self.attackers_alive.values() if s == "alive"
            ),
        }

    def _add_accel(self, name: str, accel: np.ndarray):
        # 兼容父类方法
        if self.on_plane:
            assert self.plane_normal is not None
            assert self.plane_point is not None
            accel = project_point_to_plane(accel, self.plane_normal, self.plane_point)

        if name in self.interceptor_names:
            pass  # 如果有 action queue 这里添加

        self.simulator.add_accel(name, accel * self.action_scale)
