import numpy as np
from ..RewardBase import RewardBase
from typing import Dict


class RewardComputer2v2(RewardBase):
    """
    多智能体2v2拦截任务奖励计算器。

    功能：
        1. 计算拦截者的引导奖励（距离引导、站位引导、HVT距离引导、避碰引导）；
        2. 计算拦截成功奖励和维持奖励；
        3. 对失败或违规行为施加惩罚；
        4. 综合各类奖励和惩罚生成最终奖励字典。

    属性:
        w_lead_dist: 距离引导权重
        w_lead_angle: 站位引导权重
        w_lead_hvt: HVT距离引导权重
        w_lead_collision: 避碰引导权重
        reward_intercept_success: 拦截成功奖励
        reward_maintain_intercept: 维持奖励
        reward_false: 失败惩罚
        penalty_time: 时间步惩罚
        previous_distance: 前一步距离字典
        env: 当前环境对象

    示例:
        >>> reward_model = RewardComputer2v2()
        >>> reward_model.env = env_2v2_instance
        >>> rewards = reward_model.compute_reward()
        >>> print(rewards)
        {'I1': 1.32, 'I2': 1.45}
    """

    def __init__(self):
        # 奖励权重和值
        super().__init__()
        self.w_lead_dist = 0.4  # 引导奖励权重——距离引导权重
        self.w_lead_angle = 0.6  # 引导奖励权重——站位引导权重
        self.w_lead_hvt = 0.3  # 引导奖励权重——离hvt超过40km
        self.w_lead_collision = 0.2  # 引导奖励权重——两土豆距离小于5km

        self.reward_intercept_success = 100.0  # 拦截成功奖励
        self.reward_maintain_intercept = 1.0  # 持续维持拦截姿态的奖励

        self.reward_false = -1.0  # 失败惩罚
        self.penalty_time = -0.01  # 时间步惩罚

        # 状态变量
        self.previous_distance: Dict[str, float] = (
            {}
        )  # 前序距离，在计算引导奖励中使用，存储上一步距离的字典

    def reset(self):
        """
        重置奖励计算器状态，清空前一步距离记录。

        示例:
            >>> reward_model.reset()
        """
        self.previous_distance.clear()

    def compute_reward(self) -> Dict[str, float]:
        """
        计算当前时间步所有拦截者的综合奖励。

        奖励计算流程：
            1. 检查失败条件并应用失败惩罚；
            2. 计算引导奖励（距离引导 + 站位引导）；
            3. 计算拦截成功奖励；
            4. 计算维持奖励；
            5. 将各类奖励与时间惩罚组合得到最终奖励；
            6. 更新前序距离信息。

        返回:
            Dict[str, float]: 每个拦截者的奖励值字典。

        示例:
            >>> rewards = reward_model.compute_reward()
            >>> print(rewards)
            {'I1': 1.05, 'I2': 0.98}
        """
        attacker_names = self.env.attacker_names
        interceptor_names = self.env.interceptor_names
        hvt_name = self.env.hvt_name

        def _get_states(agent_names):
            state = {}
            for name in agent_names:
                try:
                    state[name] = self.env.simulator.get_satellite_state(name)
                except KeyError:
                    print(f"Can't find agent {name}")
            return state

        # ECI[x, y, z, vx, vy, vz]
        attacker_states = _get_states(attacker_names)
        interceptor_states = _get_states(interceptor_names)
        hvt_state = self.env.simulator.get_satellite_state(hvt_name)

        """3.失败惩罚（土豆碰撞/超过50km/d(攻击-hvt)<5km）
            失败惩罚结果  float  -1.0               """
        r_false = self._compute_r_false()
        # print(f"碰撞奖励：{r_false}", end="  ")
        final_rewards: Dict[str, float] = {}
        if r_false != 0.0:
            final_rewards = {name: r_false for name in interceptor_names}
            return final_rewards

        """1.引导奖励（距离引导、站位引导）
           引导奖励结果 dict[str, float]       """
        r_leading = self._compute_r_leading(
            attacker_states, interceptor_states, hvt_state
        )
        # print(f"引导奖励：{r_leading}", end="  ")

        """2.1 拦截成功奖励     d(攻击者-拦截者)<0.1km、d(攻击者-拦截者)<d(攻击者-高价值目标)、站位角度正确(<15°)
           拦截奖励结果  dict[str, float]       """
        r_result = self._compute_r_result()
        # print(f"拦截成功奖励：{r_result}", end="  ")

        """2.2 拦截维持奖励 """
        r_maintain = self._compute_r_maintain(attacker_states, interceptor_states)
        # print(f"拦截维持奖励：{r_maintain}", end="  ")

        """总奖励（加权系数 w1 * r_leading + w2 * r_result + w3 * r_defense + w4 * r_false + w5 * r_time）"""
        for agent_name in interceptor_names:
            assigned_attacker = self.env.assignments[agent_name]
            is_task_done = self.env.attackers_alive[assigned_attacker] == "dead"

            time_penalty = 0.0 if is_task_done else self.penalty_time
            # leading_reward = 0.0 if is_task_done else r_leading.get(agent_name, 0.0)

            total_reward = (
                r_leading.get(agent_name, 0.0)  # [-1,1]
                + r_result.get(agent_name, 0.0)  # 10.0
                + r_maintain.get(agent_name, 0.0)  # 1.0
                + time_penalty  # -0.01
            )

            final_rewards[agent_name] = total_reward
        r_total = final_rewards
        # print(f"总奖励：{r_total}")
        for inter_name, att_name in self.env.assignments.items():
            if self.env.attackers_alive.get(att_name) == "dead":
                self.previous_distance.pop(inter_name, None)
        return r_total

    def _compute_r_leading(
        self,
        attacker_states: Dict[str, np.ndarray],
        interceptor_states: Dict[str, np.ndarray],
        hvt_state: np.ndarray,
    ) -> Dict[str, float]:
        """
        计算引导奖励，由“距离引导”和“站位引导”两部分组成。只引导存活目标
        Args:
            attacker_states: 所有攻击卫星的当前状态字典。
            interceptor_states: 所有拦截卫星的当前状态字典。
            hvt_state: 高价值目标的当前状态向量。
        Returns:
            Dict[str, float]。
        """
        interceptor_leading_rewards: Dict[str, float] = {}
        epsilon = 1e-6
        # 1. 遍历固定的分配对
        for interceptor_name, attacker_name in self.env.assignments.items():
            # 如果目标攻击卫星已经“死亡”，则跳过此对的引导奖励计算
            if self.env.attackers_alive[attacker_name] == "dead":
                continue

            pos_i = interceptor_states[interceptor_name][:3]
            pos_a = attacker_states[attacker_name][:3]
            pos_hvt = hvt_state[:3]

            # --- Part A: 距离引导 (鼓励靠近) ---估计[-0.3,0.3]*4——>[-1,1]
            current_dist = self.env.simulator.get_distance_to_real(
                interceptor_name, attacker_name
            )

            last_dist = self.previous_distance.get(interceptor_name, current_dist)
            r_dist = last_dist - current_dist
            r_dist = r_dist * 4  # [-1,1]
            self.previous_distance[interceptor_name] = current_dist
            if (
                current_dist < self.env.intercept_distance_threshold_km * 0.9
            ):  # 当距离小于捕获阈值的 90%，停止距离引导奖励
                r_dist = 0.0

            # --- Part B: 站位引导 (鼓励形成 A-I-HVT 站位) ---[-1,1]
            vec_ai = pos_i - pos_a
            vec_ah = pos_hvt - pos_a
            cos_theta = np.dot(vec_ai, vec_ah) / (
                np.linalg.norm(vec_ai) * np.linalg.norm(vec_ah) + epsilon
            )
            r_angle = cos_theta  # r_angle 的范围是 [-1, 1]

            # --- Part C: HVT 距离引导 (避免离 HVT过远，若距离大于 40km 给负奖励) ---[0,1]
            dist_to_hvt = self.env.simulator.get_distance_to_real(
                interceptor_name, self.env.hvt_name
            )
            r_hvt = 0.0
            if dist_to_hvt > 40.0:
                # 若距离大于 40km，给予负奖励
                r_hvt = dist_to_hvt - 40.0  # [0,10]
                r_hvt = r_hvt / 10  # [0,1]

            # --- Part D: 避碰引导（避免与其他碰撞）---[0,1]
            # r_collision_penalty = 0.0
            # safe_radius = 5.0
            # epsilon = 1e-6
            # # 拦截者 – 拦截者
            # for other_name, other_state in interceptor_states.items():      # [0,1]
            #     if other_name == interceptor_name:
            #         continue
            #     dist = np.linalg.norm(pos_i - other_state[:3])
            #     if dist < safe_radius:
            #         r_collision_penalty += (safe_radius - dist) / (safe_radius + epsilon)
            # # 拦截者 – 攻击者
            # for a_name, a_state in attacker_states.items():                 # [0,2]
            #     dist = np.linalg.norm(pos_i - a_state[:3])
            #     if dist < safe_radius:
            #         r_collision_penalty += (safe_radius - dist) / (safe_radius + epsilon)
            # # 拦截者 – HVT
            # dist_hvt = np.linalg.norm(pos_i - pos_hvt)                      # [0,1]
            # if dist_hvt < safe_radius:
            #     r_collision_penalty += (safe_radius - dist_hvt) / (safe_radius + epsilon)
            #
            # r_collision_penalty = r_collision_penalty / 4

            # --- 组合奖励 --- 0.4  0.6  0.3 0.2
            pair_reward = (
                self.w_lead_dist * r_dist
                + self.w_lead_angle * r_angle
                - self.w_lead_hvt * r_hvt
            )  # - self.w_lead_collision * r_collision_penalty
            interceptor_leading_rewards[interceptor_name] = pair_reward

        return interceptor_leading_rewards

    def _compute_r_result(self) -> Dict[str, float]:
        """
        计算拦截成功奖励。

        条件：
            1. 拦截者与攻击者距离在 0.9d <= d <= 1.1d；
            2. 拦截维持时间超过指定步数。

        返回:
            Dict[str, float]: 每个拦截者的拦截成功奖励

        示例:
            >>> r_result = reward_model._compute_r_result()
            >>> print(r_result)
            {'I1': 10.0, 'I2': 0.0}
        """
        interceptor_results_rewards: Dict[str, float] = {}

        # for interceptor_name, is_success in self.env.is_intercept.items():
        #     if (
        #         is_success
        #         and self.env.interceptor_intercept_duration[interceptor_name]
        #         >= self.env.intercept_time
        #     ):  # >=20步
        #         reward = self.reward_intercept_success
        #     else:
        #         reward = 0.0
        for interceptor_name in self.env.interceptor_names:
            # 直接读取拦截成功事件标志发钱
            if getattr(self.env, 'intercept_success_flags', {}).get(interceptor_name, False):
                reward = self.reward_intercept_success
            else:
                reward = 0.0
            interceptor_results_rewards[interceptor_name] = reward

        return interceptor_results_rewards

    def _compute_r_maintain(
        self,
        attacker_states: Dict[str, np.ndarray],
        interceptor_states: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """
        计算维持奖励，奖励当前仍在拦截状态的拦截者。

        参数:
            attacker_states: Dict[str, np.ndarray], 攻击者当前状态字典
            interceptor_states: Dict[str, np.ndarray], 拦截者当前状态字典

        返回:
            Dict[str, float]: 每个拦截者的维持奖励值

        示例:
            >>> r_maintain = reward_model._compute_r_maintain(attacker_states, interceptor_states)
            >>> print(r_maintain)
            {'I1': 1.82, 'I2': 1.0}
        """

        intercept_threshold = self.env.intercept_distance_threshold_km
        error_percentage = 0.10  # 10%的误差容忍
        error_margin = intercept_threshold * error_percentage

        interceptor_maintain_rewards: Dict[str, float] = {}
        for name, is_maintaining in self.env.is_intercept.items():
            interceptor_maintain_rewards[name] = 0.0
            if is_maintaining:
                # 1. 获取对应的拦截者和攻击者位置
                try:
                    attacker_name = self.env.assignments[name]
                    pos_i = interceptor_states[name][:3]
                    pos_a = attacker_states[attacker_name][:3]
                except KeyError:
                    # 如果找不到分配关系或状态，则跳过
                    continue
                # 2. 计算当前距离
                current_dist = self.env.simulator.get_distance_to_real(
                    name, attacker_name
                )
                abs_error = abs(current_dist - intercept_threshold)
                proximity_score = np.exp(-((abs_error / error_margin) ** 2))
                interceptor_maintain_rewards[name] = (
                    1.0 + proximity_score
                )  # self.reward_maintain_intercept * (self.env.interceptor_intercept_duration[name] / self.env.intercept_time)
        return interceptor_maintain_rewards

    def _compute_r_false(self) -> float:
        """
        计算失败惩罚。

        条件：
            1. 任意两土豆距离小于1km；
            2. 攻击者与拦截者距离超过50km；
            3. 攻击者与HVT距离小于5km；
            4. 或者智能体出界、碰撞、HVT被突破。

        返回:
            float: 失败惩罚值（-1.0或0.0）

        示例:
            >>> r_false = reward_model._compute_r_false()
            >>> print(r_false)
            -1.0
        """
        if (
            self.env.collided
            or len(self.env.out_satellite) > 0
            or self.env.is_hvt_breach
        ):  # self.env.is_out_boundary
            return self.reward_false
        else:
            return 0.0
