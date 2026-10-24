import numpy as np

from ..RewardBase import RewardBase
from ...utils import distance


class RewardComputerInterceptNvN(RewardBase):
    """
    高价值目标(HVT)捕获任务的多智能体奖励计算类。

    继承自 RewardBase，用于在多追踪者任务中计算奖励：
        1. 奖励基于智能体与HVT的距离及状态；
        2. 融合可达集思想，鼓励协同封堵和分散控制；
        3. 对过近或出界/碰撞的智能体给予负奖励；
        4. 对围绕目标形成理想分布（三角形）的智能体给予分散奖励；
        5. 考虑协作奖励和HVT靠近原点的惩罚/奖励。

    示例:
        >>> reward_model = RewardComputerInterceptNvN()
        >>> rewards = reward_model.compute_reward()
        >>> print(rewards)
        {'A1': 0.95, 'A2': 1.10, 'A3': 0.88}
    """

    def __init__(self):
        """初始化奖励计算器，继承父类功能"""
        super(RewardBase, self).__init__()

    def compute_reward(self):
        """
        NVN 拦截任务奖励函数
        """
        rewards = {name: 0.0 for name in self.env.interceptor_names}
        hvt_pos = self.env.simulator.get_satellite_position(self.hvt_name)

        # 全局状态检查
        game_over_penalty = 0.0
        if self.env.is_hvt_breach:
            game_over_penalty = -100.0

        # 遍历每个拦截者计算奖励
        for name in self.env.interceptor_names:
            r = 0.0

            # 基础生存/任务步数惩罚 (鼓励快速完成)
            r -= 0.05

            assigned_att = self.env.assignments.get(name)
            if not assigned_att:
                rewards[name] = r
                continue

            # 获取状态
            my_pos = self.env.simulator.get_satellite_position(name)
            att_pos = self.env.simulator.get_satellite_position(assigned_att)

            # --- 1. 拦截引导奖励 (Interception Guidance) ---
            # 如果目标还活着，鼓励接近并形成拦截态势
            if self.env.attackers_alive.get(assigned_att) == "alive":
                dist = distance(my_pos, att_pos)

                # Shaping Reward: 距离减小给予奖励
                prev_dist = self.env.prev_distances.get(name, dist)
                r += (prev_dist - dist) * 2.0
                self.env.prev_distances[name] = dist

                # 角度与位置逻辑
                vec_ai = my_pos - att_pos
                vec_ah = hvt_pos - att_pos
                d_ai = np.linalg.norm(vec_ai)
                d_ah = np.linalg.norm(vec_ah)

                # 计算夹角余弦
                dot = np.dot(vec_ai, vec_ah)
                cos_theta = dot / ((d_ai * d_ah) + 1e-9)

                # 如果处于正确的位置 (在 A 和 H 之间) 且 角度合适
                if d_ai < d_ah and cos_theta > np.cos(np.deg2rad(30)):  # 30度宽限
                    r += 0.05

                # 如果满足严格拦截条件 (在 _check_interception 中判断为 True)
                if self.env.is_intercept.get(name, False):
                    r += 0.2  # 持续保持拦截姿态的奖励

            # --- 2. 成功拦截奖励 (Success Reward) ---
            # 由环境逻辑触发 (target dead)，这里不重复给大分，
            # 可以在 dead 的一瞬间给，或者在这里判断状态
            if self.env.attackers_alive.get(assigned_att) == "dead":
                # 目标已死，给予持续的小额奖励或零，避免Agent为了刷分去鞭尸
                # 或者引导去帮助队友 (暂未实现)
                r += 0.0

            # --- 3. 惩罚项 ---
            # 出界惩罚
            if (
                self.env.simulator.get_distance_to_real(name, self.env.hvt_name)
                > self.env.outside_threshold
            ):
                r -= 0.5

            # 碰撞惩罚 (与队友或HVT)
            # 简单的距离检查
            for teammate in self.env.interceptor_names:
                if teammate != name:
                    t_pos = self.env.simulator.get_satellite_position(teammate)
                    if distance(my_pos, t_pos) < self.env.collision_threshold:
                        r -= 1.0

            if distance(my_pos, hvt_pos) < self.env.collision_threshold:
                r -= 5.0  # 严禁撞击HVT

            # 加上全局游戏结束惩罚
            r += game_over_penalty

            rewards[name] = r

        return rewards
