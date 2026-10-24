import numpy as np

from ..RewardBase import RewardBase
from ...utils import distance


class RewardComputerPreyNvN(RewardBase):
    """
    高价值目标(HVT)捕获任务的多智能体奖励计算类。

    继承自 RewardBase，用于在多追踪者任务中计算奖励：
        1. 奖励基于智能体与HVT的距离及状态；
        2. 融合可达集思想，鼓励协同封堵和分散控制；
        3. 对过近或出界/碰撞的智能体给予负奖励；
        4. 对围绕目标形成理想分布（三角形）的智能体给予分散奖励；
        5. 考虑协作奖励和HVT靠近原点的惩罚/奖励。

    示例:
        >>> reward_model = RewardComputerPreyNvN()
        >>> rewards = reward_model.compute_reward()
        >>> print(rewards)
        {'A1': 0.95, 'A2': 1.10, 'A3': 0.88}
    """

    def __init__(self):
        """初始化奖励计算器，继承父类功能"""
        super(RewardBase, self).__init__()

    def compute_reward(self):
        """
        [NVN 适配奖励函数]
        计算 N 个拦截者对 M 个逃逸者的围捕奖励。

        逻辑：
        1. 距离奖励（Distance Reward）：基于每个逃逸者到其“最近”拦截者的距离。
           如果拦截者成功压缩了所有逃逸者的生存空间，奖励增加。
        2. 捕获奖励（Capture Reward）：当某个逃逸者距离小于阈值时给予稀疏大奖励。
        3. 碰撞惩罚（Collision Penalty）。
        4. 越界惩罚（Outside Penalty）。
        """
        rewards = {name: 0.0 for name in self.env.interceptor_names}

        # 获取位置
        interceptor_positions = {
            name: self.env.simulator.get_satellite_position(name)
            for name in self.env.interceptor_names
        }
        attacker_positions = {
            name: self.env.simulator.get_satellite_position(name)
            for name in self.env.attacker_names
        }

        # --- 1. 计算距离矩阵 (N x M) ---
        dists = np.zeros(
            (len(self.env.interceptor_names), len(self.env.attacker_names))
        )
        for i, int_name in enumerate(self.env.interceptor_names):
            for j, att_name in enumerate(self.env.attacker_names):
                dists[i, j] = distance(
                    interceptor_positions[int_name], attacker_positions[att_name]
                )

        # --- 2. 核心围捕逻辑：最小匹配距离 ---
        # 对每个逃逸者，找到离它最近的拦截者距离
        # 目标是最小化这些“最短距离”的总和
        min_dists_to_attackers = np.min(dists, axis=0)  # 形状 (M,)
        current_total_min_dist = np.sum(min_dists_to_attackers)

        # 奖励塑形：距离减小给予正奖励
        if self.env.current_step > 0:
            dist_improvement = self.env.prev_total_distance - current_total_min_dist
            # 缩放因子，使奖励数值在合理范围
            shaping_reward = dist_improvement * 5.0
        else:
            shaping_reward = 0.0

        self.env.prev_total_distance = current_total_min_dist

        # 将团队奖励分配给所有智能体 (Cooperative)
        for name in self.env.interceptor_names:
            rewards[name] += shaping_reward

        # --- 3. 捕获事件奖励 ---
        capture_bonus = 100.0
        newly_captured = 0

        for j, att_name in enumerate(self.env.attacker_names):
            if att_name in self.env.captured_evaders:
                continue

            # 如果该逃逸者的最近距离小于阈值
            if min_dists_to_attackers[j] < self.env.capture_distance_threshold_km:
                self.env.captured_evaders.add(att_name)
                newly_captured += 1

        if newly_captured > 0:
            # 捕获奖励全队共享
            for name in self.env.interceptor_names:
                rewards[name] += capture_bonus * newly_captured

        # --- 4. 碰撞惩罚 ---
        # 检查拦截者之间的碰撞
        for i, name_i in enumerate(self.env.interceptor_names):
            for j in range(i + 1, len(self.env.interceptor_names)):
                name_j = self.env.interceptor_names[j]
                if (
                    distance(
                        interceptor_positions[name_i], interceptor_positions[name_j]
                    )
                    < self.env.collision_threshold
                ):
                    rewards[name_i] -= 10.0
                    rewards[name_j] -= 10.0

        # --- 5. 越界惩罚 ---
        # 假设有一个中心点（如原点或HVT），超出范围给予惩罚
        for name in self.env.interceptor_names:
            pos = interceptor_positions[name]
            if np.linalg.norm(pos) > self.env.outside_threshold:
                rewards[name] -= 1.0  # 每步惩罚

        return rewards
