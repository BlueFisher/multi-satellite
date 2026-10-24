import numpy as np

from ..RewardBase import RewardBase
from ...utils import distance


class RewardComputer3v1(RewardBase):
    """
    高价值目标(HVT)捕获任务的多智能体奖励计算类。

    继承自 RewardBase，用于在多追踪者任务中计算奖励：
        1. 奖励基于智能体与HVT的距离及状态；
        2. 融合可达集思想，鼓励协同封堵和分散控制；
        3. 对过近或出界/碰撞的智能体给予负奖励；
        4. 对围绕目标形成理想分布（三角形）的智能体给予分散奖励；
        5. 考虑协作奖励和HVT靠近原点的惩罚/奖励。

    示例:
        >>> reward_model = RewardCollision3Hvt()
        >>> rewards = reward_model.compute_reward()
        >>> print(rewards)
        {'A1': 0.95, 'A2': 1.10, 'A3': 0.88}
    """

    def __init__(self):
        """初始化奖励计算器，继承父类功能"""
        super(RewardBase, self).__init__()

    def compute_reward(self):
        """
        计算每个智能体的奖励值，基于其与高价值目标(HVT)的距离和状态。

        奖励计算规则：
            1. 若智能体已出界或碰撞，奖励为 -1；
            2. 若HVT处于危险状态，所有智能体奖励为 -1；
            3. 若HVT奖励已领取，所有智能体奖励为 100；
            4. 基础距离奖励根据智能体与HVT的捕获半径位置计算；
            5. 处于有效围捕区间的智能体，根据分散度形成三角形给予分散奖励；
            6. 根据处于理想距离的智能体数量给予协作奖励；
            7. 根据HVT靠近原点距离施加惩罚或微小奖励；
            8. 综合所有奖励和惩罚得到最终每个智能体的奖励值。

        Returns:
            Dict[str, float]: 每个智能体的奖励值，键为智能体名称，值为奖励分数。

        示例:
            >>> reward_model = RewardComputer3v1()
            >>> rewards = reward_model.compute_reward()
            >>> print(rewards)
            {'A1': 0.87, 'A2': 1.12, 'A3': 0.95}
        """

        # 获取捕获半径阈值
        capture_radius = self.env.capture_distance_threshold_km

        # 设置误差范围(捕获半径的10%)
        err = capture_radius * 0.1

        # 智能体间最小安全距离
        min_safe_distance = capture_radius * 0.5

        # 高价值目标保护半径
        hvt_protection_radius = 25.0  # 单位：km

        # 获取高价值目标(逃逸者)的位置和速度
        evader = self.env.simulator.get_satellite_position(self.env.attacker_name)

        # 计算逃逸者与原点的距离（高价值目标在坐标原点）
        evader_to_origin_distance = self.env.simulator.get_distance_to_real(
            self.env.attacker_name, self.env.hvt_name
        )

        # 初始化奖励字典
        ret = {name: 0 for name in self.env.agent_names}

        # 检查高价值目标是否处于危险状态
        if hasattr(self.env, "hvt_unsafe") and self.env.hvt_unsafe:
            return {name: -1 for name in self.env.agent_names}

        if self.env.capture_reward_claimed:
            return {name: 100.0 for name in self.env.agent_names}

        # 计数器：记录处于理想捕获距离范围内的智能体数量
        count = 0

        # 记录处于有效围捕区间的智能体位置
        active_agent_positions = {}

        # 获取所有正常智能体的位置
        agent_positions = {}
        for name in self.env.agent_names:
            pos = self.env.simulator.get_satellite_position(name)
            agent_positions[name] = pos
            # 检查是否在有效围捕区间
            dis = distance(pos, evader)
            if capture_radius - err <= dis < capture_radius + err:
                active_agent_positions[name] = pos
                count += 1

        # 计算分散奖励（根据有效围捕区间内的智能体数量决定）
        dispersion_reward = 0.0
        # 只有当2个或3个卫星在有效围捕区间时才计算角度奖励
        if count == 2 or count == 3:
            dispersion_reward = self._calculate_dispersion_reward(
                active_agent_positions, evader
            )
            # 两个卫星时奖励打折
            if count == 2:
                dispersion_reward *= 0.3

        # 遍历所有智能体
        for name in self.env.agent_names:
            # 处理已出界或已碰撞的智能体
            if name in self.env.out_satellite:
                ret[name] = -1
                continue
            if name in self.env.collided_satellite:
                ret[name] = -1
                continue

            # 获取当前智能体位置并计算与目标的距离
            position = self.env.simulator.get_satellite_position(name)
            dis = distance(position, evader)

            # # 处理过于接近目标的智能体(距离小于捕获半径-误差)
            # if dis < capture_radius - err:
            #     ret[name] = -0.1
            #     continue

            # 基础距离奖励
            if dis < capture_radius + err:
                if not self.env.reward_side:
                    ret[name] = 1 + np.exp(-((abs(dis - capture_radius) / err) ** 2))
                else:
                    ret[name] = 1 + (abs(dis - capture_radius) / err) ** 2
            # else:
            #     # 计算远离目标的智能体奖励(随距离增加线性递减)
            #     ret[name] = 1 - (dis - capture_radius) / (50 - capture_radius)

            # 检查与其他智能体的距离，太近则给予惩罚
            proximity_penalty = 0
            for other_name, other_pos in agent_positions.items():
                if other_name == name:  # 跳过自己
                    continue

                # 计算与其他智能体的距离
                inter_distance = distance(position, other_pos)

                # 如果距离小于安全阈值，给予惩罚
                if inter_distance < min_safe_distance:
                    # 惩罚值随距离减小而增加
                    proximity_penalty += (
                        (min_safe_distance - inter_distance) / min_safe_distance * 0.5
                    )

            # 应用距离惩罚
            ret[name] -= proximity_penalty

            # 应用分散奖励（只有在有效围捕区间的智能体数量符合条件时才有）
            if name in active_agent_positions:
                ret[name] += dispersion_reward

        # 添加协作奖励：根据处于理想距离的智能体数量给予额外奖励
        collaboration_bonus = count / len(self.env.agent_names)
        for key in ret.keys():
            ret[key] += collaboration_bonus

        # 根据HVT靠近原点的距离计算惩罚或者奖励
        penalty_strength = 1 - (evader_to_origin_distance / hvt_protection_radius)
        penalty = np.clip(penalty_strength, -1, 0)  # 最大惩罚为 1， hvt远离有一定奖励

        # 时间惩罚
        penalty += 0.05
        # 对所有智能体应用惩罚
        for key in ret.keys():
            ret[key] -= penalty

        # 对于输出和上一次差异过大的加速度给予惩罚
        if self.env.reward_min_accel:
            limit_accel_satellite = self.env.simulator.limit_accel_satellites
            for name in self.env.agent_names:
                if name in limit_accel_satellite:
                    ret[name] -= 0.5

        return ret

    def _calculate_dispersion_reward(self, agent_positions, evader_pos):
        """
        计算分散奖励：评估智能体是否围绕目标形成严格的三角形分布

        参数:
            agent_positions: 有效围捕区间内的智能体位置字典 {name: position}
            evader_pos: 目标位置

        返回:
            dispersion_reward: 分散奖励值，范围[0, 1]
        """
        num_agents = len(agent_positions)

        # 三角形需要恰好3个智能体
        if num_agents != 3:
            return 0.0

        # 获取三个智能体的位置
        positions = list(agent_positions.values())
        a, b, c = positions

        # 计算从目标指向每个智能体的向量
        vec_a = a - evader_pos
        vec_b = b - evader_pos
        vec_c = c - evader_pos

        # 计算各向量的模长（距离）
        dist_a = np.linalg.norm(vec_a)
        dist_b = np.linalg.norm(vec_b)
        dist_c = np.linalg.norm(vec_c)

        # 计算三个智能体之间的距离
        ab = np.linalg.norm(a - b)
        bc = np.linalg.norm(b - c)
        ca = np.linalg.norm(c - a)

        # 1. 检查是否形成等边三角形（角度均为60度左右）
        # 使用余弦定理计算三个角度的余弦值
        def angle_cosine(d1, d2, d3):

            return (d1**2 + d2**2 - d3**2) / (2 * d1 * d2 + 1e-10)  # 避免除零

        cos_ab = angle_cosine(dist_a, dist_b, ab)  # 目标处a和b的夹角余弦
        cos_bc = angle_cosine(dist_b, dist_c, bc)  # 目标处b和c的夹角余弦
        cos_ca = angle_cosine(dist_c, dist_a, ca)  # 目标处c和a的夹角余弦

        # 理想的等边三角形每个角为120度（从中心看），cos(120°) = -0.5
        ideal_cos = np.cos(np.radians(120))  # 约等于-0.5

        # 计算角度偏差
        angle_deviation = np.mean(
            [abs(cos_ab - ideal_cos), abs(cos_bc - ideal_cos), abs(cos_ca - ideal_cos)]
        )
        angle_reward = 1 - np.clip(angle_deviation / (abs(ideal_cos) * 2), 0, 1)

        # 2. 检查距离是否大致相等（等边三角形边长相等）
        mean_dist = np.mean([dist_a, dist_b, dist_c])
        dist_deviation = np.std([dist_a, dist_b, dist_c]) / (mean_dist + 1e-10)
        distance_reward = 1 - np.clip(dist_deviation, 0, 1)

        # 3. 确保三个点不共线（形成有效三角形）
        # 计算面积的两倍（通过叉积）
        area_twice = np.abs(np.dot(vec_a, np.cross(vec_b, vec_c)))
        # 归一化面积（与最大可能面积相比）
        max_area = (
            np.prod([dist_a, dist_b, dist_c]) * np.sqrt(3)
        ) / 4  # 等边三角形最大面积
        collinearity_reward = np.clip(area_twice / (max_area + 1e-10), 0, 1)

        # 综合奖励：角度、距离和共线性的权重
        dispersion_reward = (
            0.4 * angle_reward + 0.3 * distance_reward + 0.3 * collinearity_reward
        )

        # 确保奖励在[0, 1]范围内
        return np.clip(dispersion_reward, 0.0, 1.0)

    # def _calculate_dispersion_reward(self, agent_positions, evader_pos):
    #     """
    #     计算分散奖励：考虑角度和距离因素，鼓励智能体在圆周上均匀分布
    #
    #     参数:
    #         agent_positions: 有效围捕区间内的智能体位置字典 {name: position}
    #         evader_pos: 目标位置
    #
    #     返回:
    #         dispersion_reward: 分散奖励值，范围[0, 1]
    #     """
    #     num_agents = len(agent_positions)
    #     if num_agents < 2:
    #         return 0.0  # 单个智能体无法分散
    #
    #     # 计算所有智能体相对于目标的角度和距离
    #     angles = []
    #     distances = []
    #     for pos in agent_positions.values():
    #         # 计算从目标指向智能体的向量
    #         vec = pos - evader_pos
    #         # 计算在XY平面内的角度
    #         angle = np.arctan2(vec[1], vec[0])
    #         angles.append(angle)
    #         distances.append(np.linalg.norm(vec))
    #
    #     # 确保角度在[0, 2π)范围内并排序
    #     angles = np.array(angles) % (2 * np.pi)
    #     angles.sort()
    #
    #     # 计算相邻角度之间的间隙（包括最后一个到第一个的间隙）
    #     gaps = []
    #     for i in range(num_agents):
    #         next_angle = angles[(i + 1) % num_agents]
    #         gap = next_angle - angles[i]
    #         # 处理跨0点的间隙
    #         if gap < 0:
    #             gap += 2 * np.pi
    #         gaps.append(gap)
    #
    #     # 理想情况下的角度间隙
    #     ideal_gap = 2 * np.pi / num_agents
    #
    #     # 1. 计算角度间隙与理想值的偏差（归一化）
    #     gap_ratios = np.array(gaps) / ideal_gap
    #     gap_deviation = np.std(gap_ratios)
    #     angle_reward = 1 - gap_deviation  # 标准差越小，奖励越高
    #
    #     # 2. 计算距离均匀性奖励
    #     mean_distance = np.mean(distances)
    #     if mean_distance == 0:
    #         distance_reward = 0.0
    #     else:
    #         # 使用距离的相对标准差（变异系数）
    #         distance_deviation = np.std(distances) / mean_distance
    #         distance_reward = 1 - distance_deviation  # 距离越均匀，奖励越高
    #
    #     # 综合奖励计算：角度和距离权重
    #     dispersion_reward = 0.7 * angle_reward + 0.3 * distance_reward
    #
    #     # 确保奖励在[0, 1]范围内
    #     return np.clip(dispersion_reward, 0.0, 1.0)
