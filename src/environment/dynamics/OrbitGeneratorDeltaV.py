import numpy as np
from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit

from .IGenerator import IGenerator


class OrbitGeneratorDeltaV(IGenerator):
    """
    轨道状态生成器：用于在当前轨道状态基础上施加一次瞬时速度增量（ΔV），
    并将新的状态进行轨道传播以获得下一时刻的轨道状态。

    典型用途包括：
        - 轨道机动模拟（变轨、加速、制动）
        - 轨道调整策略测试
        - 冲量控制模型仿真

    示例:
        >>> generator = OrbitGeneratorDeltaV()
        >>> next_state = generator(state, delta_v=[0.01, 0, 0], step=60)
        >>> print(next_state)  # 输出下一时刻的轨道状态
    """

    def __call__(
        self,
        state: np.ndarray[(6,), np.float64],  # 前3维位置(km)，后3维速度(km/s)
        delta_v: np.ndarray[(3,), np.float64],  # 速度增量(km/s)
        step: float = 10,  # 传播时间(s)
    ) -> np.ndarray[(6,), np.float64]:
        """
        根据输入轨道状态施加一次瞬时速度增量（ΔV），并将轨道按指定时间传播，
        返回传播后的 [位置, 速度] 状态向量。

        Args:
            state (np.ndarray): 输入轨道状态，格式为 [x, y, z, vx, vy, vz]。
            delta_v (np.ndarray): 施加的速度增量向量 (ΔV)。
            step (float): 轨道传播的时间步长（单位: 秒）。

        Returns:
            np.ndarray: 传播后的轨道状态向量。

        示例:
            >>> generator = OrbitGeneratorDeltaV()
            >>> updated = generator(state, delta_v=np.array([0.05, 0, 0]), step=30)
            >>> print(updated)  # 输出更新后的 6 维状态
        """
        # 提取位置和速度
        r = state[:3]
        v = state[3:]

        # 应用速度增量
        new_v = v + delta_v

        # 创建临时轨道对象用于传播
        orbit = Orbit.from_vectors(attractor=Earth, r=r * u.km, v=new_v * u.km / u.s)

        # 传播到指定时间
        propagated_orbit = orbit.propagate(step * u.s)

        # 转换回状态数组
        return np.concatenate([propagated_orbit.r.value, propagated_orbit.v.value])
