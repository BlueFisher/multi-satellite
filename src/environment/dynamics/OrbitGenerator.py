from typing import Tuple

import numpy as np
from astropy import units as u
from poliastro.bodies import Earth
from poliastro.core.propagation import func_twobody
from poliastro.twobody import Orbit
from poliastro.twobody.propagation import CowellPropagator

from .IGenerator import IGenerator


class OrbitGenerator(IGenerator):
    def __call__(
        self,
        state: np.ndarray[(6,), np.float64],  # 前3维位置，后3维速度
        accel: np.ndarray[(3,), np.float64],
        step: float = 10,
    ) -> np.ndarray[(6,), np.float64]:
        """
        根据当前状态和加速度计算下一步状态

        Args:
            state: 状态数组 [x, y, z, vx, vy, vz] (单位: km, km/s)
            accel: 加速度向量 [ax, ay, az] (单位: km/s²)
            step: 时间步长 (单位: s)

        Returns:
            更新后的状态数组
        """
        if accel is None:
            accel = np.array([0.0, 0.0, 0.0])

        # 从状态数组提取位置和速度
        r = state[:3]  # 位置 (km)
        v = state[3:]  # 速度 (km/s)

        # 创建临时Orbit对象用于传播计算
        orbit = Orbit.from_vectors(Earth, r * u.km, v * u.km / u.s)

        # 传播函数，添加固定推力
        def f(t0, u_, k) -> Tuple[np.ndarray, np.ndarray]:
            du_kep = func_twobody(t0, u_, k)  # 开普勒项
            # 恒定推力项只影响速度 (单位转换: km/s² -> km/s²，保持一致)
            du_ad = np.concatenate((np.zeros(3), accel))
            return du_kep + du_ad

        # 执行传播
        propagated_orbit = orbit.propagate(
            value=step * u.s, method=CowellPropagator(f=f)
        )

        # 从传播后的轨道提取状态并转换为数组
        new_state = np.concatenate(
            [
                propagated_orbit.r.value,  # 位置 (km)
                propagated_orbit.v.value,  # 速度 (km/s)
            ]
        )

        return new_state


if __name__ == "__main__":
    r0 = [4e8, 0, 0] << u.m
    v0 = [0, 1022, 0] << (u.m / u.s)
    initial = Orbit.from_vectors(Earth, r0, v0)
    generator = OrbitGenerator()
    res = generator(initial, None)
    # res = initial.propagate(3 << u.day, method=CowellPropagator())
    print(res)
