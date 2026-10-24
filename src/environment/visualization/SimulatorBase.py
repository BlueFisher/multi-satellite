import abc
from typing import List, Tuple

import numpy as np
from astropy import units as u
from astropy.time import Time
from poliastro.bodies import Earth
from poliastro.twobody import Orbit

from .utils import eci2lvlh, lvlh2eci
from ..dynamics.IGenerator import IGenerator
from ..dynamics.OrbitGeneratorCW import OrbitGeneratorCW


class SimulatorBase:

    def __init__(
        self,
        position_generator: IGenerator,
        update_interval: float = 10,
        render: bool = True,
        store_lvlh=True,
        enable_range_error: bool = False,
        range_error_range: Tuple[float, float] = (0.030, 0.050),  # 30-50m范围误差
        enable_angle_error: bool = False,
        angle_error_range: Tuple[float, float] = (0.0, 0.1),  # 0-0.1度范围误差
        limit_accel: bool = True,
        limit_angle: float = 2,  # 约束速度变化角度不能超过 2°/s
        limit_min_accel: float = 0.00002,  # 小于 0.00002 km/s^2 不进行约束
    ):
        self.render = render
        self.position_generator = position_generator
        self.update_interval = update_interval

        self.store_lvlh = store_lvlh
        self.origin_satellite_name = None
        self.origin_satellite = None
        self.origin_position_generator = None

        self.limit_accel = limit_accel
        self.limit_angle = limit_angle
        self.limit_min_accel = limit_min_accel
        self.history_accel = {}
        self.limit_accel_satellites = set({})

        # 传感器误差配置
        self.light_factor = 1.0
        self.enable_range_error = enable_range_error
        self.range_error_range = range_error_range  # (min_error, max_error) 单位：km

        self.enable_angle_error = enable_angle_error
        self.angle_error_range = angle_error_range  # (min_error, max_error) 单位：度
        # 随机数种子
        self.rng = np.random.default_rng()
        # 意图识别信息
        self.intention_message = None

    @abc.abstractmethod
    def agent_names(self) -> List[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def group_names(self) -> List[str]:
        raise NotImplementedError

    @abc.abstractmethod
    def start(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def stop(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def update(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def render_frame(self) -> None:
        pass

    def add_accel(self, name: str, accel: np.ndarray):

        if self.limit_accel:
            accel = self._limit_accel(name, accel)

        self._add_accel(name, accel)

    @abc.abstractmethod
    def _add_accel(self, name: str, accel: np.ndarray[(3,), np.float64]) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def get_satellite(self, name: str) -> np.ndarray[(6,), np.float64]:
        raise NotImplementedError

    def get_satellite_velocity(self, name: str) -> np.array:
        return self.get_satellite(name)[3:].copy()

    def get_satellite_position(self, name: str) -> np.array:
        return self.get_satellite(name)[:3].copy()

    def get_satellite_state(self, name: str) -> np.array:
        """
        如果 store_lvlh = True 返回 lvlh 坐标系数据
        否则返回 eci 数据
        """
        return self.get_satellite(name)

    def get_all_satellite_states_eci(self) -> dict[str, np.ndarray]:
        """返回全部卫星的 ECI 状态快照。

        返回字典的顺序与 :meth:`agent_names` 一致。模拟器内部以 LVLH
        保存状态时，本方法会将普通卫星转换为 ECI，并补入单独保存的
        原点卫星状态。所有数组均为副本，调用方不能借此修改模拟器状态。
        """

        names = self.agent_names()
        if not self.store_lvlh:
            return {
                name: np.asarray(self.get_satellite(name), dtype=float).copy()
                for name in names
            }

        if self.origin_satellite_name is None or self.origin_satellite is None:
            raise RuntimeError("LVLH simulator has no initialized origin satellite")

        origin_state = np.asarray(self.get_origin_satellite_state(), dtype=float)
        states: dict[str, np.ndarray] = {}
        for name in names:
            if name == self.origin_satellite_name:
                states[name] = origin_state.copy()
            else:
                states[name] = lvlh2eci(
                    np.asarray(self.get_satellite(name), dtype=float), origin_state
                )
        return states

    def get_distance_to_real(self, origin: str, target: str) -> np.float32:
        origin_pos = self.get_satellite_position(origin)
        target_pos = self.get_satellite_position(target)

        # 计算实际距离
        actual_distance = np.linalg.norm(target_pos - origin_pos)

        return actual_distance

    def get_distance_to(self, origin: str, target: str) -> np.float32:
        """
        计算两颗卫星之间的距离（可配置是否添加范围误差）

        参数:
            origin: 源卫星名称
            target: 目标卫星名称

        返回:
            float: 距离测量值(km)
        """
        # 获取卫星位置
        actual_distance = self.get_distance_to_real(origin, target)

        # 根据配置添加范围误差
        if self.enable_range_error and self.range_error_range[1] > 0:
            # 在误差范围内生成随机误差值
            error_min, error_max = self.range_error_range
            # 生成均匀分布的误差值
            error_value = self.rng.uniform(error_min, error_max) * (
                1 - self.light_factor
            )
            # 随机选择误差的正负号
            sign = self.rng.choice([-1, 1])
            measurement_distance = actual_distance + sign * error_value
            return measurement_distance
        else:
            return actual_distance

    def get_angle_to_real(self, origin: str, target: str) -> np.ndarray:
        # 获取卫星状态
        origin_state = self.get_satellite_state(origin)
        target_state = self.get_satellite_state(target)

        # 计算相对位置
        if self.store_lvlh:
            # LVLH坐标系下直接计算相对位置
            rel_pos = target_state[:3] - origin_state[:3]
        else:
            # ECI坐标系，需要转换到LVLH
            origin_full_state = self.get_satellite(origin)
            target_full_state = self.get_satellite(target)
            # 假设eci2lvlh函数可用，与类中其他地方一致
            rel_pos = eci2lvlh(origin_full_state, target_full_state)[:3]

        # 计算方位角 (在x-y平面上的投影与y轴的夹角)
        azimuth = np.arctan2(rel_pos[0], rel_pos[1]) * 180.0 / np.pi

        # 计算俯仰角 (与x-y平面的夹角)
        xy_norm = np.sqrt(rel_pos[0] ** 2 + rel_pos[1] ** 2)
        elevation = np.arctan2(rel_pos[2], xy_norm) * 180.0 / np.pi

        # 确保角度在合理的范围内
        azimuth = azimuth % 360.0
        if azimuth > 180.0:
            azimuth -= 360.0

        return np.array([azimuth, elevation], dtype=np.float32)

    def get_angle_to(self, origin: str, target: str) -> np.ndarray:
        """
        计算目标相对于源卫星的方位角和俯仰角（在源卫星的LVLH坐标系中）

        参数:
            origin: 源卫星名称
            target: 目标卫星名称

        返回:
            np.ndarray: 角度测量值 [方位角(deg), 俯仰角(deg)]
        """

        # 根据配置添加范围误差

        real = self.get_angle_to_real(origin, target)
        azimuth = real[0]
        elevation = real[1]

        if self.enable_angle_error and self.angle_error_range[1] > 0:
            error_min, error_max = self.angle_error_range

            # 为方位角和俯仰角分别生成误差
            azimuth_error = self.rng.uniform(error_min, error_max) * (
                1 - self.light_factor
            )
            elevation_error = self.rng.uniform(error_min, error_max) * (
                1 - self.light_factor
            )

            # 随机选择误差的正负号
            azimuth_sign = self.rng.choice([-1, 1])
            elevation_sign = self.rng.choice([-1, 1])

            azimuth += azimuth_sign * azimuth_error
            elevation += elevation_sign * elevation_error

        return np.array([azimuth, elevation], dtype=np.float32)

    def get_origin_satellite_state(self):
        """
        返回的 hvt 的 eci 位置和速度
        """
        return np.concatenate(
            [self.origin_satellite.r.value, self.origin_satellite.v.value]
        )

    def get_origin_satellite_position(self, name: str) -> np.array:
        """
        返回的 hvt 的 eci 位置
        """
        return self.origin_satellite.r.value

    def get_origin_satellite_velocity(self, name: str) -> np.array:
        """
        返回的 hvt 的 eci 速度
        """
        return self.origin_satellite.v.value

    @abc.abstractmethod
    def clear(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _set_origin_satellite(self, color="yellow"):
        raise NotImplementedError

    def set_center_satellite(self, name: str) -> None:
        pass

    def _update_origin_satellite(self):
        pass

    def set_origin_satellite_by_classical(
        self, prefix: str, states: np.ndarray[(6,), np.float64]
    ) -> None:
        if self.store_lvlh:
            self.origin_satellite_name = prefix + "_0"
            self.origin_satellite = Orbit.from_classical(
                Earth,
                states[0] << u.km,
                states[1] << u.one,
                states[2] << u.deg,
                states[3] << u.deg,
                states[4] << u.deg,
                states[5] << u.deg,
            )

            if not isinstance(self.position_generator, OrbitGeneratorCW):
                raise TypeError(
                    "Position generator must be of type OrbitGeneratorCW when store_lvlh is True"
                )
            self.position_generator.set_origin(self.get_origin_satellite_state())
            self._set_origin_satellite()

    def set_origin_satellite_by_vector(
        self, prefix: str, states: np.ndarray[(6,), np.float64], color=""
    ):
        if self.store_lvlh:
            self.origin_satellite_name = prefix + "_0"
            r = states[:3] * u.km
            v = states[3:] * u.km / u.s
            self.origin_satellite = Orbit.from_vectors(Earth, r=r, v=v)

            if not isinstance(self.position_generator, OrbitGeneratorCW):
                raise TypeError(
                    "Position generator must be of type OrbitGeneratorCW when store_lvlh is True"
                )
            self.position_generator.set_origin(states)
            self._set_origin_satellite()

    def add_satellite_by_classical(
        self,
        states: np.ndarray[(6,), np.float64],
        accel: np.ndarray[(3,), np.float64],
        epoch: Time = None,
        prefix: str = "satellite",
        color: str = "blue",
    ) -> None:
        orbit = Orbit.from_classical(
            Earth,
            states[0] << u.km,
            states[1] << u.one,
            states[2] << u.deg,
            states[3] << u.deg,
            states[4] << u.deg,
            states[5] << u.deg,
        )
        states = np.concatenate((orbit.r.value, orbit.v.value))
        self.add_satellite_by_vector(
            states=states, accel=accel, prefix=prefix, color=color
        )

    def add_satellite_by_vector(
        self,
        states: np.ndarray[(6,), np.float64],
        accel: np.ndarray[(3,), np.float64],
        epoch: Time = None,
        prefix: str = "satellite",
        color: str = "blue",
    ) -> None:
        if self.store_lvlh:
            if self.origin_satellite is None or self.origin_satellite_name is None:
                raise ValueError(
                    "origin satellite or origin_satellite_name not initialized"
                )
            states = eci2lvlh(
                self.get_origin_satellite_state(),
                states,
            )
        self._add_satellite(state=states, accel=accel, prefix=prefix, color=color)

    @abc.abstractmethod
    def _add_satellite(
        self,
        state: np.ndarray[(6,), np.float64],
        accel: np.ndarray[(3,), np.float64],
        prefix: str,
        color: str = "blue",
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def set_satellite_enable(self, name: str, enable: bool) -> None:
        raise NotImplementedError

    def _limit_accel(self, name, accel):
        """
        限制卫星的加速度（基于球坐标系分量限制）

        约束条件:
        1. ||F|| <= F_max
        2. |d_alpha/dt| <= 2 deg/s (俯仰角速率)
        3. |d_beta/dt| <= 2 deg/s (方位角速率)
        """
        if name in self.limit_accel_satellites:
            self.limit_accel_satellites.remove(name)

        # 1. 处理最大模长限制 (图片约束: ||F|| <= ||F_max||)
        # 假设 self.max_accel 是最大允许加速度模长
        accel_norm = np.linalg.norm(accel)

        # 如果没有历史记录，初始化并返回
        if name not in self.history_accel:
            self.history_accel[name] = accel
            return accel

        old_accel = self.history_accel[name]
        old_norm = np.linalg.norm(old_accel)

        # 处理极小值情况，避免除零错误, 或者加速度小于 0.02 m/s^2
        if old_norm < 1e-10 or accel_norm < self.limit_min_accel:
            self.history_accel[name] = accel
            return accel

        # --- 坐标转换 (Cartesian -> Spherical) ---
        # 定义：z轴为极轴，xy平面为赤道面
        # alpha: 俯仰角 (-pi/2 到 pi/2), asin(z/r)
        # beta: 方位角 (-pi 到 pi), atan2(y, x)

        # 旧状态角度
        old_alpha = np.arcsin(np.clip(old_accel[2] / old_norm, -1.0, 1.0))
        old_beta = np.arctan2(old_accel[1], old_accel[0])

        # 新状态目标角度
        new_alpha = np.arcsin(np.clip(accel[2] / accel_norm, -1.0, 1.0))
        new_beta = np.arctan2(accel[1], accel[0])

        # --- 计算限制步长 ---
        limit_step_rad = self.update_interval * (
            self.limit_angle * np.pi / 180.0
        )  # 2度/秒 * dt

        # --- 2. 限制 Alpha (俯仰角) ---
        delta_alpha = new_alpha - old_alpha
        # 简单的截断限制
        if abs(delta_alpha) > limit_step_rad:
            new_alpha = old_alpha + np.sign(delta_alpha) * limit_step_rad

        # 确保 alpha 在 [-pi/2, pi/2] (虽然数学上asin结果已在此范围，但逻辑上需保证)
        new_alpha = np.clip(new_alpha, -np.pi / 2 + 1e-6, np.pi / 2 - 1e-6)

        # --- 3. 限制 Beta (方位角) ---
        delta_beta = new_beta - old_beta

        # 处理角度跨越 +/- pi 的情况 (例如从 179度 变到 -179度，实际只差2度)
        while delta_beta > np.pi:
            delta_beta -= 2 * np.pi
        while delta_beta < -np.pi:
            delta_beta += 2 * np.pi

        if abs(delta_beta) > limit_step_rad:
            new_beta = old_beta + np.sign(delta_beta) * limit_step_rad

        # 归一化 beta 到 [-pi, pi]
        if new_beta > np.pi:
            new_beta -= 2 * np.pi
        if new_beta < -np.pi:
            new_beta += 2 * np.pi

        # --- 坐标还原 (Spherical -> Cartesian) ---
        # 使用限制后的角度和新的模长(或者限制后的模长)重建向量
        # x = r * cos(alpha) * cos(beta)
        # y = r * cos(alpha) * sin(beta)
        # z = r * sin(alpha)

        constrained_accel = np.array(
            [
                accel_norm * np.cos(new_alpha) * np.cos(new_beta),
                accel_norm * np.cos(new_alpha) * np.sin(new_beta),
                accel_norm * np.sin(new_alpha),
            ]
        )
        self.limit_accel_satellites.add(name)
        self.history_accel[name] = constrained_accel
        return constrained_accel
