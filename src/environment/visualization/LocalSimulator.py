from typing import Dict, List

import astropy.units as u
import numpy as np
from poliastro.bodies import Earth
from poliastro.twobody import Orbit

from .SimulatorBase import SimulatorBase
from .utils import lvlh2eci
from ..dynamics import IGenerator


class LocalSimulator(SimulatorBase):
    """局部坐标系卫星模拟器。

    该类实现了一个以原点卫星为基准的多卫星动力学模拟器，所有渲染均以 ECI 坐标进行。
    普通卫星的状态与轨迹独立存储，而原点卫星通过父类中的 Orbit 对象进行传播。

    此类主要职责包括：
    - 管理普通卫星与原点卫星的状态
    - 支持 LVLH 与 ECI 坐标混合存储
    - 管理可视化窗口中的卫星渲染、轨迹记录与中心卫星切换
    - 提供统一的模拟更新接口以及卫星加速度设置

    Attributes:
        max_trail_length (int): 每个卫星轨迹的最大长度。
        sampling_interval (int): 轨迹采样间隔。
        store_lvlh (bool): 是否存储为 LVLH 坐标。
        satellites (List[Dict]): 普通卫星的所有状态、轨迹与加速度信息。
        group_name_counters (Dict[str, int]): 每个卫星组的计数器。
        name2index (Dict[str, int]): 卫星名称到索引的映射。
        origin_visual_index (int): 原点卫星在可视化窗口中的索引。
        origin_trail_data (Dict): 原点卫星轨迹数据。
        windows: 可视化窗口对象。
    """

    def __init__(
        self,
        position_generator: IGenerator,
        update_interval: float = 10,
        max_trail_length: int = 1000,
        sampling_interval: int = 5,
        render: bool = True,
        store_lvlh: bool = False,
    ):
        super().__init__(
            position_generator,
            update_interval,
            render=render,
            store_lvlh=store_lvlh,
        )

        self.max_trail_length = max_trail_length
        self.sampling_interval = sampling_interval
        self.store_lvlh = store_lvlh

        # 普通卫星存储（与原点卫星完全独立）
        self.satellites: list[Dict] = []
        self.group_name_counters: Dict[str, int] = {}
        self.name2index: Dict[str, int] = {}

        # 原点卫星可视化相关属性（数据来自父类的Orbit变量）
        self.origin_visual_index = -1  # 原点在可视化窗口中的索引
        self.origin_trail_data = {"positions": [], "sampled_positions": []}

        self.windows = None

        if self.render:
            from .Window.LocalScaleWindow import LocalSatelliteAnimationWindow

            if not isinstance(self.windows, LocalSatelliteAnimationWindow):
                self.windows = LocalSatelliteAnimationWindow()

    def start(self) -> None:
        """启动可视化窗口。

        若启用了渲染功能，则此方法将打开交互式可视化窗口。

        Example:
            >>> sim = LocalSimulator(generator)
            >>> sim.start()
        """
        if self.render:
            self.windows.show()

    def stop(self) -> None:
        """关闭可视化窗口。

        用于结束可视化渲染并释放图形窗口资源。

        Example:
            >>> sim.stop()
        """

        if self.render:
            self.windows.close()

    def group_names(self) -> list[str]:
        """返回当前所有卫星所属的组名前缀列表。

        Returns:
            list[str]: 所有组名前缀。

        Example:
            >>> groups = sim.group_names()
        """

        return list(self.group_name_counters.keys())

    def render_frame(self) -> None:
        """渲染可视化窗口的单帧。

        只有 render=True 时才会执行。

        Example:
            >>> sim.render_frame()
        """

        if self.render:
            self.windows.render()

    def _set_origin_satellite(self, color="yellow") -> None:
        """初始化原点卫星的可视化元素。

        该方法会在窗口中为原点卫星分配渲染索引，并初始化其轨迹缓存。

        Args:
            color (str): 原点卫星的渲染颜色。

        Example:
            >>> sim._set_origin_satellite(color="yellow")
        """

        # 为原点卫星分配独立的可视化索引
        if self.render:
            self.origin_visual_index = self.windows.add_satellite(color)
            # 初始化轨迹数据
            self.origin_trail_data = {
                "positions": [],
                "sampled_positions": [],
                "index": -1,
            }
            self.satellites.append(self.origin_trail_data)

    def set_center_satellite(self, name: str) -> None:
        """设置可视化窗口的中心卫星。

        支持普通卫星与原点卫星的切换。

        Args:
            name (str): 目标卫星名称。

        Example:
            >>> sim.set_center_satellite("sat_0")
        """

        if self.render == False:
            return

        # 处理原点卫星中心化（使用父类定义的原点名称）
        if self.store_lvlh and name == self.origin_satellite_name:
            self.windows.set_center_satellite(self.origin_visual_index)
            return

        # 处理普通卫星中心化
        if name in self.name2index:
            self.windows.set_center_satellite(self.name2index[name])

    def agent_names(self) -> list[str]:
        """返回所有卫星名称。

        若启用了 LVLH 模式，则会包含原点卫星名称。

        Returns:
            list[str]: 所有卫星名称。

        Example:
            >>> names = sim.agent_names()
        """

        names = list(self.name2index.keys())
        if self.store_lvlh:
            # 添加父类定义的原点卫星名称
            names.append(self.origin_satellite_name)
        return names

    def get_satellite(self, name: str) -> np.ndarray:
        """获取指定名称卫星的状态向量。

        Args:
            name (str): 卫星名称。

        Returns:
            np.ndarray: 卫星状态向量。

        Raises:
            KeyError: 当名称不存在时。

        Example:
            >>> state = sim.get_satellite("sat_0")
        """
        # 原点卫星状态从父类Orbit变量获取
        if self.store_lvlh and name == self.origin_satellite_name:
            return np.zeros((6,))

        # 普通卫星状态
        if name not in self.name2index:
            raise KeyError(f"未知卫星: {name}")
        index = self.name2index[name]
        return self.satellites[index]["state"].copy()

    def clear(self):
        """清空所有普通卫星及其轨迹。

        原点卫星将被保留并重新初始化其渲染资源。

        Example:
            >>> sim.clear()
        """

        self.satellites.clear()
        self.group_name_counters.clear()
        self.name2index.clear()

        if self.render:
            self.windows.reset()

    def _add_satellite(
        self,
        state: np.ndarray,
        accel: np.ndarray,
        prefix: str,
        color: str = "blue",
    ) -> None:
        """创建并注册一个普通卫星。

        不会影响原点卫星，且按组名前缀进行编号。

        Args:
            state (np.ndarray): 初始状态向量。
            accel (np.ndarray): 初始加速度向量。
            prefix (str): 名称前缀。
            color (str): 绘制颜色。

        Example:
            >>> sim._add_satellite(state, accel, "sat")
        """

        if self.store_lvlh and prefix == self.origin_satellite_name:
            raise ValueError(
                f"前缀不能与原点卫星名称冲突: {self.origin_satellite_name}"
            )

        counter = self.group_name_counters.get(prefix, -1) + 1
        self.group_name_counters[prefix] = counter
        name = f"{prefix}_{counter}"

        if name in self.name2index:
            raise ValueError(f"卫星名称已存在: {name}")

        index = (
            self.windows.add_satellite(color) if self.render else len(self.satellites)
        )
        self.name2index[name] = index
        self.satellites.append(
            {
                "group": prefix,
                "name": name,
                "state": state,
                "accel": accel,
                "positions": [],
                "sampled_positions": [],
                "index": index,
            }
        )

    def _update_origin_satellite(self) -> None:
        """更新原点卫星状态并同步其可视化轨迹。

        通过父类 Orbit 进行传播，并记录渲染轨迹。

        Example:
            >>> sim._update_origin_satellite()
        """

        # 1. 传播父类中的Orbit变量（时间步进）
        self.origin_satellite = self.origin_satellite.propagate(
            self.update_interval * u.s
        )

        if not self.render:
            return

        # 2. 从Orbit获取ECI坐标状态
        pos = self.origin_satellite.r.value  # 位置（ECI）
        vel = self.origin_satellite.v.value  # 速度（ECI）
        origin_state = np.concatenate([pos, vel])

        # 3. 更新原点卫星轨迹（使用ECI坐标）
        self._update_satellite_trail(
            origin_state, self.origin_trail_data, self.origin_visual_index
        )

    def update(self) -> None:
        """执行一次完整的模拟更新步骤。

        包括：
        1. 更新原点卫星（若使用 LVLH）
        2. 更新普通卫星状态
        3. 同步可视化窗口并刷新轨迹

        Example:
            >>> sim.update()
        """

        # 优先更新原点卫星（基于父类Orbit）
        if self.store_lvlh:
            self._update_origin_satellite()

        # 更新普通卫星
        for sat in self.satellites:
            if sat["index"] == -1:
                continue

            # 更新卫星状态
            sat["state"] = self.position_generator(
                sat["state"], sat["accel"], step=self.update_interval
            )

            # 更新可视化（统一使用ECI坐标）
            if self.render:
                # 转换LVLH到ECI（如果需要）
                if self.store_lvlh:
                    # 从父类Orbit获取原点状态用于转换
                    origin_state = np.concatenate(
                        [self.origin_satellite.r.value, self.origin_satellite.v.value]
                    )
                    eci_state = lvlh2eci(sat["state"], origin_state)
                else:
                    eci_state = sat["state"]

                # 更新轨迹（ECI坐标）
                self._update_satellite_trail(eci_state, sat, sat["index"])

    def _update_satellite_trail(
        self, state: np.ndarray, trail_data: Dict, visual_index: int
    ) -> None:
        """通用轨迹管理方法。

        根据采样间隔更新卫星轨迹数据，并同步到可视化窗口。

        Args:
            state (np.ndarray): 卫星的 ECI 状态。
            trail_data (Dict): 轨迹存储字典。
            visual_index (int): 可视化窗口的目标索引。

        Example:
            >>> sim._update_satellite_trail(state, data, idx)
        """

        pos_scaled = state[:3] / 1e3
        norm = np.linalg.norm(pos_scaled)

        # 视觉偏移优化
        if norm > 0:
            pos_scaled += pos_scaled / norm * 15

        # 维护轨迹数据
        trail_data["positions"].append(pos_scaled.tolist())
        if len(trail_data["positions"]) > self.max_trail_length:
            trail_data["positions"] = trail_data["positions"][-self.max_trail_length :]

        # 采样并更新可视化
        if len(trail_data["positions"]) % self.sampling_interval == 0:
            trail_data["sampled_positions"] = trail_data["positions"][
                :: self.sampling_interval
            ]
            self.windows.update_satellite(
                visual_index, pos_scaled.tolist(), trail_data["sampled_positions"]
            )

    def _add_accel(self, name: str, accel: np.ndarray) -> None:
        """设置普通卫星的加速度。

        原点卫星的加速度由 Orbit 动力学管理，不允许设置。

        Args:
            name (str): 卫星名称。
            accel (np.ndarray): 加速度向量。

        Example:
            >>> sim._add_accel("sat_0", np.array([0, 0, 0]))
        """

        if self.store_lvlh and name == self.origin_satellite_name:
            return  # 原点卫星加速度由父类Orbit管理

        if name not in self.name2index:
            raise KeyError(f"未知卫星: {name}")

        index = self.name2index[name]
        self.satellites[index]["accel"] = accel

    def set_satellite_enable(self, name: str, enable: bool) -> None:
        pass

    def get_formatted_state_data(self) -> List[Dict]:
        """返回所有卫星的纯净状态数据（JSON 友好格式）。

        返回的数据均为 ECI 坐标，并可选地从 LVLH 转换而来。

        Returns:
            List[Dict]: 每个卫星的 r、v 信息。

        Example:
            >>> data = sim.get_formatted_state_data()
        """

        data = []

        # 获取原点卫星 ECI
        origin_eci = None
        if self.store_lvlh and self.origin_satellite_name and self.origin_satellite:
            origin_eci = np.concatenate(
                [self.origin_satellite.r.value, self.origin_satellite.v.value]
            )

        for name, index in self.name2index.items():

            item = self.satellites[index]

            stored_state = item["state"]

            # ECI 转换
            if (
                self.store_lvlh
                and name != self.origin_satellite_name
                and origin_eci is not None
            ):
                eci_state = lvlh2eci(stored_state, origin_eci)
            else:
                eci_state = stored_state

            # 计算根数
            temp_orbit = Orbit.from_vectors(
                Earth, eci_state[:3] * u.km, eci_state[3:6] * u.km / u.s
            )
            elem = [e.value for e in temp_orbit.classical()]

            data.append(
                {
                    "name": name,
                    # "id": item["index"],
                    # "element": elem,
                    "r": eci_state[:3].tolist(),
                    "v": eci_state[3:6].tolist(),
                    # "accel": item["accel"].tolist(),
                    # "enable": item.get("enable", False),
                }
            )

        return data
