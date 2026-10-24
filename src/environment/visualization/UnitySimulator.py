import json
import time
from typing import Dict, List, Any

import numpy as np
from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit

from .SimulatorBase import SimulatorBase
from .network.Server import ZMQServer
from .utils import lvlh2eci, eci2lvlh  # 导入LVLH转ECI接口
from ..dynamics.IGenerator import IGenerator


class UnitySimulator(SimulatorBase):
    """Unity 交互式轨道仿真器。

    该类通过 ZMQ 与 Unity 建立通信链路，并负责：
    1. 初始化并同步卫星状态（ECI 或 LVLH 坐标）。
    2. 接收 Unity 发来的卫星轨道要素、加速度等数据并更新本地状态。
    3. 调用轨道生成器推进卫星状态并实时发布给 Unity。
    4. 管理原点卫星（作为 LVLH 参考）以及普通卫星的坐标更新与转换。

    Attributes:
        position_generator (IGenerator): 卫星状态推进器，由外部提供。
        update_interval (float): 每次状态更新的时间间隔（秒）。
        satellite (Dict[str, Any]): 存储所有卫星的当前状态（LVLH 或 ECI）。
        group (Dict[str, int]): 保存卫星名称前缀分类与其数量。
        _unity_init (bool): Unity 端是否完成初始化。
        store_lvlh (bool): 是否在本地存储 LVLH 坐标。
        server (ZMQServer): 与 Unity 交互的 ZMQ 服务器。
    """

    def __init__(
        self,
        position_generator: IGenerator,
        update_interval: float = 10,
        render: bool = True,
        store_lvlh: bool = False,
    ):
        # 传递所有父类参数，特别是store_lvlh
        super().__init__(position_generator, update_interval, render, store_lvlh)

        self.position_generator = position_generator
        self.update_interval = update_interval
        self.satellite: Dict[str, Any] = {}  # 存储状态（LVLH或ECI，取决于store_lvlh）
        self.group: Dict[str, int] = {}
        self._unity_init = False

        self.server = ZMQServer()

    def group_names(self) -> List[str]:
        """返回当前已注册的卫星组名称列表。

        Returns:
            List[str]: 卫星名称前缀（组名）的列表。

        Example:
            >>> sim.group_names()
            ['teamA', 'teamB']
        """
        return list(self.group.keys())

    def get_satellite(self, name: str) -> np.ndarray:
        """获取指定卫星的当前状态向量。

        状态的返回格式取决于是否启用了 LVLH 存储：
        - 若 store_lvlh=True 且该卫星是原点卫星，则返回零向量。
        - 否则返回本地存储的 6 维状态（位置+速度）。

        Args:
            name (str): 卫星名称。

        Returns:
            np.ndarray: 6 维状态向量。

        Raises:
            KeyError: 当指定卫星不存在时。

        Example:
            >>> state = sim.get_satellite("red_0")
            >>> print(state.shape)
            (6,)
        """

        if name not in self.satellite:
            raise KeyError(f"Unknown satellite {name}")
        if self.store_lvlh and name == self.origin_satellite_name:
            return np.zeros((6,))
        return self.satellite[name]["state"].copy()

    def start(self):
        """启动仿真器，与 Unity 建立通信并等待 Unity 完成初始化。

        主要用于注册 Unity 请求处理器、启动 ZMQ 服务器，并等待 Unity 发送启动消息。

        Example:
            >>> sim.start()
            Waiting for unity start...
        """

        self.server.register_request_handler("ReqConfig", self._handler_req_config)
        self.server.register_request_handler("ReqStart", self._handler_req_start)
        self.server.register_request_handler("ReqQuit", self._handler_req_quit)
        self.server.start()
        print("Waiting for unity start...")
        while not self._unity_init:
            time.sleep(1)

    def _handler_req_config(self, message):
        """处理 Unity 的“ReqConfig”请求。

        返回轨道生成器的初始化信息，以便 Unity 正确初始化场景。

        Args:
            message: Unity 发送的消息内容（未使用）。

        Returns:
            str: JSON 格式的初始化回复。

        Example:
            >>> reply = sim._handler_req_config("")
            >>> print(reply)
        """

        print("Unity connected, Waiting for unity start...")
        return self._generator_init_message()

    def _handler_req_start(self, message: str) -> str:
        """处理 Unity 发来的“ReqStart”请求并完成卫星状态同步。

        该方法负责：
        1. 接收 Unity 传来的所有卫星状态与轨道要素。
        2. 将其转换为本地存储格式（ECI 或 LVLH）。
        3. 动态添加、更新或删除卫星。
        4. 初始化原点卫星轨道（若使用 LVLH）。

        Args:
            message (str): Unity 发送的卫星列表 JSON 字符串。

        Returns:
            str: JSON 格式的状态同步结果。

        Example:
            >>> sim._handler_req_start('[{"name":"sat_0","id":0,"element":[...]}]')
        """

        print("Unity request start. Synchronizing Python state with Unity...")

        try:
            # 1. 解析Unity回传的JSON数据
            unity_sats_list = json.loads(message)
            unity_sats_map = {
                sat_data["name"]: sat_data for sat_data in unity_sats_list
            }

            # 2. 动态同步过程
            # 步骤A: 处理删除
            python_sat_names = set(self.satellite.keys())
            unity_sat_names = set(unity_sats_map.keys())
            deleted_names = python_sat_names - unity_sat_names

            if deleted_names:
                print(f"Deleting satellites not found in Unity: {deleted_names}")
                for name in deleted_names:
                    del self.satellite[name]

            # 步骤B: 处理增加与修改（先处理原点卫星，再处理普通卫星）
            origin_name = self.origin_satellite_name if self.store_lvlh else None

            # 分离原点卫星和普通卫星数据
            origin_sat_data = None
            other_sats = []
            for name, sat_data in unity_sats_map.items():
                if self.store_lvlh and name == origin_name:
                    origin_sat_data = (name, sat_data)
                else:
                    other_sats.append((name, sat_data))

            # 处理原点卫星（始终存储ECI状态）
            if origin_sat_data:
                name, sat_data = origin_sat_data
                elements = sat_data["element"]
                accel = np.asarray(sat_data["accel"])

                # 创建原点卫星轨道（ECI）
                temp_orbit = Orbit.from_classical(
                    Earth,
                    elements[0] * u.km,
                    elements[1] * u.one,
                    elements[2] * u.deg,
                    elements[3] * u.deg,
                    elements[4] * u.deg,
                    elements[5] * u.deg,
                )
                eci_state = np.concatenate([temp_orbit.r.value, temp_orbit.v.value])

                # 更新或添加原点卫星
                if name in self.satellite:
                    print(f"Updating origin satellite: {name}")
                    self.satellite[name]["state"] = eci_state
                    self.satellite[name]["accel"] = accel
                else:
                    print(f"Adding origin satellite from Unity: {name}")
                    prefix = name.rsplit("_", 1)[0]
                    self.group[prefix] = self.group.get(prefix, 0) + 1
                    self.satellite[name] = {
                        "prefix": prefix,
                        "name": name,
                        "index": sat_data["id"],
                        "state": eci_state,  # 原点始终存储ECI
                        "accel": accel,
                        "mode": sat_data.get("mode", "rule"),
                        "mass": sat_data.get("mass", 500.0),
                    }
                self.origin_satellite = temp_orbit  # 更新原点轨道

            # 处理普通卫星（根据store_lvlh决定存储格式）
            for name, sat_data in other_sats:
                elements = sat_data["element"]
                accel = np.asarray(sat_data["accel"])

                # 生成ECI状态
                temp_orbit = Orbit.from_classical(
                    Earth,
                    elements[0] * u.km,
                    elements[1] * u.one,
                    elements[2] * u.deg,
                    elements[3] * u.deg,
                    elements[4] * u.deg,
                    elements[5] * u.deg,
                )
                eci_state = np.concatenate([temp_orbit.r.value, temp_orbit.v.value])

                # 转换为LVLH存储（如果需要）
                if self.store_lvlh and self.origin_satellite is not None:
                    origin_eci = np.concatenate(
                        [self.origin_satellite.r.value, self.origin_satellite.v.value]
                    )
                    stored_state = eci2lvlh(eci_state, origin_eci)
                else:
                    stored_state = eci_state

                # 更新或添加普通卫星
                if name in self.satellite:
                    print(f"Updating satellite: {name}")
                    self.satellite[name]["state"] = stored_state
                    self.satellite[name]["accel"] = accel
                else:
                    print(f"Adding new satellite from Unity: {name}")
                    prefix = name.rsplit("_", 1)[0]
                    self.group[prefix] = self.group.get(prefix, 0) + 1
                    self.satellite[name] = {
                        "prefix": prefix,
                        "name": name,
                        "index": sat_data["id"],
                        "state": stored_state,  # 存储LVLH或ECI
                        "accel": accel,
                        "mode": sat_data.get("mode", "rule"),
                        "mass": sat_data.get("mass", 500.0),
                    }

            print(
                f"Synchronization complete. Current satellites: {list(self.satellite.keys())}"
            )
            self._unity_init = True
            return json.dumps(
                {"status": "ok", "message": "Synchronization successful."}
            )

        except Exception as e:
            print(f"Error processing ReqStart: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    def _handler_req_quit(self, message):
        """处理 Unity 的退出请求。

        Unity 通知退出后会重置部分标志，但不会销毁现有卫星。

        Args:
            message (str): Unity 消息内容。

        Returns:
            str: JSON 格式的确认信息。

        Example:
            >>> sim._handler_req_quit("")
        """

        print("Received quit request")
        self._unity_init = False
        return '{"status": "ok"}'

    def clear(self):
        """清空仿真器内部的所有卫星与分组信息。

        用于重新开始仿真。不会停止 ZMQ 服务。

        Example:
            >>> sim.clear()
        """

        self.group = {}
        self.satellite = {}
        # self._unity_init = False
        self.origin_satellite = None  # 清除原点卫星

    def stop(self):
        """停止 ZMQ 服务器。

        用于关闭与 Unity 的通信。

        Example:
            >>> sim.stop()
        """

        self.server.stop()

    def update(self) -> None:
        """推进一次卫星状态更新流程。

        若 Unity 尚未初始化则不执行任何操作。

        Example:
            >>> sim.update()
        """

        if not self._unity_init:
            return
        self._update_satellite()

    def _update_satellite(self):
        """执行一次卫星状态推进并将最新 ECI 状态发送给 Unity。

        核心功能包括：
        1. 推进原点卫星轨道（若使用 LVLH）。
        2. 使用生成器推进其他卫星状态。
        3. LVLH 与 ECI 之间的转换。
        4. 生成要发送给 Unity 的状态包。

        Example:
            >>> sim._update_satellite()
        """

        unity_state = {}  # 存储要发送给Unity的状态
        self.keep_position = []
        # 1. 更新原点卫星轨道（如果启用LVLH坐标系）
        origin_eci_state = None
        if self.store_lvlh and self.origin_satellite is not None:
            # 传播原点卫星轨道
            self.origin_satellite = self.origin_satellite.propagate(
                self.update_interval * u.s
            )

            # 更新本地存储的原点卫星ECI状态
            if self.origin_satellite_name in self.satellite:
                origin_position = self.origin_satellite.r.value
                origin_velocity = self.origin_satellite.v.value
                origin_eci_state = np.concatenate([origin_position, origin_velocity])
                self.satellite[self.origin_satellite_name]["state"] = origin_eci_state

        # 2. 处理所有卫星，更新状态并转换为ECI
        for group_name, satellite_count in self.group.items():
            for index in range(satellite_count):
                satellite_id = f"{group_name}_{index}"
                satellite_data = self.satellite[satellite_id]

                # 判断是否为原点卫星
                is_origin_satellite = satellite_id == self.origin_satellite_name

                # 跳过原点卫星或需要保持位置的卫星的状态更新，但仍会处理发送逻辑
                if not is_origin_satellite and satellite_id not in self.keep_position:
                    # 在存储坐标系下更新状态（LVLH或ECI）
                    updated_state = self.position_generator(
                        state=satellite_data["state"],
                        accel=satellite_data["accel"],
                        step=self.update_interval,
                    )
                    satellite_data["state"] = updated_state
                else:
                    # 不更新状态，直接使用当前状态
                    updated_state = satellite_data["state"]

                # 转换为ECI坐标系（用于发送给Unity）
                if (
                    self.store_lvlh
                    and not is_origin_satellite
                    and self.origin_satellite is not None
                ):
                    # LVLH转ECI
                    eci_state = lvlh2eci(updated_state, origin_eci_state)
                else:
                    # 原点卫星或ECI模式直接使用当前状态
                    eci_state = updated_state

                # 提取位置和速度，生成轨道要素
                position = eci_state[:3]
                velocity = eci_state[3:6]
                orbit = Orbit.from_vectors(
                    Earth, position * u.km, velocity * u.km / u.s
                )
                orbital_elements = [element.value for element in orbit.classical()]

                # 构建发送给Unity的状态数据（遵循原有协议）
                state_data = np.concatenate(
                    [position, velocity, satellite_data["accel"]]
                ).tolist()
                unity_state[satellite_id] = state_data + orbital_elements

        # 发送所有卫星状态到Unity，包括原点卫星
        self.server.publish_game_state("StateChange", unity_state)

    def _generator_init_message(self) -> str:
        """生成发送给 Unity 的初始化信息。

        会将当前所有卫星转换为 ECI 状态并附上轨道六根数。

        Returns:
            str: JSON 格式的卫星初始化数据。

        Example:
            >>> msg = sim._generator_init_message()
        """

        data = []
        for name, item in self.satellite.items():
            team_id = item["index"]
            stored_state = item["state"]

            # 转换为ECI
            if (
                self.store_lvlh
                and name != self.origin_satellite_name
                and self.origin_satellite is not None
            ):
                origin_eci = np.concatenate(
                    [self.origin_satellite.r.value, self.origin_satellite.v.value]
                )
                eci_state = lvlh2eci(stored_state, origin_eci)
            else:
                eci_state = stored_state

            # 提取位置、速度和轨道要素
            r = eci_state[:3]
            v = eci_state[3:6]
            temp_orbit = Orbit.from_vectors(Earth, r * u.km, v * u.km / u.s)
            elem = [e.value for e in temp_orbit.classical()]

            data_dict = {
                "name": name,
                "id": team_id,
                "element": elem,
                "r": r.tolist(),
                "v": v.tolist(),
                "accel": item["accel"].tolist(),
            }
            data.append(data_dict)
        return json.dumps(data)

    def agent_names(self) -> List[str]:
        """返回当前全部卫星名称列表。

        Returns:
            List[str]: 所有卫星名称。

        Example:
            >>> sim.agent_names()
            ['blue_0', 'blue_1', 'yellow_0']
        """

        return list(self.satellite.keys())

    def render_frame(self):
        pass  # Unity端负责渲染

    def _add_accel(self, name: str, accel: np.ndarray[(3,), float]) -> None:
        """为指定卫星设置新的加速度向量。

        Args:
            name (str): 卫星名称。
            accel (np.ndarray): 3 维加速度向量。

        Raises:
            KeyError: 当指定卫星不存在时。

        Example:
            >>> sim._add_accel("blue_0", np.array([0,0,0.1]))
        """

        if name not in self.satellite:
            raise KeyError(f"Can't find satellite {name}")
        self.satellite[name]["accel"] = accel

    def _add_satellite(
        self,
        state: np.ndarray[(6,), np.float64],  # 已由父类转换为LVLH（如果需要）
        accel: np.ndarray[(3,), np.float64],
        prefix: str,
        color: str = "blue",
    ) -> None:
        """向仿真器中添加一个新卫星。

        通常由父类调用，用于创建新的 ECI/LVLH 初始化状态。

        Args:
            state (np.ndarray): 6 维状态向量。
            accel (np.ndarray): 3 维加速度。
            prefix (str): 卫星名称前缀。
            color (str): 卫星颜色（未使用）。

        Example:
            >>> sim._add_satellite(np.zeros(6), np.zeros(3), "team")
        """

        if prefix not in self.group:
            self.group[prefix] = 0

        index = self.group[prefix]
        self.group[prefix] += 1
        name = f"{prefix}_{index}"

        self.satellite[name] = {
            "prefix": prefix,
            "name": name,
            "index": index,
            "state": state,  # 存储LVLH或ECI
            "accel": accel,
        }

    def _set_origin_satellite(self, color="yellow"):
        """设置或更新原点卫星存储状态。

        若启用了 LVLH，则原点卫星始终存储为 ECI 坐标。

        Args:
            color (str): 原点卫星颜色（未使用）。

        Example:
            >>> sim._set_origin_satellite()
        """

        if self.store_lvlh and self.origin_satellite_name and self.origin_satellite:
            # 原点卫星存储为ECI状态
            origin_eci = np.concatenate(
                [self.origin_satellite.r.value, self.origin_satellite.v.value]
            )
            if self.origin_satellite_name not in self.satellite:
                prefix = self.origin_satellite_name.rsplit("_", 1)[0]
                self._add_satellite(state=origin_eci, accel=np.zeros(3), prefix=prefix)
            else:
                self.satellite[self.origin_satellite_name]["state"] = origin_eci

    def set_satellite_enable(self, name: str, enable: bool) -> None:
        pass
