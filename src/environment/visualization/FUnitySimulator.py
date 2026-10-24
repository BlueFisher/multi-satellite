import json
import time
from typing import Dict, List, Any

import numpy as np
from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit

from ..dynamics.IGenerator import IGenerator
from ..utils import lvlh2eci, eci2lvlh  # 导入LVLH转ECI接口
from .SimulatorBase import SimulatorBase
from .network.Server import ZMQServer


class FUnitySimulator(SimulatorBase):
    """与 Unity 进行同步的卫星动力学模拟器。

    此类通过 ZeroMQ 与 Unity 建立通信，用于在 Python 侧管理卫星状态、
    同步坐标系、处理任务事件，并在需要时将 LVLH 与 ECI 坐标互转。

    该类继承自 SimulatorBase，并扩展了与 Unity 的交互功能，包括：
    - 接收 Unity 请求（启动、配置、角色、任务、退出等）
    - 存储与更新卫星状态
    - 将本地计算的状态实时发送给 Unity
    - 动态同步 Unity 场景中的卫星增删改

    Attributes:
        position_generator: 用于推进动力学状态的状态生成器。
        update_interval: 状态更新时间步长。
        render: 是否启用可视化。
        store_lvlh: 是否以 LVLH 坐标存储状态。
        satellite: 存储所有卫星的状态字典。
        group: 按前缀分组的卫星数量统计。
        keep_position: 指定保持位置不更新的卫星。
        interceptor_name: 拦截者名称列表。
        attacker_name: 攻击者名称列表。
        sync_interval: Unity 请求的同步周期。
        server: ZeroMQ 通信服务器。
    """

    def __init__(
        self,
        position_generator: IGenerator,
        update_interval: float = 10,
        render: bool = True,
        store_lvlh: bool = True,
    ):
        # 传递所有父类参数，特别是store_lvlh
        super().__init__(position_generator, update_interval, render, store_lvlh)

        self._on_sync_data = None
        self._on_server_start = None
        self._on_req_sync_time_change = None
        self._on_task_run = None
        self._on_task_quit = None
        self._on_connect = None
        self._on_task_stop = None
        self._on_sync_interval_change = None
        self.position_generator = position_generator
        self.update_interval = update_interval
        self.satellite: Dict[str, Any] = {}  # 存储状态（LVLH或ECI，取决于store_lvlh）
        self.group: Dict[str, int] = {}

        self.keep_position: List[str] = []

        self.interceptor_name = []
        self.attacker_name = []

        self._unity_init = False
        self.sync_interval = 1
        self.server = ZMQServer()

    def group_names(self) -> List[str]:
        """返回所有卫星分组名称。

        Returns:
            List[str]: 所有分组前缀名称。

        Example:
            >>> sim.group_names()
            ['sat', 'target']
        """
        return list(self.group.keys())

    def set_keep_position(self, positions: List[str]):
        """设置在更新循环中保持位置不变的卫星名称列表。

        Args:
            positions (List[str]): 需要保持位置不更新的卫星名称。

        Example:
            >>> sim.set_keep_position(['sat_0', 'sat_3'])
        """
        self.keep_position = positions

    def set_satellite_name(self, interceptor_name: list[str], attacker_name: list[str]):
        """设置拦截者与攻击者的名称列表。

        Args:
            interceptor_name (List[str]): 拦截者名称。
            attacker_name (List[str]): 攻击者名称。

        Example:
            >>> sim.set_satellite_name(['inter_0'], ['attacker_1'])
        """
        self.interceptor_name = interceptor_name
        self.attacker_name = attacker_name

    def set_sync_interval(self, sync_interval: float):
        """设置与 Unity 同步的时间间隔。

        Args:
            sync_interval (float): 同步周期（秒）。

        Example:
            >>> sim.set_sync_interval(1.0)
        """
        self.sync_interval = sync_interval

    def get_satellite(self, name: str) -> np.ndarray:
        """获取指定卫星的状态数组。

        若开启 store_lvlh 且名称为原点卫星，则返回零状态。

        Args:
            name (str): 卫星名称。

        Returns:
            np.ndarray: 卫星的状态向量。

        Raises:
            KeyError: 当卫星不存在时。

        Example:
            >>> state = sim.get_satellite('sat_0')
        """
        if name not in self.satellite:
            raise KeyError(f"Unknown satellite {name}")
        if self.store_lvlh and name == self.origin_satellite_name:
            return np.zeros((6,))
        return self.satellite[name]["state"].copy()

    def start(self):
        """启动 ZeroMQ 服务器并等待 Unity 连接。

        该方法会注册所有 Unity 请求的回调处理器，然后进入等待 Unity 初始化的循环。

        Example:
            >>> sim.start()
        """
        self.server.register_request_handler("ReqConfig", self._handler_req_config)
        self.server.register_request_handler("ReqSyncTime", self._handler_req_sync_time)
        self.server.register_request_handler("ReqStart", self._handler_req_start)
        self.server.register_request_handler(
            "ReqStartTask", self._handler_req_start_task
        )
        self.server.register_request_handler("ReqRole", self._hander_req_role)
        self.server.register_request_handler("ReqStopTask", self._handler_req_stop_task)
        self.server.register_request_handler("ReqQuit", self._handler_req_quit)
        self.server.register_request_handler(
            "ReqSyncTimeChange", self._handler_req_sync_time_change
        )
        self.server.register_request_handler(
            "ReqLightFactor", self._handler_req_light_factor
        )
        self.server.register_request_handler(
            "ReqIntentionFactor", self._handler_req_satellites_intention
        )

        self.server.start()

        if self._on_server_start is not None:
            self._on_server_start()

        while not self._unity_init:
            time.sleep(1)

    def on_server_start(self, func):
        """设置服务启动后的回调函数

        Args:
            func: 要调用的回调函数。

        Example:
            >>>  sim.on_server_start(lambda msg: print("Server Start"))
        """
        assert callable(func)
        self._on_server_start = func

    def _handler_req_config(self, message):
        """处理 Unity 发送的配置请求。

        Args:
            message: Unity 发来的请求数据。

        Returns:
            str: 返回初始化配置 JSON。

        Example:
            >>> sim._handler_req_config('{}')
        """

        if self._on_connect is not None:
            self._on_connect()

        return self._generator_init_message()

    def _hander_req_role(self, message):
        """处理 Unity 请求角色信息。

        Args:
            message: Unity 传入的消息。

        Returns:
            str: JSON 格式的角色信息。

        Example:
            >>> sim._hander_req_role('{}')
        """
        return json.dumps(
            {
                "hvt": self.origin_satellite_name,
                "interceptor": self.interceptor_name,
                "attacker": self.attacker_name,
            }
        )

    def _handler_req_sync_time(self, message):
        """处理 Unity 请求同步时间的消息。

        Args:
            message: Unity 请求。

        Returns:
            str: JSON 格式同步时间配置。

        Example:
            >>> sim._handler_req_sync_time('{}')
        """
        return json.dumps(
            {
                "sync_interval": self.sync_interval,
                "update_interval": self.update_interval,
            }
        )

    def _handler_req_start(self, message: str) -> str:
        """处理 Unity 的启动同步请求。

        执行卫星增删改同步，并根据需要在 LVLH/ECI 之间转换。

        Args:
            message (str): Unity 传来的卫星状态 JSON。

        Returns:
            str: 同步结果 JSON。

        Example:
            >>> sim._handler_req_start('[{"name":"sat_0"}]')
        """
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
                        "enable": True,
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
                    self.satellite[name]["state"] = stored_state
                    self.satellite[name]["accel"] = accel
                else:
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
                        "enable": True,
                    }

            self._unity_init = True

            if self._on_sync_data is not None:
                self._on_sync_data()

            return json.dumps(
                {"status": "ok", "message": "Synchronization successful."}
            )
        except Exception as e:
            print(f"Error processing ReqStart: {e}")
            return json.dumps({"status": "error", "message": str(e)})

    def _handler_req_stop_task(self, message):
        """处理 Unity 请求停止当前任务的消息。

        Args:
            message: Unity 消息。

        Returns:
            str: JSON 结果。

        Example:
            >>> sim._handler_req_stop_task('{}')
        """

        print("Request stop task")
        if self._on_task_stop is not None:
            self._on_task_stop()
        return '{"status": "ok"}'

    def _handler_req_quit(self, message):
        """处理 Unity 退出请求。

        Args:
            message: Unity 退出消息。

        Returns:
            str: JSON 响应。

        Example:
            >>> sim._handler_req_quit('{}')
        """
        print("Received quit request")
        self._unity_init = False

        if self._on_task_quit is not None:
            self._on_task_quit()

        return '{"status": "ok"}'

    def _handler_req_start_task(self, message):
        """处理 Unity 请求启动任务的消息。

        Args:
            message: Unity 消息。

        Returns:
            str: JSON 结果。

        Example:
            >>> sim._handler_req_start_task('{}')
        """
        if self._on_task_run is not None:
            self._on_task_run(message)
        return '{"status": "ok"}'

    def _handler_req_sync_time_change(self, message):
        """处理 Unity 请求同步时间间隔变化的消息。

        Args:
            message: Unity 消息。

        Returns:
            str: JSON 结果。

        Example:
            >>> sim._handler_req_sync_time_change('{"sync_interval": 10.0}')
        """
        if self._on_task_run is not None:
            self._on_sync_interval_change(message)
        return '{"status": "ok"}'

    def _handler_req_light_factor(self, message):
        """处理 Unity 请求光照变化的消息
        Args:
            message : Unity 消息

        returns:
            str: JSON 结果

        """
        try:
            self.light_factor = float(message)
        except Exception as e:
            print("Prase message Error, {}".format(e))

        return '{"status": "ok"}'

    def on_req_sync_time_change(self, func):
        """注册请求时间同步回调函数。

        Args:
            func: 任务运行回调函数。

        Example:
            >>> sim.on_req_sync_time_change(lambda msg: print(msg))
        """
        assert callable(func)
        self._on_req_sync_time_change = func

    def on_task_run(self, func):
        """注册任务运行回调函数。

        Args:
            func: 任务运行回调函数。

        Example:
            >>> sim.on_task_run(lambda msg: print(msg))
        """
        assert callable(func)
        self._on_task_run = func

    def on_task_quit(self, func):
        """注册任务退出回调函数。

        Args:
            func: 任务退出回调函数。

        Example:
            >>> sim.on_task_quit(lambda: print("Task quit"))
        """
        assert callable(func)
        self._on_task_quit = func

    def on_connect(self, func):
        """注册连接回调函数。

        Args:
            func: 连接回调函数。

        Example:
            >>> sim.on_sync_data(lambda: print("Connected"))
        """
        assert callable(func)
        self._on_connect = func

    def on_task_stop(self, func):
        """注册任务停止回调函数。

        Args:
            func: 任务停止回调函数。

        Example:
            >>> sim.on_task_stop(lambda: print("Task stop"))
        """
        assert callable(func)
        self._on_task_stop = func

    def on_sync_interval_change(self, func):
        """注册同步时间间隔变化回调函数。

        Args:
            func: 同步时间间隔变化回调函数。

        Example:
            >>> sim.on_sync_interval_change(lambda msg: print(msg))
        """
        assert callable(func)
        self._on_sync_interval_change = func

    def on_sync_data(self, func):
        """注册同步数据的回调函数。

        Args:
            func: 同步时间间隔变化回调函数。

        Example:
            >>> sim.on_sync_interval_change(lambda msg: print(msg))
        """
        assert callable(func)
        self._on_sync_data = func

    def clear(self):
        """清空当前模拟器中的所有卫星与分组数据。

        Example:
            >>> sim.clear()
        """

        self.group = {}
        self.satellite = {}
        self.origin_satellite = None  # 清除原点卫星

    def stop(self):
        """停止当前模拟器中的任务。

        Example:
            >>> sim.stop()
        """
        self.server.stop()

    def update(self) -> None:
        """执行一次状态更新循环。

        若 Unity 尚未建立初始化，则不会执行任何操作。

        Example:
            >>> sim.update()
        """
        if not self._unity_init:
            return
        self._update_satellite()

    def _update_satellite(self):
        """更新所有卫星的状态并将 ECI 格式发送给 Unity。

        Example:
            >>> sim._update_satellite()
        """
        unity_state = {}  # 存储要发送给Unity的状态

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
                satellite_enable = satellite_data.get("enable", True)

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

                # 构建发送给Unity的状态数据
                state_data = np.concatenate(
                    [position, velocity, satellite_data["accel"], updated_state]
                ).tolist()
                unity_state[satellite_id] = (
                    state_data + orbital_elements + [satellite_enable]
                )

        # 发送所有卫星状态到Unity，包括原点卫星
        self.server.publish_game_state("StateChange", unity_state)

    def _generator_init_message(self) -> str:
        """生成发送给 Unity 的初始化消息（全部为 ECI 坐标）。

        Returns:
            str: JSON 数据。

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
            lr = stored_state[:3]
            lv = stored_state[3:6]
            temp_orbit = Orbit.from_vectors(Earth, r * u.km, v * u.km / u.s)
            elem = [e.value for e in temp_orbit.classical()]

            data_dict = {
                "name": name,
                "id": team_id,
                "element": elem,
                "r": r.tolist(),
                "v": v.tolist(),
                "accel": item["accel"].tolist(),
                "lr": lr.tolist(),
                "lv": lv.tolist(),
            }
            data.append(data_dict)
        return json.dumps(data)

    def agent_names(self) -> List[str]:
        """返回当前所有卫星名称列表。

        Returns:
            List[str]: 卫星名称。

        Example:
            >>> names = sim.agent_names()
        """
        return list(self.satellite.keys())

    def render_frame(self):

        pass  # Unity端负责渲染

    def _add_accel(self, name: str, accel: np.ndarray[(3,), float]) -> None:
        """为指定卫星设置加速度。

        Args:
            name (str): 卫星名称。
            accel (np.ndarray): 加速度向量。

        Example:
            >>> sim._add_accel('sat_0', np.zeros(3))
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
        """添加一个新卫星（内部使用，不对外暴露）。

        Args:
            state (np.ndarray): 初始状态（LVLH 或 ECI）。
            accel (np.ndarray): 初始加速度。
            prefix (str): 分组前缀。
            color (str): 颜色（未使用）。

        Example:
            >>> sim._add_satellite(np.zeros(6), np.zeros(3), 'sat')
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
            "enable": True,
        }

    def _set_origin_satellite(self, color="yellow"):
        """初始化或更新原点卫星的存储状态（必须为 ECI）。

        Args:
            color (str): 原点卫星颜色。

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
        """启用或禁用某个卫星。

        Args:
            name (str): 卫星名称。
            enable (bool): 是否启用。

        Example:
            >>> sim.set_satellite_enable('sat_0', False)
        """

        self.satellite[name]["enable"] = enable

    def get_formatted_state_data(self) -> List[Dict]:
        """返回仅含位置与速度的纯净 JSON 格式卫星状态数据。

        Returns:
            List[Dict]: 每个卫星的 r 与 v 信息。

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

        for name, item in self.satellite.items():
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
                    "element": elem,
                    "r": eci_state[:3].tolist(),
                    "v": eci_state[3:6].tolist(),
                    # "accel": item["accel"].tolist(),
                    # "enable": item.get("enable", False),
                }
            )

        return data

    def _handler_req_satellites_intention(self, message):
        """处理Unity 意图识别的信息(频率最好放低一点，比python传输给unity的频率要慢)
        Args:
            message:Unity 卫星意图识别信息

        returns:
            strs: JSON结果

        """
        try:
            self.intention_message = json.loads(message)

        except Exception as e:
            print(f"Intention Message Error,{e}")
        return '{"status":"ok"}'
