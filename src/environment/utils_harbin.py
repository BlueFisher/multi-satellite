from typing import Dict

import numpy as np

from .dynamics.OrbitGeneratorCW import OrbitGeneratorCW
from .visualization import SimulatorBase


def create_scenario_from_harbin_config(simulator: SimulatorBase, config: Dict):
    # assert config.get("is_harbin") is True
    attacker_prefix = config["attacker_prefix"]
    interceptor_prefix = config["interceptor_prefix"]
    hvt_name: str = config["hvt_name"]
    hvt_prefix = hvt_name.split("_")[0]
    # 先处理HVT卫星，因为它需要被设置为原点卫星
    hvt_satellite = None
    other_satellites = []

    # 分离HVT和其他卫星
    for satellite in config["satellites"]:
        group = satellite["group"]
        if group == "HVT":
            hvt_satellite = satellite
        else:
            other_satellites.append(satellite)

    simulator.store_lvlh = True

    # 处理HVT卫星
    if hvt_satellite:
        group = hvt_satellite["group"]
        init_type = hvt_satellite["init_type"]
        init_data = np.array(hvt_satellite["init_data"], dtype=np.float64)
        # 根据初始化类型调用相应的方法添加HVT卫星并设置为原点
        if init_type == "classic":
            simulator.set_origin_satellite_by_classical(
                prefix=hvt_prefix, states=init_data
            )
        elif init_type == "vector":
            simulator.set_origin_satellite_by_vector(
                prefix=hvt_prefix, states=init_data
            )
        else:
            raise ValueError(f"Unknown init_type: {init_type}")
        assert isinstance(simulator.position_generator, OrbitGeneratorCW)
        simulator.position_generator.set_origin(simulator.get_origin_satellite_state())

    attacker_count = 0
    agent_count = 0

    # 处理其他卫星
    for satellite in other_satellites:
        group = satellite["group"]
        init_type = satellite["init_type"]
        init_data = np.array(satellite["init_data"], dtype=np.float64)

        # 设置默认加速度为零向量
        accel = np.zeros(3, dtype=np.float64)

        # 根据卫星组别设置颜色和命名
        if group == attacker_prefix:
            prefix = attacker_prefix  # 后缀由simulator处理
            color = "red"  # 攻击者用橙色表示
            attacker_count += 1
        elif group == interceptor_prefix:
            prefix = interceptor_prefix  # 后缀由simulator处理
            color = "blue"  # 拦截器用绿色表示
            agent_count += 1
        else:
            raise ValueError(f"Unknown group: {group}")

        # 根据初始化类型调用相应的方法
        if init_type == "classic":
            simulator.add_satellite_by_classical(
                states=init_data, accel=accel, prefix=prefix, color=color
            )
        elif init_type == "vector":
            simulator.add_satellite_by_vector(
                states=init_data, accel=accel, prefix=prefix, color=color
            )
        else:
            raise ValueError(f"Unknown init_type: {init_type}")
    return agent_count, attacker_count
