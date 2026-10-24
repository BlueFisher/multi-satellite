from enum import Enum, auto

from ..dynamics.IGenerator import IGenerator
from ..visualization.SimulatorBase import SimulatorBase


class SimulatorType(Enum):
    Local = auto()
    Unity = auto()
    FUnity = auto()


class SimulatorFactory:
    @staticmethod
    def get_simulator(
        simulator_type: SimulatorType, position_generator: IGenerator
    ) -> SimulatorBase:
        # 根据类型返回不同的模拟器实例，主程序只调用接口
        if simulator_type.value == SimulatorType.Local.value:  # 具体实现放另一个模块
            from .LocalSimulator import LocalSimulator

            return LocalSimulator(position_generator)
        elif simulator_type.value == SimulatorType.Unity.value:
            from .UnitySimulator import UnitySimulator

            return UnitySimulator(position_generator)
        elif simulator_type.value == SimulatorType.FUnity.value:
            from .FUnitySimulator import FUnitySimulator

            return FUnitySimulator(position_generator)
        else:
            raise ValueError(f"Unknown simulator type {simulator_type}")
