import os

import msgpack_numpy
import torch

msgpack_numpy.patch()

import zerorpc
from typing import Dict, List

import argparse

from ..utils_func import get_alg_manager
from ..Logger.Logger import Logger


class InferenceServer:
    def __init__(self):
        self.alg = None
        self.logger = Logger()

    def _check_args(self, args: Dict):
        assert "device" in args
        assert "load_path" in args

        ret_args = args.copy()

        ret_args["device"] = "cuda" if torch.cuda.is_available() else "cpu"

        load_path: str = args["load_path"].replace("\\", "/")
        index = load_path.find("/PythonSatellite/")

        this_file_dir = os.path.dirname(__file__)
        utils_dir = os.path.dirname(this_file_dir)
        src_dir = os.path.dirname(utils_dir)
        pythonsatellite_dir = os.path.dirname(src_dir)
        root_dir = os.path.dirname(pythonsatellite_dir)

        load_path = root_dir + load_path[index:]
        ret_args["load_path"] = load_path
        return ret_args

    def load_algorithm(self, algorithms_args: Dict):
        if self.alg is None:
            assert "alg_name" in algorithms_args
            alg_name = algorithms_args["alg_name"]
            del algorithms_args["alg_name"]
            alg_instance = get_alg_manager(alg_name)

            assert "args" in algorithms_args

            algorithms_args["args"] = self._check_args(algorithms_args["args"])

            self.logger.info(
                f"开始加载算法: {alg_name}, 使用设备: {algorithms_args['args']['device']}",
            )

            args = argparse.Namespace(**algorithms_args["args"])
            algorithms_args["args"] = args

            self.alg = alg_instance(**algorithms_args)
            self.logger.info(
                f"算法 {alg_name} 加载成功，当前算法实例: {type(self.alg).__name__}"
            )

    def reset_algorithm(self):
        if self.alg is not None:
            alg_name = type(self.alg).__name__
            self.alg = None
            self.logger.info(f"算法 {alg_name} 已重置，环境已清理")

    def choose_action(self, agent_list, obs, eval, index: List[int]):

        log_agent = [agent_list[idx] for idx in index]
        self.logger.info(f"正在为智能体 {log_agent} 计算动作，评估模式: {eval}")
        actions = self.alg.choose_action(agent_list, obs, eval=eval)["action"]
        selected_actions = [actions[idx] for idx in index]
        self.logger.debug(f"动作计算结果: {selected_actions}")  # 使用debug级别避免刷屏
        return selected_actions


class RpcServer:

    def __init__(self, ip="127.0.0.1", port=5678):
        self.ip = ip
        self.port = port

    def start(self):
        rpc = zerorpc.Server(InferenceServer())
        rpc.bind(f"tcp://{self.ip}:{self.port}")
        rpc.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5678)

    args = parser.parse_args()

    logger = Logger()
    logger.info(f"Run server at {args.ip}:{args.port}")
    server = RpcServer(args.ip, args.port)
    server.start()
