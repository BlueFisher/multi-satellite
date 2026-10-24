from argparse import Namespace
from typing import List, Dict, Union
import numpy as np

import msgpack_numpy

msgpack_numpy.patch()

import zerorpc
import threading

from ..Logger.Logger import Logger


def _group_endpoints(remote_config: Dict) -> List[Dict]:
    """
    将 remote_config 按 (ip, port) 分组，返回分组信息列表。

    remote_config 格式:
        {
            "agent_name": {
                "ip_address": str,
                "port": int,
                "index": int
            },
            ...
        }

    返回:
        [
            {
                "ip_address": str,
                "port": int,
                "indices": List[int],
                "agent_names": List[str]
            },
            ...
        ]
    """
    endpoint_map = {}
    for agent_name, cfg in remote_config.items():
        ip = cfg["ip_address"]
        port = cfg["port"]
        idx = cfg["index"]
        key = (ip, port)
        if key not in endpoint_map:
            endpoint_map[key] = {
                "ip_address": ip,
                "port": port,
                "indices": [],
                "agent_names": [],
            }
        endpoint_map[key]["indices"].append(idx)
        endpoint_map[key]["agent_names"].append(agent_name)
    return list(endpoint_map.values())


def _create_remote_connection(remote_config: Dict) -> List[Dict]:
    """
    根据分组信息建立 ZeroRpc 的远程连接，相同 (ip, port) 只创建一个连接。

    返回:
        [
            {
                "connection": zerorpc.Client(),
                "indices": List[int],
                "agent_names": List[str],
                "ip_address": str,
                "port": int
            },
            ...
        ]
    """
    groups = _group_endpoints(remote_config)
    connections = []
    for group in groups:
        client = zerorpc.Client()
        client.connect(f"tcp://{group['ip_address']}:{group['port']}")
        group["connection"] = client
        connections.append(group)
    return connections


class InferenceEngine:
    def __init__(
        self,
        algorithms_args: Dict,
        logger: Logger,
        remote_config: Union[Dict, None] = None,
        local_inference=False,
    ):
        """
        algorithms_args = {
            "alg_name": real_alg_name,
            "name_agents": args.name_agents,
            "obs_shapes": args.n_obs,
            "global_state_shape": global_state_shape,
            "action_shapes": args.n_actions,
            "args": args,
            "action_continuous": args.continuous,
        }
        """
        self.algorithms_args = algorithms_args
        self.logger = logger

        assert "name_agents" in algorithms_args
        remote_config = self._check_remote_config(remote_config, algorithms_args)

        # 准备算法参数（Namespace 转 dict）
        args = algorithms_args.get("args", {})
        if isinstance(args, Namespace):
            algorithms_args["args"] = vars(args)

        # 按 (ip, port) 分组
        groups = _group_endpoints(remote_config)

        # 如果需要本地推理，为所有本地地址启动独立的服务器线程
        if local_inference:
            for group in groups:
                ip = group["ip_address"]
                port = group["port"]
                # 仅对本地地址（127.0.0.1 或 localhost）启动本地服务器
                if ip not in ("127.0.0.1", "localhost"):
                    self.logger.error("本地推理时 ip 地址必须为 127.0.0.1 或 localhost")
                    raise KeyError()
                self._start_local_server(ip, port)

        # 创建所有客户端连接（包括本地和远程）
        self.connections = _create_remote_connection(remote_config)

        # 加载算法到每个连接的服务端
        for meta in self.connections:
            connection: zerorpc.Client = meta["connection"]
            ip = meta["ip_address"]
            port = meta["port"]
            connection.load_algorithm(self.algorithms_args)
            self.logger.info(
                f"已成功建立与服务器 {ip}:{port} 的连接，该服务器管理智能体列表：{meta['agent_names']}"
            )

    def _start_local_server(self, ip: str, port: int):
        """启动一个本地推理服务器线程"""
        threading.Thread(
            target=self._run_local_server, args=(ip, port), daemon=True
        ).start()
        self.logger.info(f"本地推理服务器已在 {ip}:{port} 上启动。")

    def _run_local_server(self, ip: str, port: int):
        from .InferenceServer import RpcServer

        server = RpcServer(ip, port)
        server.start()

    def _check_remote_config(self, remote_config: Dict, algorithms_args: Dict):
        assert isinstance(remote_config, Dict)
        for name in algorithms_args["name_agents"]:
            if name not in remote_config:
                self.logger.error(f"没有找到智能体 {name} 对应的配置信息")
                raise KeyError()

        ret_config = remote_config
        for name in list(ret_config.keys()):
            if name not in algorithms_args["name_agents"]:
                del ret_config[name]
        return ret_config

    def choose_action(
        self, agent_list: List[str], obs: List[List[np.ndarray]], eval: bool = True
    ) -> dict:
        """
        参数:
            agent_list: 智能体名称列表
            obs: 每个智能体的观察列表，格式为 [[obs_agent0], [obs_agent1], ...]
            eval: 是否为评估模式
        返回:
            {"action": np.ndarray}  shape: (num_agents, action_dim)
        """
        num_agents = len(agent_list)
        actions = [None] * num_agents

        for meta in self.connections:
            connection = meta["connection"]
            indices = meta["indices"]

            try:
                batch_actions = connection.choose_action(agent_list, obs, eval, indices)
            except Exception as e:
                self.logger.error(
                    f"远程过程调用失败，目标索引 {indices}，错误详情：{e}"
                )
                raise

            if isinstance(batch_actions, list):
                for idx, act in zip(indices, batch_actions):
                    actions[idx] = act
            elif isinstance(batch_actions, np.ndarray):
                for i, idx in enumerate(indices):
                    actions[idx] = batch_actions[i]
            else:
                raise TypeError(
                    f"Unexpected return type from remote: {type(batch_actions)}"
                )

        if any(a is None for a in actions):
            missing = [i for i, a in enumerate(actions) if a is None]
            raise RuntimeError(f"Actions not received for indices: {missing}")

        action_array = np.stack(actions, axis=0)
        return {"action": action_array}
