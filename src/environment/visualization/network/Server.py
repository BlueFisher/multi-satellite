import json
import threading
import time
from typing import Callable, Dict, Any

import zmq


def _handle_verify_connection(_):
    """
    处理连接验证请求。

    用途：
        响应Unity客户端的 VerifyConnection 请求，返回固定字符串表示连接已验证。

    参数:
        _: 任意传入数据（未使用）

    返回:
        str: "connection_verified"

    示例:
        >>> _handle_verify_connection(None)
        'connection_verified'
    """
    return "connection_verified"


def _handle_heartbeat_request(_):
    """
    处理心跳请求。

    用途：
        响应Unity客户端的 HeartbeatRequest 请求，返回固定字符串表示服务器存活。

    参数:
        _: 任意传入数据（未使用）

    返回:
        str: "heartbeat_response"

    示例:
        >>> _handle_heartbeat_request(None)
        'heartbeat_response'
    """
    return "heartbeat_response"


class ZMQServer:
    """
    ZeroMQ服务器，处理与Unity的双向通信。

    功能：
        1. 发布游戏状态到Unity客户端（PUB模式）；
        2. 接收Unity请求并返回响应（REP模式）；
        3. 支持注册自定义请求处理器；
        4. 支持发送控制消息到Unity（线程安全队列）。

    属性:
        context: zmq上下文
        pub_socket: 用于发布游戏状态的PUB套接字
        rep_socket: 用于处理请求-响应的REP套接字
        request_handlers: 注册的请求处理器字典
        running: 服务器运行状态标志
        rep_thread: 请求处理线程
        control_thread: 控制消息处理线程（可扩展）
        control_messages: 待发送控制消息队列
        control_lock: 控制消息队列锁

    示例:
        >>> server = ZMQServer(pub_port=5556, rep_port=5557)
        >>> server.start()
    """

    def __init__(
        self, pub_port: int = 5556, rep_port: int = 5557, control_port: int = 5558
    ):
        """
        初始化ZMQ服务器

        Args:
            pub_port: 发布游戏状态的端口
            rep_port: 处理请求-响应的端口
            control_port: 发送控制流的端口
        """
        self.context = zmq.Context()

        # 用于发布游戏状态的套接字
        self.pub_socket = self.context.socket(zmq.PUB)
        self.pub_socket.bind(f"tcp://*:{pub_port}")

        # 用于处理请求-响应的套接字
        self.rep_socket = self.context.socket(zmq.REP)
        self.rep_socket.bind(f"tcp://*:{rep_port}")

        # 存储注册的请求处理器
        self.request_handlers: Dict[str, Callable[[str], str]] = {}

        # 线程相关
        self.running = False
        self.rep_thread = None
        self.control_thread = None

        # 控制流队列
        self.control_messages = []
        self.control_lock = threading.Lock()

        self._register_base_handler()

    def _register_base_handler(self):
        """
        注册基础协议处理器。

        功能：
            1. 注册连接验证处理器（verify_connection）
            2. 注册心跳请求处理器（heartbeat_request）

        示例:
            >>> server._register_base_handler()
        """
        # 1. 连接验证处理器（对应Unity的VerifyConnection）
        self.register_request_handler(
            request_type="verify_connection", handler=_handle_verify_connection
        )
        # 2. 心跳请求处理器（对应Unity的HeartbeatRequest）
        self.register_request_handler(
            request_type="heartbeat_request", handler=_handle_heartbeat_request
        )

    def register_request_handler(
        self, request_type: str, handler: Callable[[str], str]
    ):
        """
        注册请求处理器

        Args:
            request_type: 请求类型
            handler: 处理函数，接收请求数据并返回响应数据
        """
        self.request_handlers[request_type] = handler

    def publish_game_state(self, topic: str, data: Dict[str, Any]):
        """
        发布状态数据

        Args:
            topic: 主题
            data: 要发布的数据
        """
        message = json.dumps({"timestamp": time.time(), "data": data})
        self.pub_socket.send_string(f"{topic} {message}")

    def send_control_message(self, topic: str, data: Dict[str, Any]):
        """
        发送控制消息到Unity

        Args:
            topic: 控制消息主题
            data: 控制消息数据
        """
        with self.control_lock:
            self.control_messages.append((topic, data))

    def start(self):
        """
        启动服务器线程，开始处理请求。

        功能：
            1. 启动REP请求处理线程
            2. 设置运行标志为True

        示例:
            >>> server.start()
        """
        if self.running:
            return

        self.running = True

        # 启动请求处理线程
        self.rep_thread = threading.Thread(target=self._handle_requests, daemon=True)
        self.rep_thread.start()

    def stop(self):
        """
        停止服务器线程。

        功能：
            1. 设置运行标志为False
            2. 停止REP请求处理线程并等待线程结束

        示例:
            >>> server.stop()
        """
        self.running = False

        if self.rep_thread:
            self.rep_thread.join(timeout=1.0)

        if self.control_thread:
            self.control_thread.join(timeout=1.0)

        self.pub_socket.close()
        self.rep_socket.close()

        self.context.destroy()

    def _handle_requests(self):
        """
        处理请求-响应循环。

        功能：
            1. 循环监听REP套接字请求
            2. 根据请求类型调用对应处理器
            3. 将响应发送回客户端
            4. 异常捕获并返回错误信息

        示例:
            >>> server._handle_requests()
        """
        poller = zmq.Poller()
        poller.register(self.rep_socket, zmq.POLLIN)

        while self.running:
            try:
                socks = dict(poller.poll(timeout=100))  # 100ms超时

                if self.rep_socket in socks and socks[self.rep_socket] == zmq.POLLIN:
                    # 接收请求
                    request = self.rep_socket.recv_string()

                    try:
                        # 解析请求
                        request_parts = request.split(" ", 1)
                        request_type = request_parts[0]
                        request_data = (
                            request_parts[1] if len(request_parts) > 1 else ""
                        )

                        # 处理请求
                        if request_type in self.request_handlers:
                            response = self.request_handlers[request_type](request_data)
                        else:
                            response = json.dumps(
                                {"error": f"未知请求类型: {request_type}"}
                            )

                        # 发送响应
                        self.rep_socket.send_string(response)

                    except Exception as e:
                        error_response = json.dumps(
                            {"error": f"处理请求时出错: {str(e)}"}
                        )
                        self.rep_socket.send_string(error_response)

            except Exception as e:
                if self.running:
                    print(f"处理请求时发生异常: {str(e)}")
