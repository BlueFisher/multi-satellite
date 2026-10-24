import sys
import threading
import json
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from queue import Queue, Empty
from typing import Dict, Any, Optional

from ..Logger.LoggerConfig import *

try:
    from torch.utils.tensorboard.writer import SummaryWriter
except ImportError:
    SummaryWriter = None


class IHandler(ABC):
    """
    日志处理器抽象基类。

    Methods:
        handle_metric(metric_name, value, step, tags): 处理指标日志。
        handle_message(level, message, context): 处理消息日志。
        close(): 关闭处理器。
    """

    @abstractmethod
    def handle_metric(
        self, metric_name: str, value: float, step: Optional[int], tags: Dict[str, Any]
    ):
        """
        处理指标日志。

        Args:
            metric_name (str): 指标名称。
            value (float): 指标值。
            step (Optional[int]): 当前步骤。
            tags (Dict[str, Any]): 附加标签。

        Example:
            >>> handler.handle_metric("accuracy", 0.95, step=10, tags={"group": "train"})
        """
        pass

    @abstractmethod
    def handle_message(self, level: int, message: str, context: Dict[str, Any]):
        """
        处理消息日志。

        Args:
            level (int): 日志级别。
            message (str): 日志消息。
            context (Dict[str, Any]): 上下文信息。

        Example:
            >>> handler.handle_message(logging.INFO, "Training started", {"epoch": 1})
        """
        pass

    def close(self):
        """默认实现，子类按需重写"""
        pass


class TensorBoardHandler(IHandler):
    """
    TensorBoard 日志处理器。

    Attributes:
        writer (SummaryWriter): TensorBoard 写入器实例。
        step_map (Dict[str, int]): 步骤映射，用于记录每个指标的当前步骤。

    Methods:
        handle_metric(metric_name, value, step, tags): 记录指标到 TensorBoard。
        handle_message(level, message, context): 处理消息日志。
        close(): 关闭 TensorBoard 写入器。
    """

    def __init__(self, log_dir: str):
        if SummaryWriter is None:
            raise RuntimeError("TensorBoard support requires torch")
        self.writer = SummaryWriter(log_dir=log_dir)
        self.step_map: Dict[str, int] = defaultdict(int)

    def handle_metric(
        self, metric_name: str, value: float, step: Optional[int], tags: Dict[str, Any]
    ):
        """
        记录指标到 TensorBoard。

        Args:
            metric_name (str): 指标名称。
            value (float): 指标值。
            step (Optional[int]): 当前步骤。
            tags (Dict[str, Any]): 附加标签。

        Example:
            >>> handler.handle_metric("loss", 0.1, step=5, tags={"group": "train"})
        """
        group = tags.get("group", "default")
        key = f"{group}/{metric_name}"
        if step is None:
            self.step_map[key] += 1
            step = self.step_map[key]
        self.writer.add_scalar(key, value, step)

    def handle_message(self, level: int, message: str, context: Dict[str, Any]):
        """
        处理消息日志。

        Args:
            level (int): 日志级别。
            message (str): 日志消息。
            context (Dict[str, Any]): 上下文信息。

        Example:
            >>> handler.handle_message(logging.WARNING, "Low disk space", {})
        """
        pass

    def close(self):
        """
        关闭 TensorBoard 写入器。

        Example:
            >>> handler.close()
        """
        self.writer.close()


class ContextFormatter(logging.Formatter):
    """自定义格式化器，在消息末尾添加上下文信息"""

    def format(self, record):
        # 确保 ctx 属性存在且非空
        if hasattr(record, "ctx") and record.ctx:
            # 使用 record.msg 而非 record.message
            record.msg = f"{record.msg} {record.ctx}"
        return super().format(record)


class ConsoleHandler(IHandler):
    """
    控制台日志处理器。

    Attributes:
        logger (logging.Logger): 控制台日志记录器。
        color_map (Dict[int, str]): 日志级别到颜色的映射。

    Methods:
        handle_metric(metric_name, value, step, tags): 处理指标日志。
        handle_message(level, message, context): 处理消息日志。
    """

    def __init__(self, config: LoggerConfig):
        self.logger = logging.getLogger("console_logger")
        self.logger.setLevel(config.log_level)
        self.logger.propagate = False

        handler = logging.StreamHandler(sys.stdout)
        # 优化日志格式，使其更美观
        formatter = ContextFormatter("[%(asctime)s][%(levelname)-5s] %(message)s")
        handler.setFormatter(formatter)
        handler.setLevel(logging.DEBUG if config.debug_mode else config.log_level)

        self.logger.addHandler(handler)
        self.color_map = {
            logging.DEBUG: "\033[94m",  # 蓝色
            logging.INFO: "\033[92m",  # 绿色
            logging.WARNING: "\033[93m",  # 黄色
            logging.ERROR: "\033[91m",  # 红色
            logging.CRITICAL: "\033[95m",  # 品红色
        }

    def handle_metric(
        self, metric_name: str, value: float, step: Optional[int], tags: Dict[str, Any]
    ):
        """
        处理指标日志。

        Args:
            metric_name (str): 指标名称。
            value (float): 指标值。
            step (Optional[int]): 当前步骤。
            tags (Dict[str, Any]): 附加标签。

        Example:
            >>> handler.handle_metric("accuracy", 0.9, step=20, tags={"group": "eval"})
        """
        pass

    def handle_message(self, level: int, message: str, context: Dict[str, Any]):
        """
        处理消息日志。

        Args:
            level (int): 日志级别。
            message (str): 日志消息。
            context (Dict[str, Any]): 上下文信息。

        Example:
            >>> handler.handle_message(logging.ERROR, "File not found", {})
        """
        color = self.color_map.get(level, "")
        reset = "\033[0m"
        # 传递 context 作为 extra 字段
        self.logger.log(level, f"{color}{message}{reset}", extra={"ctx": context})

    def close(self):
        """
        关闭处理器。

        Example:
            >>> handler.close()
        """
        pass


class AsyncLogWorker(threading.Thread):
    """
    异步日志工作线程，集中处理日志分发。

    Attributes:
        queue (Queue): 日志任务队列。
        handlers (List[IHandler]): 日志处理器列表。
        running (bool): 线程运行状态。

    Methods:
        run(): 启动线程，处理日志任务。
        stop(): 停止线程。
        put(method, args): 添加日志任务到队列。
    """

    def __init__(self, handlers: List[IHandler]):
        super().__init__(daemon=True)
        self.queue = Queue()
        self.handlers = handlers
        self.running = True

    def run(self):
        """
        启动线程，处理日志任务。

        Example:
            >>> worker.run()
        """
        while self.running or not self.queue.empty():
            try:
                task = self.queue.get(timeout=1)
                for handler in self.handlers:
                    getattr(handler, task["method"])(*task["args"])
                self.queue.task_done()
            except Empty:
                continue
            except Exception as e:
                print(f"Logging error in async worker: {str(e)}", file=sys.stderr)

    def stop(self):
        """
        停止线程。

        Example:
            >>> worker.stop()
        """
        self.running = False
        self.queue.join()

    def put(self, method: str, args: tuple):
        """
        添加日志任务到队列。

        Args:
            method (str): 方法名称。
            args (tuple): 方法参数。

        Example:
            >>> worker.put("handle_message", (logging.INFO, "Task completed", {}))
        """
        self.queue.put({"method": method, "args": args})


class Logger:
    """
    多功能日志记录器。

    Attributes:
        config (LoggerConfig): 日志配置。
        handlers (List[IHandler]): 日志处理器列表。
        async_worker (AsyncLogWorker): 异步日志工作线程。

    Methods:
        add_handler(handler): 添加日志处理器。
        log_metric(name, value, step, tags): 记录指标日志。
        log_message(level, message, **context): 记录消息日志。
        debug(message, **context): 记录调试级别日志。
        info(message, **context): 记录信息级别日志。
        warning(message, **context): 记录警告级别日志。
        error(message, **context): 记录错误级别日志。
        set_mode(mode): 设置日志模式。
        add_episode_rewards(episode_rewards): 添加一轮奖励。
        step(): 记录当前步骤。
        reset(): 重置日志状态。
        close(): 关闭日志记录器。
    """

    _instance: Optional["Logger"] = None
    _lock = threading.Lock()

    def __new__(cls, config: Optional[LoggerConfig] = None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialize(config or LoggerConfig())
        return cls._instance

    def _initialize(self, config: LoggerConfig):
        self.config = config
        self.handlers: List[IHandler] = []
        self._mode = LoggerMode.TRAIN
        self._states = {
            mode: {
                "step": 0,
                "episode_rewards": defaultdict(list),
                "moving_rewards": defaultdict(
                    lambda: deque(
                        maxlen=config.get_reward_config(mode)["reward_moving_window"]
                    )
                ),
            }
            for mode in [LoggerMode.TRAIN, LoggerMode.EVAL]
        }

        self.add_handler(ConsoleHandler(config))
        if not self.config.debug_mode and self.config.enable_tensorboard:
            try:
                self.add_handler(TensorBoardHandler(self.config.tensorboard_dir))
            except Exception as e:
                self.warning(f"Failed to initialize TensorBoard: {str(e)}")

        self.async_worker = AsyncLogWorker(self.handlers)
        self.async_worker.start()

    def add_handler(self, handler: IHandler):
        """
        添加日志处理器。

        Args:
            handler (IHandler): 日志处理器实例。

        Example:
            >>> logger.add_handler(ConsoleHandler(config))
        """
        self.handlers.append(handler)

    def log_metric(
        self,
        name: str,
        value: float,
        step: Optional[int] = None,
        tags: Optional[Dict] = None,
    ):
        """
        记录指标日志。

        Args:
            name (str): 指标名称。
            value (float): 指标值。
            step (Optional[int]): 当前步骤。
            tags (Optional[Dict]): 附加标签。

        Example:
            >>> logger.log_metric("loss", 0.05, step=10, tags={"group": "train"})
        """
        if self.config.debug_mode:
            return
        tags = tags or {}
        self.async_worker.put("handle_metric", (name, value, step, tags))

    def log_message(self, level: int, message: str, **context):
        """
        记录消息日志。

        Args:
            level (int): 日志级别。
            message (str): 日志消息。
            **context: 上下文信息。

        Example:
            >>> logger.log_message(logging.INFO, "Training started", epoch=1)
        """
        format_ctx = context.pop("format", False)
        if format_ctx and context:
            formatted_ctx = json.dumps(
                context, indent=2, ensure_ascii=False, sort_keys=True, default=str
            )
        else:
            formatted_ctx = context or ""
        self.async_worker.put("handle_message", (level, message, formatted_ctx))

    def debug(self, message: str, **context):
        """
        记录调试级别日志。

        Args:
            message (str): 日志消息。
            **context: 上下文信息。

        Example:
            >>> logger.debug("Debugging model initialization")
        """
        self.log_message(logging.DEBUG, message, **context)

    def info(self, message: str, **context):
        """
        记录信息级别日志。

        Args:
            message (str): 日志消息。
            **context: 上下文信息。

        Example:
            >>> logger.info("Model training started")
        """
        self.log_message(logging.INFO, message, **context)

    def warning(self, message: str, **context):
        """
        记录警告级别日志。

        Args:
            message (str): 日志消息。
            **context: 上下文信息。

        Example:
            >>> logger.warning("Low disk space")
        """
        self.log_message(logging.WARNING, message, **context)

    def error(self, message: str, **context):
        """
        记录错误级别日志。

        Args:
            message (str): 日志消息。
            **context: 上下文信息。

        Example:
            >>> logger.error("File not found")
        """
        self.log_message(logging.ERROR, message, **context)

    def set_mode(self, mode: LoggerMode):
        """
        设置日志模式。

        Args:
            mode (LoggerMode): 日志模式。

        Example:
            >>> logger.set_mode(LoggerMode.TRAIN)
        """
        self._mode = mode

    def add_episode_rewards(self, episode_rewards: Dict[str, List[float]]):
        """
        添加一轮奖励。

        Args:
            episode_rewards (Dict[str, List[float]]): 每个代理的奖励列表。

        Example:
            >>> logger.add_episode_rewards({"agent1": [1, 2, 3], "agent2": [4, 5, 6]})
        """
        mode = self._mode
        cfg = self.config.get_reward_config(mode)
        state = self._states[mode]
        for agent, rewards in episode_rewards.items():
            if not rewards:
                continue

            if cfg["reward_aggregate_mode"] == AggregateMode.SUM:
                value = sum(rewards)
            elif cfg["reward_aggregate_mode"] == AggregateMode.MEAN:
                value = sum(rewards) / len(rewards)
            elif cfg["reward_aggregate_mode"] == AggregateMode.LAST:
                value = rewards[-1]
            else:
                raise ValueError(
                    f"Unknown aggregate mode: {cfg['reward_aggregate_mode']}"
                )

            state["episode_rewards"][agent].append(value)
            state["moving_rewards"][agent].append(value)

            if CurveType.VS_STEP in cfg["reward_curve_types"]:
                self.log_message(
                    logging.INFO,
                    f"{cfg['reward_prefix']}_vs_step_{agent}",
                    reward=value,
                    episode=state["step"],
                    tag=self._mode.value,
                )

    def step(self):
        """
        记录当前步骤。

        Example:
            >>> logger.step()
        """
        mode = self._mode
        cfg = self.config.get_reward_config(mode)
        state = self._states[mode]
        state["step"] += 1

        if state["step"] % cfg["reward_log_interval"] == 0:
            for agent, rewards in state["episode_rewards"].items():
                if not rewards:
                    continue

                recent = rewards[-cfg["reward_log_interval"] :]

                if CurveType.AVERAGE in cfg["reward_curve_types"]:
                    avg = sum(recent) / len(recent)
                    self.log_metric(
                        f"{cfg['reward_prefix']}_avg_{agent}",
                        avg,
                        step=state["step"],
                        tags={"group": mode.value},
                    )

                if CurveType.MOVING_AVG in cfg["reward_curve_types"]:
                    moving_rewards = state["moving_rewards"][agent]
                    mv = (
                        sum(moving_rewards) / len(moving_rewards)
                        if moving_rewards
                        else 0
                    )
                    self.log_metric(
                        f"{cfg['reward_prefix']}_moving_avg_{agent}",
                        mv,
                        step=state["step"],
                        tags={"group": mode.value},
                    )

                if CurveType.MAX in cfg["reward_curve_types"]:
                    mx = max(recent)
                    self.log_metric(
                        f"{cfg['reward_prefix']}_max_{agent}",
                        mx,
                        step=state["step"],
                        tags={"group": mode.value},
                    )

            self._clear_episode_rewards()

    def _clear_episode_rewards(self):
        for state in self._states.values():
            state["episode_rewards"].clear()

    def reset(self):
        """
        重置日志状态。

        Example:
            >>> logger.reset()
        """
        for state in self._states.values():
            state["step"] = 0
            state["episode_rewards"].clear()
            state["moving_rewards"].clear()

    def close(self):
        """关闭日志记录器，停止异步线程并关闭所有处理器"""

        def _close_handler(handler):
            try:
                handler.close()
            except Exception as e:
                print(f"Error closing handler: {str(e)}", file=sys.stderr)

        self.async_worker.stop()
        self.async_worker.join()
        for handler in self.handlers:
            handler.close()

    def __del__(self):
        pass


if __name__ == "__main__":
    config = LoggerConfig(use_tensorboard=True, debug_mode=False)

    logger = Logger(config)

    # 记录指标
    logger.log_metric("accuracy", 0.85, step=1, tags={"group": "val"})

    # 记录文本日志（包含上下文）
    logger.info("Training started", epoch=10, lr=0.001)

    # 记录文本日志（无上下文）
    logger.warning("This is a warning message with no extra context.")

    # 模拟添加奖励
    logger.set_mode(LoggerMode.TRAIN)
    episode_rewards = {"agent1": [1, 2, 3], "agent2": [4, 5, 6]}
    logger.add_episode_rewards(episode_rewards)
    logger.step()

    # 再次模拟 step
    episode_rewards = {"agent1": [7, 8, 9], "agent2": [10, 11, 12]}
    logger.add_episode_rewards(episode_rewards)
    logger.step()

    # 在Debug模式下自动禁用所有日志
    debug_config = LoggerConfig(debug_mode=True)
    debug_logger = Logger(debug_config)
    debug_logger.log_metric("loss", 1.2)
    debug_logger.info("This is a debug message.")

    # 显式关闭日志器
    logger.close()
    debug_logger.close()
