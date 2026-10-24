import logging
from enum import Enum, auto
from typing import Optional, List


class CurveType(Enum):
    """
    曲线类型枚举类。

    枚举值:
        AVERAGE: 平均值曲线。
        MOVING_AVG: 移动平均曲线。
        MAX: 最大值曲线。
        VS_STEP: 步数对比曲线。
    """
    AVERAGE = auto()
    MOVING_AVG = auto()
    MAX = auto()
    VS_STEP = auto()


class AggregateMode(Enum):
    """
    聚合模式枚举类。

    枚举值:
        SUM: 求和模式。
        MEAN: 平均值模式。
        LAST: 最后一个值模式。
    """
    SUM = auto()
    MEAN = auto()
    LAST = auto()


class LoggerMode(Enum):
    """
    日志模式枚举类。

    枚举值:
        TRAIN: 训练模式。
        EVAL: 评估模式。
    """
    TRAIN = "train"
    EVAL = "eval"


class LoggerConfig:
    """
    日志配置类，用于管理训练和评估过程中的日志记录配置。

    属性:
        enable_tensorboard (bool): 是否启用 TensorBoard。
        tensorboard_dir (str): TensorBoard 日志存储目录。
        debug_mode (bool): 是否启用调试模式。
        log_level (int): 日志记录级别。
        log_file (Optional[str]): 日志文件路径。
        max_bytes (int): 日志文件的最大字节数。
        backup_count (int): 日志文件的备份数量。
        when (str): 日志轮转的时间单位。
        interval (int): 日志轮转的时间间隔。
        train_reward_log_interval (int): 训练奖励的日志记录间隔。
        train_reward_aggregate_mode (AggregateMode): 训练奖励的聚合模式。
        train_reward_moving_window (int): 训练奖励的移动窗口大小。
        train_reward_prefix (str): 训练奖励的前缀。
        train_reward_curve_types (List[CurveType]): 训练奖励的曲线类型。
        eval_reward_log_interval (int): 评估奖励的日志记录间隔。
        eval_reward_aggregate_mode (AggregateMode): 评估奖励的聚合模式。
        eval_reward_moving_window (int): 评估奖励的移动窗口大小。
        eval_reward_prefix (str): 评估奖励的前缀。
        eval_reward_curve_types (List[CurveType]): 评估奖励的曲线类型。
    """

    def __init__(
        self,
        use_tensorboard: bool = False,
        tensorboard_dir: str = "./logs/tensorboard",
        debug_mode: bool = False,
        log_level: int = logging.INFO,
        log_file: Optional[str] = "./logs/app.log",
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        when: str = "midnight",
        interval: int = 1,
        train_reward_log_interval: int = 10,
        train_reward_aggregate_mode: AggregateMode = AggregateMode.SUM,
        train_reward_moving_window: int = 100,
        train_reward_prefix: str = "train_reward",
        train_reward_curve_types: List[CurveType] = None,
        eval_reward_log_interval: int = 10,
        eval_reward_aggregate_mode: AggregateMode = AggregateMode.SUM,
        eval_reward_moving_window: int = 100,
        eval_reward_prefix: str = "eval_reward",
        eval_reward_curve_types: List[CurveType] = None,
    ):
        """
        初始化 LoggerConfig 实例。

        参数:
            use_tensorboard (bool): 是否启用 TensorBoard。
            tensorboard_dir (str): TensorBoard 日志存储目录。
            debug_mode (bool): 是否启用调试模式。
            log_level (int): 日志记录级别。
            log_file (Optional[str]): 日志文件路径。
            max_bytes (int): 日志文件的最大字节数。
            backup_count (int): 日志文件的备份数量。
            when (str): 日志轮转的时间单位。
            interval (int): 日志轮转的时间间隔。
            train_reward_log_interval (int): 训练奖励的日志记录间隔。
            train_reward_aggregate_mode (AggregateMode): 训练奖励的聚合模式。
            train_reward_moving_window (int): 训练奖励的移动窗口大小。
            train_reward_prefix (str): 训练奖励的前缀。
            train_reward_curve_types (List[CurveType]): 训练奖励的曲线类型。
            eval_reward_log_interval (int): 评估奖励的日志记录间隔。
            eval_reward_aggregate_mode (AggregateMode): 评估奖励的聚合模式。
            eval_reward_moving_window (int): 评估奖励的移动窗口大小。
            eval_reward_prefix (str): 评估奖励的前缀。
            eval_reward_curve_types (List[CurveType]): 评估奖励的曲线类型。
        """
        self.enable_tensorboard = use_tensorboard
        self.tensorboard_dir = tensorboard_dir
        self.debug_mode = debug_mode
        self.log_level = log_level
        self.log_file = log_file
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.when = when
        self.interval = interval
        self.train_reward_log_interval = train_reward_log_interval
        self.train_reward_aggregate_mode = train_reward_aggregate_mode
        self.train_reward_moving_window = train_reward_moving_window
        self.train_reward_prefix = train_reward_prefix
        self.train_reward_curve_types = train_reward_curve_types or [
            CurveType.AVERAGE,
            CurveType.MOVING_AVG,
            CurveType.MAX,
            CurveType.VS_STEP,
        ]
        self.eval_reward_log_interval = eval_reward_log_interval
        self.eval_reward_aggregate_mode = eval_reward_aggregate_mode
        self.eval_reward_moving_window = eval_reward_moving_window
        self.eval_reward_prefix = eval_reward_prefix
        self.eval_reward_curve_types = eval_reward_curve_types or [
            CurveType.AVERAGE,
            CurveType.MOVING_AVG,
            CurveType.MAX,
            CurveType.VS_STEP,
        ]

    def get_reward_config(self, mode: LoggerMode):
        """
        根据模式获取对应的奖励配置。

        参数:
            mode (LoggerMode): 日志模式，TRAIN 或 EVAL。

        返回:
            dict: 包含奖励配置的字典。

        示例:
            >>> config = LoggerConfig()
            >>> train_config = config.get_reward_config(LoggerMode.TRAIN)
            >>> eval_config = config.get_reward_config(LoggerMode.EVAL)
        """
        if mode == LoggerMode.TRAIN:
            return {
                "reward_log_interval": self.train_reward_log_interval,
                "reward_aggregate_mode": self.train_reward_aggregate_mode,
                "reward_moving_window": self.train_reward_moving_window,
                "reward_prefix": self.train_reward_prefix,
                "reward_curve_types": self.train_reward_curve_types,
            }
        elif mode == LoggerMode.EVAL:
            return {
                "reward_log_interval": self.eval_reward_log_interval,
                "reward_aggregate_mode": self.eval_reward_aggregate_mode,
                "reward_moving_window": self.eval_reward_moving_window,
                "reward_prefix": self.eval_reward_prefix,
                "reward_curve_types": self.eval_reward_curve_types,
            }
        else:
            raise ValueError(f"Unsupported mode: {mode}")
