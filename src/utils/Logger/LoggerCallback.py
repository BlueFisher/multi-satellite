from utils.Logger.Logger import Logger
from utils.Logger.LoggerConfig import LoggerMode
import numpy as np


class BaseLoggerCallback:
    """
    基础日志回调类，定义了日志记录的基本接口。

    属性:
        args: 配置参数。
        logger (Logger): 用于记录日志的 Logger 实例。
    """

    def __init__(self, logger: Logger, args):
        """
        初始化 BaseLoggerCallback 实例。

        参数:
            logger (Logger): 用于记录日志的 Logger 实例。
            args: 配置参数。
        """
        self.args = args
        self.logger = logger

    def after_episode(self, episode_rewards, info):
        """
        每个训练回合结束后调用，用于更新相关信息。

        参数:
            episode_rewards: 当前回合的奖励。
            info: 包含当前回合的相关信息。
        """
        pass

    def on_train_log_interval(self, episode):
        """
        在训练日志记录间隔时调用。

        参数:
            episode: 当前的回合数。
        """
        pass

    def after_eval_episode(self, episode_rewards, info):
        """
        每个评估回合结束后调用。

        参数:
            episode_rewards: 当前评估回合的奖励。
            info: 包含当前评估回合的相关信息。
        """
        pass

    def on_eval_log_interval(self, info):
        """
        在评估日志记录间隔时调用。

        参数:
            info: 包含评估过程的相关信息。
        """
        pass


class Satellite3v1LoggerCallback(BaseLoggerCallback):
    """
    3v1 卫星场景的日志回调类，用于记录训练和评估过程中的相关指标。

    属性:
        win (int): 胜利次数。
        collided (int): 碰撞次数。
        step (int): 总步数。
        capture_step (int): 捕获持续时间。
        total_capture_step (int): 总捕获持续时间。
        out_side (int): 超出边界次数。
        eval_capture_count (int): 评估捕获次数。
        eval_collided_count (int): 评估碰撞次数。
        eval_overtime_count (int): 评估超时次数。
        eval_total_steps (int): 评估总步数。
    """

    def __init__(self, logger: Logger, args):
        """
        初始化 Satellite3v1LoggerCallback 实例。

        参数:
            logger (Logger): 用于记录日志的 Logger 实例。
            args: 配置参数。
        """
        super().__init__(logger, args)
        self._reset_train_values()
        self._reset_eval_values()

    def _reset_train_values(self):
        """
        重置训练相关的指标。
        """
        self.win = 0
        self.collided = 0
        self.step = 0
        self.capture_step = 0
        self.total_capture_step = 0
        self.out_side = 0

    def _reset_eval_values(self):
        """
        重置评估相关的指标。
        """
        self.eval_capture_count = 0
        self.eval_collided_count = 0
        self.eval_overtime_count = 0
        self.eval_total_steps = 0

    def after_episode(self, episode_rewards, info):
        """
        每个训练回合结束后更新训练指标。

        参数:
            episode_rewards: 当前回合的奖励。
            info (dict): 包含当前回合的相关信息，例如碰撞次数、步数等。

        示例:
            >>> callback = Satellite3v1LoggerCallback(logger, args)
            >>> callback.after_episode(episode_rewards, {"capture_reward_claimed": True, "collided": False})
        """
        if info["capture_reward_claimed"]:
            self.win += 1

        if info["collided"]:
            self.collided += 1
        if len(info["out_satellite"]) > 0:
            self.out_side += 1

        self.step += info["current_step"]
        self.capture_step += info["capture_sustain_counter"]
        self.total_capture_step += info["total_capture_sustain_counter"]

    def on_train_log_interval(self, episode):
        """
        在训练日志记录间隔时记录训练指标。

        参数:
            episode (int): 当前的回合数。

        示例:
            >>> callback = Satellite3v1LoggerCallback(logger, args)
            >>> callback.on_train_log_interval(10)
        """
        self.logger.log_metric("win_count", self.win, episode, tags={"group": "Info"})
        self.logger.log_metric(
            "collided_count", self.collided, episode, tags={"group": "Info"}
        )

        self.logger.log_metric(
            "steps", self.step / self.args.log_interval, episode, tags={"group": "Info"}
        )
        self.logger.log_metric(
            "capture_step",
            self.capture_step / self.args.log_interval,
            episode,
            tags={"group": "Info"},
        )

        self.logger.log_metric(
            "total_capture_step",
            self.total_capture_step / self.args.log_interval,
            episode,
            tags={"group": "Info"},
        )
        self.logger.log_metric(
            "out_side_count",
            self.out_side,
            episode,
            tags={"group": "Info"},
        )

        self._reset_train_values()

    def after_eval_episode(self, episode_rewards, info):
        """
        每个评估回合结束后更新评估指标。

        参数:
            episode_rewards: 当前评估回合的奖励。
            info (dict): 包含当前评估回合的相关信息。

        示例:
            >>> callback = Satellite3v1LoggerCallback(logger, args)
            >>> callback.after_eval_episode(episode_rewards, {"capture": True, "collided": False})
        """
        self.eval_capture_count += 1 if info["capture"] else 0
        self.eval_collided_count += 1 if info["collided"] else 0
        self.eval_overtime_count += 1 if info["runtime_over"] else 0
        self.eval_total_steps += info["steps"]

    def on_eval_log_interval(self, info):
        """
        在评估日志记录间隔时记录评估指标。

        参数:
            info (dict): 包含评估过程的相关信息。

        示例:
            >>> callback = Satellite3v1LoggerCallback(logger, args)
            >>> callback.on_eval_log_interval(info)
        """
        self.logger.log_metric(
            "capture_count",
            self.eval_capture_count / self.args.eval_episode,
            self.logger._states[LoggerMode.EVAL]["step"],
            tags={"group": "eval_info"},
        )
        self.logger.log_metric(
            "collided_count",
            self.eval_collided_count / self.args.eval_episode,
            self.logger._states[LoggerMode.EVAL]["step"],
            tags={"group": "eval_info"},
        )
        self.logger.log_metric(
            "overtime_count",
            self.eval_overtime_count / self.args.eval_episode,
            self.logger._states[LoggerMode.EVAL]["step"],
            tags={"group": "eval_info"},
        )
        self.logger.log_metric(
            "total_steps",
            self.eval_total_steps / self.args.eval_episode,
            self.logger._states[LoggerMode.EVAL]["step"],
            tags={"group": "eval_info"},
        )

        if self.args.debug_mode:
            print("eval episode: ", self.args.eval_episode)
            print("capture_count:", self.eval_capture_count / self.args.eval_episode)
            print("collided_count:", self.eval_collided_count / self.args.eval_episode)
            print("overtime_count:", self.eval_overtime_count / self.args.eval_episode)
            print("total_steps:", self.eval_total_steps / self.args.eval_episode)

        self._reset_eval_values()


class Satellite3v1HvtLoggerCallback(BaseLoggerCallback):
    def __init__(self, logger: Logger, args):
        super().__init__(logger, args)
        self._reset_train_values()
        self._reset_eval_values()

    def _reset_train_values(self):
        """
        重置训练相关的指标。
        """
        self.win = 0
        self.collided = 0
        self.step = 0
        self.capture_step = 0
        self.total_capture_step = 0
        self.failure_step = 0
        self.out_side = 0

    def _reset_eval_values(self):
        """
        重置评估相关的指标。
        """
        self.eval_capture_count = 0
        self.eval_collided_count = 0
        self.eval_outside_count = 0
        self.eval_failure_count = 0
        self.eval_total_steps = 0
        self.eval_win_count = 0

    def after_episode(self, episode_rewards, info):
        """
        每个训练回合结束后更新训练指标。

        参数:
            episode_rewards: 当前回合的奖励。
            info (dict): 包含当前回合的相关信息，例如碰撞次数、步数等。

        示例:
            >>> callback = Satellite3v1HvtLoggerCallback(logger, args)
            >>> callback.after_episode(episode_rewards, {"capture_reward_claimed": True, "collided": False})
        """
        if info["capture_reward_claimed"]:
            self.win += 1

        if info["collided"]:
            self.collided += 1
        if len(info["out_satellite"]) > 0:
            self.out_side += 1

        self.step += info["current_step"]
        self.capture_step += info["capture_sustain_counter"]
        self.total_capture_step += info["total_capture_sustain_counter"]
        self.failure_step += info["hvt_unsafe"]

    def on_train_log_interval(self, episode):
        """
        在训练日志记录间隔时记录训练指标。

        参数:
            episode (int): 当前的回合数。

        示例:
            >>> callback = Satellite3v1HvtLoggerCallback(logger, args)
            >>> callback.on_train_log_interval(10)
        """
        self.logger.log_metric("win_count", self.win, episode, tags={"group": "Info"})
        self.logger.log_metric(
            "collided_count", self.collided, episode, tags={"group": "Info"}
        )

        self.logger.log_metric(
            "steps",
            self.step / self.args.log_interval,
            episode,
            tags={"group": "Info"},
        )
        self.logger.log_metric(
            "capture_step",
            self.capture_step / self.args.log_interval,
            episode,
            tags={"group": "Info"},
        )

        self.logger.log_metric(
            "total_capture_step",
            self.total_capture_step / self.args.log_interval,
            episode,
            tags={"group": "Info"},
        )
        self.logger.log_metric(
            "out_side_count",
            self.out_side,
            episode,
            tags={"group": "Info"},
        )

        self.logger.log_metric(
            "failure_count",
            self.failure_step,
            episode,
            tags={"group": "Info"},
        )

        self._reset_train_values()

    def after_eval_episode(self, episode_rewards, info):
        """
        每个评估回合结束后更新评估指标。

        参数:
            episode_rewards: 当前评估回合的奖励。
            info (dict): 包含当前评估回合的相关信息。

        示例:
            >>> callback = Satellite3v1HvtLoggerCallback(logger, args)
            >>> callback.after_eval_episode(episode_rewards, {"capture": True, "collided": False})
        """
        self.eval_capture_count += info["total_capture_sustain_counter"]
        self.eval_collided_count += 1 if info["collided"] else 0
        self.eval_outside_count += 1 if len(info["out_satellite"]) > 0 else 0
        self.eval_failure_count += 1 if info["hvt_unsafe"] else 0
        self.eval_total_steps += info["current_step"]
        self.eval_win_count += 1 if info["capture_reward_claimed"] else 0
        if self.args.just_eval:
            print(info)

    def on_eval_log_interval(self, info):
        """
        在评估日志记录间隔时记录评估指标。

        参数:
            info (dict): 包含评估过程的相关信息。

        示例:
            >>> callback = Satellite3v1HvtLoggerCallback(logger, args)
            >>> callback.on_eval_log_interval(info)
        """
        if self.logger is not None:
            self.logger.log_metric(
                "capture_count",
                self.eval_capture_count / self.args.eval_episode,
                self.logger._states[LoggerMode.EVAL]["step"],
                tags={"group": "eval_info"},
            )
            self.logger.log_metric(
                "collided_count",
                self.eval_collided_count / self.args.eval_episode,
                self.logger._states[LoggerMode.EVAL]["step"],
                tags={"group": "eval_info"},
            )
            self.logger.log_metric(
                "outside_count",
                self.eval_outside_count / self.args.eval_episode,
                self.logger._states[LoggerMode.EVAL]["step"],
                tags={"group": "eval_info"},
            )
            self.logger.log_metric(
                "total_steps",
                self.eval_total_steps / self.args.eval_episode,
                self.logger._states[LoggerMode.EVAL]["step"],
                tags={"group": "eval_info"},
            )

            self.logger.log_metric(
                "failure_count",
                self.eval_failure_count / self.args.eval_episode,
                self.logger._states[LoggerMode.EVAL]["step"],
                tags={"group": "eval_info"},
            )

        if self.args.debug_mode:
            print("eval episode: ", self.args.eval_episode)
            print("capture_count:", self.eval_capture_count / self.args.eval_episode)
            print("collided_count:", self.eval_collided_count / self.args.eval_episode)
            print("outside_count:", self.eval_outside_count / self.args.eval_episode)
            print("failure_count:", self.eval_failure_count / self.args.eval_episode)
            print("total_steps:", self.eval_total_steps / self.args.eval_episode)
            print("win count:", self.eval_win_count)

        self._reset_eval_values()


class Satellite2v2LoggerCallback(BaseLoggerCallback):
    """
    2v2 卫星场景的日志回调类，用于记录训练和评估过程中的相关指标。

    属性:
        step (int): 总步数。
        failure (int): 失败次数。
        collided (int): 碰撞次数。
        out_side (int): 超出边界次数。
        hvt_breach (int): 高价值目标突破次数。
        win (int): 胜利次数。
        eval_win_count (int): 评估胜利次数。
        eval_failure (int): 评估失败次数。
        eval_collided (int): 评估碰撞次数。
        eval_out_side (int): 评估超出边界次数。
        eval_hvt_breach (int): 评估高价值目标突破次数。
    """

    def __init__(self, logger: Logger, args):
        """
        初始化 Satellite2v2LoggerCallback 实例。

        参数:
            logger (Logger): 用于记录日志的 Logger 实例。
            args: 配置参数。
        """
        super().__init__(logger, args)
        self._reset_train_values()
        self._reset_eval_values()

    def _reset_train_values(self):
        """
        重置训练相关的指标。
        """
        self.step = 0
        self.failure = 0
        self.collided = 0
        self.out_side = 0
        self.hvt_breach = 0
        self.win = 0
        self.max_durations = {}
        self.total_durations = {}

    def _reset_eval_values(self):
        """
        重置评估相关的指标。
        """
        self.eval_win_count = 0
        self.eval_failure = 0
        self.eval_collided = 0
        self.eval_out_side = 0
        self.eval_hvt_breach = 0
        self.eval_max_durations = {}
        self.eval_total_durations = {}

    def after_episode(self, episode_rewards, info):
        """
        每个训练回合结束后更新训练指标。

        参数:
            episode_rewards: 当前回合的奖励。
            info (dict): 包含当前回合的相关信息，例如碰撞次数、步数等。

        示例:
            >>> callback = Satellite2v2LoggerCallback(logger, args)
            >>> callback.after_episode(episode_rewards, {"current_step": 100, "collided": True})
        """
        self.step += info["current_step"]
        if (
            info["is_hvt_breach"]
            or info["collided"]
            or len(info["out_of_boundary"]) > 0
        ):
            self.failure += 1
        if info["collided"]:
            self.collided += 1
        if len(info["out_of_boundary"]):
            self.out_side += 1
        if info["is_hvt_breach"]:
            self.hvt_breach += 1
        if info["all_intercept_claimed"]:
            self.win += 1
        # for key, value in info.items():
        #     if "interceptor_duration" in key:
        #         if key not in self.train_durations:
        #             self.train_durations[key] = 0
        #         self.train_durations[key] += value
        for key, value in info.items():
            if key.startswith("_"):
                continue
            if "max_interceptor_duration" in key:
                if key not in self.max_durations:
                    self.max_durations[key] = 0
                if isinstance(value, (list, tuple, np.ndarray)):
                    self.max_durations[key] += np.sum(value)
                else:
                    self.max_durations[key] += value
            if "total_interceptor_duration" in key:
                if key not in self.total_durations:
                    self.total_durations[key] = 0
                if isinstance(value, (list, tuple, np.ndarray)):
                    self.total_durations[key] += np.sum(value)
                else:
                    self.total_durations[key] += value

    def on_train_log_interval(self, episode):
        """
        在训练日志记录间隔时记录训练指标。

        参数:
            episode (int): 当前的回合数。

        示例:
            >>> callback = Satellite2v2LoggerCallback(logger, args)
            >>> callback.on_train_log_interval(10)
        """
        self.logger.log_metric(
            "steps", self.step / self.args.log_interval, episode, tags={"group": "Info"}
        )
        self.logger.log_metric(
            "collided",
            self.collided / self.args.log_interval,
            episode,
            tags={"group": "Info"},
        )
        self.logger.log_metric(
            "out_of_boundary",
            self.out_side / self.args.log_interval,
            episode,
            tags={"group": "Info"},
        )
        self.logger.log_metric(
            "hvt_breach",
            self.hvt_breach / self.args.log_interval,
            episode,
            tags={"group": "Info"},
        )
        self.logger.log_metric(
            "win_rate",
            self.win / self.args.log_interval,
            episode,
            tags={"group": "Info"},
        )
        self.logger.log_metric(
            "failure_rate",
            self.failure / self.args.log_interval,
            episode,
            tags={"group": "Info"},
        )
        for key, max_duration in self.max_durations.items():
            self.logger.log_metric(
                key,
                max_duration / self.args.log_interval,
                episode,
                tags={"group": "Duration"}
            )
        for key, total_duration in self.total_durations.items():
            self.logger.log_metric(
                key,
                total_duration / self.args.log_interval,
                episode,
                tags={"group": "Duration"}
            )

        self._reset_train_values()

    def after_eval_episode(self, episode_rewards, info):
        """
        每个评估回合结束后更新评估指标。

        参数:
            episode_rewards: 当前评估回合的奖励。
            info (dict): 包含当前评估回合的相关信息。

        示例:
            >>> callback = Satellite2v2LoggerCallback(logger, args)
            >>> callback.after_eval_episode(episode_rewards, {"all_intercept_claimed": True})
        """
        if info["all_intercept_claimed"]:
            self.eval_win_count += 1
        if (
            info["is_hvt_breach"]
            or info["collided"]
            or len(info["out_of_boundary"]) > 0
        ):
            self.eval_failure += 1
        if info["collided"]:
            self.eval_collided += 1
        if len(info["out_of_boundary"]):
            self.eval_out_side += 1
        if info["is_hvt_breach"]:
            self.eval_hvt_breach += 1

        for key, value in info.items():
            if key.startswith("_"):
                continue
            if "max_interceptor_duration" in key:
                if key not in self.eval_max_durations:
                    self.eval_max_durations[key] = 0
                if isinstance(value, (list, tuple, np.ndarray)):
                    self.eval_max_durations[key] += np.sum(value)
                else:
                    self.eval_max_durations[key] += value
            if "total_interceptor_duration" in key:
                if key not in self.eval_total_durations:
                    self.eval_total_durations[key] = 0
                if isinstance(value, (list, tuple, np.ndarray)):
                    self.eval_total_durations[key] += np.sum(value)
                else:
                    self.eval_total_durations[key] += value

        if self.args.just_eval:
            print(info)

    def on_eval_log_interval(self, info):
        """
        在评估日志记录间隔时记录评估指标。

        参数:
            info (dict): 包含评估过程的相关信息。

        示例:
            >>> callback = Satellite2v2LoggerCallback(logger, args)
            >>> callback.on_eval_log_interval(info)
        """
        if not self.args.debug_mode:
            self.logger.log_metric(
                "eval_win",
                self.eval_win_count / self.args.eval_episode,
                self.logger._states[LoggerMode.EVAL]["step"],
                tags={"group": "eval_info"},
            )
            self.logger.log_metric(
                "eval_failure",
                self.eval_failure / self.args.eval_episode,
                self.logger._states[LoggerMode.EVAL]["step"],
                tags={"group": "eval_info"},
            )
            self.logger.log_metric(
                "eval_collided",
                self.eval_collided / self.args.eval_episode,
                self.logger._states[LoggerMode.EVAL]["step"],
                tags={"group": "eval_info"},
            )
            self.logger.log_metric(
                "eval_hvt_breach",
                self.eval_hvt_breach / self.args.eval_episode,
                self.logger._states[LoggerMode.EVAL]["step"],
                tags={"group": "eval_info"},
            )
            self.logger.log_metric(
                "eval_out_side",
                self.eval_out_side / self.args.eval_episode,
                self.logger._states[LoggerMode.EVAL]["step"],
                tags={"group": "eval_info"},
            )

            for key, max_durations in self.eval_max_durations.items():
                self.logger.log_metric(
                    f"eval_{key}",
                    max_durations / self.args.eval_episode,
                    self.logger._states[LoggerMode.EVAL]["step"],
                    tags={"group": "eval_info"}
                )
            for key, total_durations in self.eval_total_durations.items():
                self.logger.log_metric(
                    f"eval_{key}",
                    total_durations / self.args.eval_episode,
                    self.logger._states[LoggerMode.EVAL]["step"],
                    tags={"group": "eval_info"}
                )

        if self.args.debug_mode:
            print("eval episode: ", self.args.eval_episode)
            print("win count:", self.eval_win_count)
            print("win rate:", self.eval_win_count / self.args.eval_episode)
            print("failure rate:", self.eval_failure / self.args.eval_episode)
            print("collided rate:", self.eval_collided / self.args.eval_episode)
            print("hvt_breach rate:", self.eval_hvt_breach / self.args.eval_episode)
            print("out_side rate:", self.eval_out_side / self.args.eval_episode)
            for key, max_duration in self.eval_max_durations.items():
                print(f"eval max duration {key} avg:", max_duration / self.args.eval_episode)
            for key, total_duration in self.eval_total_durations.items():
                print(f"eval total duration {key} avg:", total_duration / self.args.eval_episode)
        self._reset_eval_values()
