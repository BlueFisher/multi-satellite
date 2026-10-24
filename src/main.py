"""强化学习训练与评估主程序。

该模块是强化学习训练和评估流程的主入口，负责配置加载、环境构建、
模型训练和评估等核心功能。

Attributes:
    None

Example:
    >>> python main.py --env_name SatelliteEnv3v1 --alg_name PPO
    启动一个3v1卫星环境的PPO算法训练

Note:
    支持多进程环境并行训练，支持Windows和Linux平台
"""

import os


import warnings

import multiprocessing as mp
import time
from argparse import Namespace
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
from gymnasium import VectorizeMode

warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

from Evaluation import Evaluator
from environment import SimulatorType, register_env
from eval_alg import eval_func
from train import train
from utils import Logger
from utils.Logger.LoggerCallback import (
    Satellite3v1LoggerCallback,
    Satellite3v1HvtLoggerCallback,
    Satellite2v2LoggerCallback,
)
from utils.Logger.LoggerConfig import AggregateMode, LoggerConfig
from utils.utils_func import (
    load_config,
    load_args,
    update_args,
    get_alg_manager,
    is_linux,
)


def should_pause_before_close(args: Namespace) -> bool:
    """仅在调用方明确启用时等待关闭控制台。"""

    return bool(getattr(args, "pause_before_close", False))


def start_info():
    print("欢迎使用星群智能协同一体化虚实结合训练与演示系统")


def update_hyperparameter(args: Namespace) -> None:
    """根据环境数量更新超参数。

    根据并行环境数量调整批次大小和缓冲区大小。

    Args:
        args: 包含超参数的命令行参数

    Returns:
        None
    """
    num_envs = args.num_envs
    args.batch_size = args.batch_size * num_envs
    args.buffer_size = args.buffer_size * min(num_envs, 4)


def main() -> None:
    """主函数。

    强化学习训练与评估流程的主要控制函数，包括：
    1. 加载配置参数
    2. 创建环境和算法
    3. 构建数据标准化器
    4. 训练或评估模型

    Returns:
        None

    Raises:
        FileNotFoundError: 如果配置文件不存在
        RuntimeError: 如果环境创建失败
        ValueError: 如果参数配置错误
    """
    # 1. 加载默认配置
    args = load_config("configs/default_config.yaml")

    # 2. 第一次读取命令行参数，获取可能修改的alg_name和env_name等基础配置
    cmd_args_first_pass = load_args(deepcopy(args))

    # unity 会传入这个参数, 测试环境尽量不要使用这个参数帮助debug
    # torch 和 gym 会弹出一些无关紧要的警告，这里为了打印清楚日志直接忽略掉
    if (
        hasattr(cmd_args_first_pass, "filter_warnings")
        and cmd_args_first_pass.filter_warnings
    ):
        warnings.simplefilter("ignore")

    # 3. 读取算法配置（使用可能被命令行修改过的alg_name）
    alg_manager = get_alg_manager(alg_name=cmd_args_first_pass.alg_name)
    args = update_args(
        args, os.path.join("configs", "alg", f"{cmd_args_first_pass.alg_name}.yaml")
    )

    # 默认的环境特殊配置文件就是环境名本身
    # 如果直接运行这个脚本，env_name 由 default_config.yaml设置，保存在 args.env_name
    # 如果Unity调用这个脚本，最原始的 env_name 由 Unity 传入, 保存在 cmd_args_first_pass.env_name
    # 并且 Unity 会设置 special_env_name 这个变量，表示最终需要加载的环境配置文件
    special_env_name = (  # 如果命令行中传入了 env_name, 则使用命令行中的 env_name
        args.env_name  # 这个是 default_config中设置的 env_name, 这个变量一定存在
        if not hasattr(cmd_args_first_pass, "env_name")
        else cmd_args_first_pass.env_name
    )

    if hasattr(
        cmd_args_first_pass, "special_env_name"
    ):  # 如果命令行传入了 special_env_name, 则使用 special_env_name的配置
        special_env_name = cmd_args_first_pass.special_env_name

    # 在这里正式加载环境的特殊配置
    # 比如 SatelliteEnv3v1HvtNoPlane这个环境，OnPlane=False会在这里设置
    # 但是这里运行的还是 SatelliteEnv3v1Hvt这个环境。
    special_env_args = Namespace()
    special_env_args = update_args(
        special_env_args,
        os.path.join("configs", "env_config", f"{special_env_name}.yaml"),
    )

    # 移除冗余参数
    for key in ["task_type", "env_name"]:
        if key in special_env_args.__dict__:
            del special_env_args.__dict__[key]

    # 5. 第二次读取命令行参数，确保命令行参数具有最高优先级
    args = load_args(args)

    # 6. 用最终的命令行参数替换special_env_args中的对应值
    # 这里是为了保险, 实际上Unity也会读取环境的特殊配置然后用命令行传入
    # 在实际调用这个脚本的时候
    # 也可以直接 传入 --on_plane False --env_name SatelliteEnv3v1Hvt 等价于调用 SatelliteEnv3v1HvtNoPlane
    for key in special_env_args.__dict__:
        if hasattr(args, key):
            setattr(special_env_args, key, getattr(args, key))

    # 更新超参数
    update_hyperparameter(args)

    log_env_name = (
        args.env_name
        if not hasattr(args, "special_env_name")
        else args.special_env_name
    )

    log_alg_name = (
        args.alg_name  # unity 会设置 special_alg_name
        if not hasattr(args, "special_alg_name")
        else args.special_alg_name
    )

    # 设置日志和模型保存路径
    args.logdir = str(
        os.path.join(
            args.logdir,
            log_env_name,
            log_alg_name,
            args.exp_name + "-" + time.strftime("%Y%m%d-%H%M%S"),
        )
    )

    args.models_dir = str(
        os.path.join(
            args.models_dir,
            log_env_name,
            log_alg_name,
            args.exp_name + "-" + time.strftime("%Y%m%d-%H%M%S"),
        )
    )

    # 规范路径
    args.logdir = str(os.path.normpath(args.logdir))
    args.models_dir = str(os.path.normpath(args.models_dir))

    # 创建模型保存目录
    if not args.debug_mode and not os.path.exists(args.models_dir):
        os.makedirs(args.models_dir)

    # 配置日志记录器
    logger_config = LoggerConfig(
        use_tensorboard=args.use_tensorboard,
        tensorboard_dir=args.logdir,
        debug_mode=args.debug_mode,
        log_file=args.logdir + ".log",
        train_reward_aggregate_mode=AggregateMode.MEAN,
        eval_reward_aggregate_mode=AggregateMode.MEAN,
        train_reward_log_interval=args.log_interval,
        eval_reward_log_interval=args.eval_episode,
    )

    logger = Logger(logger_config)
    args.logger = logger

    start_info()
    logger.info(f"开始实验: {args.exp_name}")
    logger.info(f"使用环境: {log_env_name}")
    logger.info(f"使用算法: {log_alg_name}")
    logger.info(f"随机种子: {args.seed}")
    logger.info(f"并行环境数量: {args.num_envs}")
    logger.info(f"使用奖励函数：{special_env_args.reward_computer_type}")
    logger.info(f"日志输出路径: {args.logdir}")
    logger.info(f"模型输出路径: {args.models_dir}")

    # 创建环境
    train_env = None
    env_kwargs = {
        "id": args.env_name,
        "render_mode": "human" if args.render else None,
        "update_interval": args.update_interval,
        "action_scale": args.action_scale,
        "collision_threshold": args.collision_threshold,
        "agent_heuristic": args.agent_heuristic,
        "opponent_heuristic": args.opponent_heuristic,
        "m_max_episode_steps": args.max_episode_steps,
        "reward_computer_type": special_env_args.reward_computer_type,
        "on_plane": args.on_plane,
        "simulator": SimulatorType.Local,
    }

    if hasattr(args, "env_config"):
        env_kwargs.update({"config_name": args.env_config})

    env_kwargs.update(vars(special_env_args))

    for key in (
        "target_sector_angle_deg",
        "target_sector_radius_km",
        "attacker_initial_rho_min_km",
        "attacker_initial_rho_max_km",
    ):
        if hasattr(args, key):
            env_kwargs[key] = getattr(args, key)

    # 创建评估环境
    eval_env = gym.make(**env_kwargs)

    # 创建训练环境（支持多环境并行）
    if args.num_envs > 1 and not args.just_eval:
        train_env = gym.make_vec(
            vectorization_mode=VectorizeMode.ASYNC,
            vector_kwargs={
                "autoreset_mode": gym.vector.AutoresetMode.SAME_STEP,
                "context": "spawn",
            },
            num_envs=args.num_envs,
            **env_kwargs,
        )
    else:
        train_env = eval_env

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 获取观测和动作空间信息
    obs_space = eval_env.observation_space
    action_space = eval_env.action_space

    # 设置全局状态标志
    if not hasattr(args, "use_global_state"):
        args.use_global_state = False

    global_state_shape = (
        0
        if not args.use_global_state
        else eval_env.unwrapped.global_state_space.shape[0]
    )

    # 设置智能体相关信息
    args.name_agents = list(obs_space.keys())
    args.num_agents = len(args.name_agents)
    args.n_actions = [action_space[name].shape[0] for name in args.name_agents]
    args.n_obs = [obs_space[name].shape[0] for name in args.name_agents]

    # 设置计算设备
    args.device = "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"

    # 创建算法实例
    alg = alg_manager(
        name_agents=args.name_agents,
        obs_shapes=args.n_obs,
        global_state_shape=global_state_shape,
        action_shapes=args.n_actions,
        args=args,
        action_continuous=args.continuous,
    )

    # 创建评估器
    eval = Evaluator(
        env=eval_env,
        name_agents=args.name_agents,
        alg=alg,
        args=args,
    )

    # 设置环境特定的日志回调
    logger_callback = None
    if args.env_name == "SatelliteEnv3v1":
        logger_callback = Satellite3v1LoggerCallback(logger, args)
    elif args.env_name == "SatelliteEnv3v1Hvt":
        logger_callback = Satellite3v1HvtLoggerCallback(logger, args)
    elif args.env_name == "SatelliteEnv2v2":
        logger_callback = Satellite2v2LoggerCallback(logger, args)
    else:
        logger.warning("Unimplemented logger_callback for env {}".format(args.env_name))

    # 执行训练或评估
    if args.just_eval:
        eval_func(eval, args, logger, logger_callback=logger_callback)
    else:
        assert train_env is not None
        train(
            args,
            alg,
            train_env,
            eval,
            logger,
            logger_callback=logger_callback,
        )

    # 清理资源
    if not args.just_eval and args.num_envs > 1:
        train_env.close()
    eval_env.close()
    logger.close()

    # 如果需要，在退出前暂停
    if should_pause_before_close(args):
        os.system("pause")


if __name__ == "__main__":
    """脚本入口点。

    注册环境并启动主函数。
    """
    # 注册自定义环境
    register_env()

    # 设置多进程启动方法
    mp.set_start_method("spawn", force=True)
    start_info()

    # 启动主函数
    main()
