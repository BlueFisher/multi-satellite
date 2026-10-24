import atexit
import os
from argparse import Namespace

import gymnasium as gym
import numpy as np

from Evaluation import Evaluator
from algorithms import Manager
from environment.SatelliteEnvBase import SatelliteEnvBase
from eval_alg import eval_func
from utils.Logger import Logger
from utils.Logger.LoggerCallback import BaseLoggerCallback
from utils.Logger.LoggerConfig import LoggerMode


def train(
    args: Namespace,
    alg: Manager,
    env,
    eval: Evaluator,
    logger: Logger,
    logger_callback: BaseLoggerCallback = None,
):
    """
    启动并协调训练（单环境或并行环境）

    该函数会验证环境数量，保存当前模型到磁盘，创建训练期间的锁文件，
    并根据 ``args.num_envs`` 的值调用 :func:`train_single` 或 :func:`train_parallel`。

    参数：
        args (Namespace): 运行时参数和超参数。
            包括但不限于：
                - ``num_envs`` (int): 并行环境数量。
                - ``models_dir`` (str): 模型保存目录。
                - ``n_episodes`` (int): 训练的总回合数。
                - ``log_interval`` (int), ``save_interval`` (int), ``eval_interval`` (int): 日志记录和检查点保存的间隔。
                - ``render`` (bool): 是否渲染环境。
        alg (Manager): 算法管理器，负责动作选择、经验存储、训练和模型保存。
        env: Gym 环境实例。单环境使用 :class:`SatelliteEnvBase` 或其未包装实例。
            并行训练时传入 ``gym.vector.AsyncVectorEnv``。
        eval (Evaluator): 评估器实例，用于对当前策略进行周期性评估。
        logger (Logger): 日志记录器实例，用于记录训练期间的指标和回合奖励。
        logger_callback (BaseLoggerCallback, 可选): 可选的回调函数，用于处理回合结束或训练日志间隔等事件。默认为 None。

    异常：
        ValueError: 如果 ``args.num_envs`` 小于 1。

    副作用：
        - 在训练开始前将模型保存到 ``args.models_dir``。
        - 在 ``args.models_dir`` 中创建 ``.lock`` 文件，并确保在进程退出时删除。
        - 调用 :func:`train_single` 或 :func:`train_parallel`，运行训练循环并执行日志记录、评估和周期性模型保存。

    返回：
        None: 该函数用于协调训练，不返回值；通过副作用记录进度并保存模型。
    """

    if args.num_envs < 1:
        raise ValueError("num_envs must be greater than 0")

    if not args.debug_mode:
        alg.save_model(args.models_dir)

        open(os.path.join(args.models_dir, ".lock"), "w").close()
        atexit.register(lambda: os.remove(os.path.join(args.models_dir, ".lock")))

    logger.info("start train with args", format=True, **vars(args))

    if args.num_envs == 1:
        train_single(args, alg, env, eval, logger, logger_callback)
        return
    train_parallel(args, alg, env, eval, logger, logger_callback)
    logger.info("模型训练完成")


def run_episode(
    env: SatelliteEnvBase,
    alg: Manager,
    args: Namespace,
    test=False,
):
    """
    执行单个回合的训练或测试。

    该函数负责在单环境中运行一个完整的回合，包括动作选择、环境交互、
    数据存储和模型训练（如果未处于测试模式）。

    参数：
        env (SatelliteEnvBase): 卫星环境实例，必须是单环境。
        alg (Manager): 算法管理器，负责动作选择、经验存储和训练。
        args (Namespace): 运行时参数和超参数。
            包括但不限于：
                - ``num_envs`` (int): 并行环境数量，必须为 1。
                - ``num_agents`` (int): 智能体数量。
                - ``n_actions`` (list[int]): 每个智能体的动作维度。
                - ``max_episode_steps`` (int): 每个回合的最大步数。
                - ``use_global_state`` (bool): 是否使用全局状态。
                - ``render`` (bool): 是否渲染环境。
                - ``train_after_episode`` (bool): 是否在回合结束后训练模型。
        test (bool, 可选): 是否处于测试模式。默认为 False。

    返回：
        tuple:
            - loss (dict): 每个损失项的平均值。
            - episode_rewards (dict): 每个智能体在当前回合的奖励。
            - info (dict): 环境返回的额外信息。

    异常：
        AssertionError: 如果 ``args.num_envs`` 不等于 1。

    副作用：
        - 在非测试模式下存储经验数据。
        - 在非测试模式下可能触发模型训练。
    """

    assert args.num_envs == 1

    num_agents = args.num_agents
    num_actions = args.n_actions[0]
    num_envs = args.num_envs

    observations, info = env.reset()
    global_state = env.get_global_state() if args.use_global_state else None

    alg.before_episode(0)

    episode_rewards = {agent: [] for agent in args.name_agents}
    obs_np = np.array([observations[agent] for agent in args.name_agents]).reshape(
        num_agents, num_envs, -1
    )

    loss = {}
    train_counter = 0

    def _train():
        if not alg.can_train():
            return False
        t_loss = alg.train()
        for key, value in t_loss.items():
            if key in loss:
                loss[key] += value
            else:
                loss[key] = value
        return True

    extra_data = {}
    if args.use_global_state:
        extra_data["global_state"] = global_state

    for step in range(args.max_episode_steps):
        actions = {}
        choose_ret = alg.choose_action(
            args.name_agents,
            obs=obs_np,
            global_state=global_state,
        )

        for idx, agent in enumerate(args.name_agents):
            actions[agent] = choose_ret["action"][idx].reshape(num_actions)

        for key, value in choose_ret.items():
            if key != "action":
                extra_data[key] = value

        # 环境执行一步
        next_observations, rewards, terminations, truncations, info = env.step(actions)

        next_global_state = env.get_global_state() if args.use_global_state else None

        if args.render:
            env.render()

        # 转换数据格式
        next_obs_np = np.array(
            [next_observations[agent] for agent in args.name_agents]
        ).reshape(num_agents, num_envs, -1)

        reward_np = np.array(rewards).reshape(num_envs, num_agents)

        done_np = np.logical_or([terminations], [terminations])
        store_done_np = np.repeat(done_np, num_agents)  # (num_agents, )
        store_done_np = np.expand_dims(store_done_np, axis=0)  # (num_envs, n_agents)
        break_done = all(done_np)

        actions_np = np.array([actions[agent] for agent in args.name_agents]).reshape(
            num_agents, num_envs, -1
        )

        if not test:
            # 存储经验
            experience_kwargs = {
                "obs": obs_np,
                "actions": actions_np,
                "rewards": reward_np,
                "next_obs": next_obs_np,
                "dones": store_done_np,
            }

            # 有额外数据时添加到参数字典
            if len(extra_data) > 0:
                experience_kwargs["extra_data"] = extra_data

            # 统一调用方法
            alg.store_experience(**experience_kwargs)

        # 更新观测和奖励
        obs_np = next_obs_np
        global_state = next_global_state
        for idx, agent in enumerate(args.name_agents):
            episode_rewards[agent].append(rewards[idx])

        if not test and not args.train_after_episode:
            if _train():
                train_counter += 1

        # 检查是否结束
        if break_done:
            break

    alg.after_episode(0)

    if not test and args.train_after_episode:
        if _train():
            train_counter += 1

    for key, _ in loss.items():
        loss[key] /= train_counter

    return loss, episode_rewards, info


def train_single(
    args: Namespace,
    alg: Manager,
    env: gym.Env,
    eval: Evaluator,
    logger: Logger,
    logger_callback: BaseLoggerCallback = None,
):
    """
    在单环境中训练智能体。

    该函数负责在单环境中运行完整的训练过程，包括回合循环、
    数据存储、模型训练、日志记录和模型保存。

    参数：
        args (Namespace): 运行时参数和超参数。
            包括但不限于：
                - ``n_episodes`` (int): 训练的总回合数。
                - ``log_interval`` (int): 日志记录的间隔。
                - ``save_interval`` (int): 模型保存的间隔。
                - ``eval_interval`` (int): 评估的间隔。
                - ``debug_mode`` (bool): 是否启用调试模式。
        alg (Manager): 算法管理器，负责动作选择、经验存储、训练和模型保存。
        env (gym.Env): Gym 环境实例，必须是单环境。
        eval (Evaluator): 评估器实例，用于对当前策略进行周期性评估。
        logger (Logger): 日志记录器实例，用于记录训练期间的指标和回合奖励。
        logger_callback (BaseLoggerCallback, 可选): 可选的回调函数，用于处理回合结束或训练日志间隔等事件。默认为 None。

    异常：
        AssertionError: 如果 ``env`` 不是 ``SatelliteEnvBase`` 的实例。

    副作用：
        - 在每个回合结束时记录日志。
        - 在指定间隔保存模型。
        - 在指定间隔运行评估。

    返回：
        None: 该函数用于协调单环境训练，不返回值；通过副作用记录进度并保存模型。
    """

    assert isinstance(
        env.unwrapped, SatelliteEnvBase
    ), "The environment must be a SatelliteEnvBase when nums_envs equals 1"

    logger.set_mode(LoggerMode.TRAIN)

    total_loss = {}
    loss_counter = 0
    for episode in range(args.n_episodes):
        loss, episode_rewards, info = run_episode(env.unwrapped, alg, args)

        if len(loss.keys()) > 0:
            loss_counter += 1
            for key, value in loss.items():
                total_loss[key] = total_loss.get(key, 0) + value

        if len(loss) > 0:
            logger.info(f"Train loss at episode {episode}", **loss)
        if logger_callback is not None:
            logger_callback.after_episode(episode_rewards, info)

        logger.add_episode_rewards(episode_rewards=episode_rewards)
        logger.step()

        if (episode + 1) % args.log_interval == 0:
            for key, value in total_loss.items():
                logger.log_metric(
                    key, value / loss_counter, episode, tags={"group": "Loss"}
                )
            total_loss = {}
            loss_counter = 0

            if logger_callback is not None:
                logger_callback.on_train_log_interval(episode)

        if (episode + 1) % args.save_interval == 0 and not args.debug_mode:
            alg.save_model(os.path.join(args.models_dir, str(episode + 1)))

        if (episode + 1) % args.eval_interval == 0:
            eval_func(eval, args, logger)


def train_parallel(
    args: Namespace,
    alg: Manager,
    env: gym.vector.AsyncVectorEnv,
    evaluator: Evaluator,
    logger: Logger,
    logger_callback: BaseLoggerCallback = None,
):
    """
    在并行环境中训练智能体。

    该函数利用 Gym 的并行环境特性，运行多个环境实例以加速训练过程。
    它负责管理并行环境的初始化、数据收集、模型训练、日志记录和模型保存。

    参数：
        args (Namespace): 运行时参数和超参数。
            包括但不限于：
                - ``n_episodes`` (int): 训练的总回合数。
                - ``log_interval`` (int): 日志记录的间隔。
                - ``save_interval`` (int): 模型保存的间隔。
                - ``eval_interval`` (int): 评估的间隔。
                - ``debug_mode`` (bool): 是否启用调试模式。
                - ``render`` (bool): 是否渲染环境。
        alg (Manager): 算法管理器，负责动作选择、经验存储、训练和模型保存。
        env (gym.vector.AsyncVectorEnv): 并行环境实例。
        evaluator (Evaluator): 评估器实例，用于对当前策略进行周期性评估。
        logger (Logger): 日志记录器实例，用于记录训练期间的指标和回合奖励。
        logger_callback (BaseLoggerCallback, 可选): 可选的回调函数，用于处理回合结束或训练日志间隔等事件。默认为 None。

    异常：
        AssertionError: 如果环境初始化失败或参数无效。

    副作用：
        - 在每个回合结束时记录日志。
        - 在指定间隔保存模型。
        - 在指定间隔运行评估。
        - 渲染环境（如果启用）。

    返回：
        None: 该函数用于协调并行环境训练，不返回值；通过副作用记录进度并保存模型。
    """

    logger.set_mode(LoggerMode.TRAIN)
    # 初始化环境
    observations, info = env.reset(
        seed=args.seed + 1
    )  # 这里只需要设置一个 seed, gym 会自动递增seed
    num_envs = env.num_envs

    # 跟踪每个环境的累积奖励
    episode_rewards = {
        env_idx: {name: [] for name in args.name_agents} for env_idx in range(num_envs)
    }
    # 跟踪每个环境当前的episode步数
    episode_steps = {env_idx: 0 for env_idx in range(num_envs)}
    # 记录已完成的episode总数
    total_completed_episodes = 0

    # 处理观测值
    obs_list = [observations[agent] for agent in args.name_agents]

    total_loss = {}
    loss_counter = 0
    [alg.before_episode(env_idx) for env_idx in range(num_envs)]
    # 持续训练直到达到目标episode数量
    while total_completed_episodes < args.n_episodes:
        # 获取全局状态（从info中）
        global_states = info.get("global_state", [None] * num_envs)

        # 准备额外数据
        extra_data = {}
        if hasattr(args, "use_global_states") and args.use_global_states:
            extra_data["global_state"] = global_states

        # 选择动作
        choose_ret = alg.choose_action(
            args.name_agents,
            obs=np.array(obs_list),
            global_state=global_states,
        )

        # 构建动作字典
        actions = {
            agent: choose_ret["action"][idx]
            for idx, agent in enumerate(args.name_agents)
        }

        # 添加额外返回数据
        for key, value in choose_ret.items():
            if key != "action":
                extra_data[key] = value

        # 环境执行一步
        next_observations, rewards, terminations, truncations, info = env.step(actions)

        # 渲染（如果需要）
        if args.render:
            env.render()

        # 处理下一个观测值（保持为reset后的新观测）
        next_obs_list = [next_observations[agent] for agent in args.name_agents]
        next_obs_list_scaled = next_obs_list

        # 用于存储经验的next_obs（初始为环境返回的观测）
        experience_next_obs = [obs.copy() for obs in next_obs_list_scaled]

        # 检查每个环境是否完成当前episode
        dones = np.logical_or(terminations, truncations)

        # 处理每个环境的奖励和完成状态
        for env_idx in range(num_envs):
            # 更新当前episode的奖励和步数
            for idx, agent in enumerate(args.name_agents):
                episode_rewards[env_idx][agent].append(rewards[env_idx][idx])
            episode_steps[env_idx] += 1

            # 检查是否达到最大步数或环境已结束
            if dones[env_idx] or episode_steps[env_idx] >= args.max_episode_steps:
                # 记录完成的episode
                total_completed_episodes += 1
                episode_idx = total_completed_episodes

                # 获取真正的终止状态观测值（非重置后的新观测）
                actual_next_obs = info["final_obs"][env_idx]

                # 将终止时的真实观测值用于经验存储
                for idx, agent in enumerate(args.name_agents):
                    # 处理单个智能体的终止观测
                    agent_actual_obs = actual_next_obs[agent]
                    # 确保形状匹配
                    if agent_actual_obs.ndim == 1:
                        agent_actual_obs = agent_actual_obs.reshape(1, -1)
                    # 更新经验中的next_obs为终止观测
                    experience_next_obs[idx][env_idx] = agent_actual_obs.flatten()

                actual_info = info["final_info"]

                # 调用算法的episode前后处理
                alg.after_episode(env_idx)
                logger.add_episode_rewards(episode_rewards=episode_rewards[env_idx])
                logger.step()

                if logger_callback is not None:
                    env_info = {k: v[env_idx] for k, v in actual_info.items()}
                    logger_callback.after_episode(episode_rewards[env_idx], env_info)

                # 日志输出
                if episode_idx % args.log_interval == 0:
                    for key, value in total_loss.items():
                        logger.log_metric(
                            key,
                            value / loss_counter,
                            episode_idx,
                            tags={"group": "Loss"},
                        )
                    total_loss = {}
                    loss_counter = 0

                    if logger_callback is not None:
                        logger_callback.on_train_log_interval(episode_idx)

                alg.before_episode(env_idx)

                # 重置该环境的跟踪信息
                episode_rewards[env_idx] = {name: [] for name in args.name_agents}
                episode_steps[env_idx] = 0

                # 检查是否需要保存模型
                if episode_idx % args.save_interval == 0 and not args.debug_mode:
                    alg.save_model(args.models_dir)

                # 检查是否需要评估
                if episode_idx % args.eval_interval == 0:
                    eval_func(evaluator, args, logger)

        num_agents = len(args.name_agents)
        # 将 (num_envs,) 转换为 (num_envs, num_agents)
        store_dones = np.repeat(terminations[:, np.newaxis], num_agents, axis=1)
        # 非测试模式下存储经验（使用修正后的next_obs）
        experience_kwargs = {
            "obs": np.array(obs_list),
            "actions": np.array([actions[agent] for agent in args.name_agents]),
            "rewards": rewards,
            "next_obs": np.array(experience_next_obs),  # 使用包含终止观测的版本
            "dones": store_dones,
        }
        if len(extra_data) > 0:
            experience_kwargs["extra_data"] = extra_data

        alg.store_experience(**experience_kwargs)

        look_train = any(dones) or not args.train_after_episode
        if look_train and alg.can_train():
            t_loss = alg.train()
            for key, value in t_loss.items():
                total_loss[key] = total_loss.get(key, 0.0) + value
            loss_counter += 1

        # 更新观测值（使用环境返回的reset后新观测，保持训练流程正确）
        obs_list = next_obs_list_scaled
    env.close()
