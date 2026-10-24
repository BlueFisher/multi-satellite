"""
！！！ 注意 ！！！
这个只是一个内部快速评估模型的脚本，不需要启动Unity, 快速的对模型进行测试。实际脚本中很多内容和环境强耦合
如果需要使用该脚本对模型进行测试，请仔细阅读代码并针对需要测试的环境做出修改
如果不想修改代码，Unity也集成了测试功能（Unity 会调用 cmd_show_sys.py这个脚本，该脚本对所有环境做出了适配），请使用Unity进行测试
"""

import os
import gc

# ！！！极其重要：切断 PyTorch 的多线程，防止多个子进程互相争抢 CPU 导致死锁和性能断崖
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import random
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
from algorithms.Network.Actor.IntentPredictActor import IntentDeterministicActor
from environment import SimulatorType
from algorithms.Network.Actor.SharedMessageDeterministicActor import (
    SharedMessageDeterministicActor,
)
from argparse import Namespace


try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ================== 压测全局配置 ==================
TASK_TYPE = "3v1"
MODEL_PATH = r"D:\Ropes\Laboratory\803\PythonSatellite\output\logs\SatelliteEnv3v1HvtNoPlane\SharedMessageMATD3\SharedMessageMATD3-NoPlane-Collision0.1c100-20260325-140328\actor_critic.pth"
# ！！！将每个参数条件下的测试局数拉满！！！
NUM_TOTAL_EPISODES = 500
# 将这些局数切片分配给多个 CPU 核心（例如每个核心一次领 10 局任务，跑完再领，完美负载均衡）
CHUNK_SIZE = 10

args = Namespace()
args.use_relation = True
args.use_ndq = True
args.use_message_gate = True
args.device = "cpu"  # 压测使用 CPU 更稳定
args.relation_topk = 0.4


DEFAULT_PARAMS = {
    "update_interval": 2,
    "attacker_max_accel": 5 / 6,
    "m_max_episode_steps": 2000,
    "collision_threshold": 0.1,
    "capture_sustain_steps": 400,
}

# 庞大的参数矩阵
PARAM_VALUES = {
    # "update_interval": [1, 2, 3, 4, 5],
    # "attacker_max_accel": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    # "m_max_episode_steps": [1000, 1500, 2000, 2500, 3000],
    # "collision_threshold": [0, 0.5, 1.0, 1.5],
    "capture_sustain_steps": [10, 50, 100, 200, 300, 400, 500, 600],
}

# 子进程全局缓存
WORKER_ACTORS = None
WORKER_NAME_AGENTS = None

# ==================================================


def init_worker_process(model_path, n_obs, n_actions, num_agents, name_agents):
    global WORKER_ACTORS, WORKER_NAME_AGENTS
    WORKER_NAME_AGENTS = name_agents

    device = torch.device("cpu")
    all_model = torch.load(model_path, map_location=device)
    torch.set_grad_enabled(False)

    # 单个 actor 处理所有 agent
    actor = SharedMessageDeterministicActor(
        obs_dim=n_obs[0],  # 单个 agent 的观测维度
        action_dim=n_actions[0],  # 单个 agent 的动作维度
        hidden_dim=128,
        num_agents=num_agents,
        args=args,
    ).to(device)
    actor.load_state_dict(all_model["actor"])
    actor.eval()
    WORKER_ACTORS = [actor]  # 仍然用列表，但只包含一个元素


def run_episodes_chunk(task_config):
    """
    核心运行循环：一个进程领取一个 Chunk（如 10 局）跑完返回结果。
    """
    param_name = task_config["param_name"]
    param_value = task_config["param_value"]
    seeds_chunk = task_config["seeds"]

    env_kwargs = {
        "render_mode": None,
        "update_interval": DEFAULT_PARAMS["update_interval"],
        "action_scale": DEFAULT_PARAMS["update_interval"] * 0.0005,
        "collision_threshold": DEFAULT_PARAMS["collision_threshold"],
        "agent_heuristic": False,
        "opponent_heuristic": True,
        "m_max_episode_steps": DEFAULT_PARAMS["m_max_episode_steps"],
        "on_plane": False,
        "simulator": SimulatorType.Local,
        "limit_accel": True,
        "use_diff_obs": True,
        "obs_add_distance": False,
        "obs_compatible": False,
        "obs_window_size": 2,
        "action_window_size": 1,
        "attacker_max_accel": DEFAULT_PARAMS["attacker_max_accel"],
        "capture_sustain_steps": DEFAULT_PARAMS["capture_sustain_steps"],
    }

    env_kwargs[param_name] = param_value
    if param_name == "update_interval":
        env_kwargs["action_scale"] = param_value * 0.0005

    from src.environment.SatelliteEnv3v1Hvt import SatelliteEnv3v1Hvt

    env = SatelliteEnv3v1Hvt(**env_kwargs)

    wins = 0

    # 【极限优化2】在紧凑计算环节临时关闭垃圾回收，消除环境 Step 的微小卡顿
    gc.disable()

    try:
        agents = WORKER_NAME_AGENTS
        actor = WORKER_ACTORS[0]  # 取唯一 actor

        for seed in seeds_chunk:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)

            observations, info = env.reset(seed=seed)  # 假设支持 seed

            while True:
                # 收集所有 agent 的观测
                obs_list = [observations[name] for name in agents]
                # 拼接成 (1, num_agents, obs_dim) 张量
                obs_tensor = torch.tensor(obs_list, dtype=torch.float32).unsqueeze(0)
                # 注意：如果模型期望输入在 CPU 上，无需显式 .to(device)
                # 但 actor 在初始化时已经 .to(device)，这里需要确保输入也在同一设备
                obs_tensor = obs_tensor.to(actor.device)  # 或使用 .to('cpu')

                # 前向推理，取第一个返回值（动作）
                actions_tensor, _ = actor(
                    obs_tensor
                )  # shape: (1, num_agents, action_dim)
                actions_tensor = (
                    actions_tensor.squeeze(0).cpu().detach()
                )  # shape: (num_agents, action_dim)

                # 转换为字典动作
                actions = {}
                for idx, name in enumerate(agents):
                    actions[name] = actions_tensor[idx].numpy()

                next_observations, rewards, done, truncated, info = env.step(actions)
                observations = next_observations

                if done or truncated:
                    # 判断胜利条件（以第一个脚本中的 info.get("capture_reward_claimed") 为准）
                    if info.get("capture_reward_claimed", False):
                        wins += 1
                    break
    finally:
        gc.enable()
        env.close()

    return (param_name, param_value, wins, len(seeds_chunk))


def get_environment_shapes():
    """提取网络维度信息"""
    from src.environment.SatelliteEnv3v1Hvt import SatelliteEnv3v1Hvt

    dummy_kwargs = DEFAULT_PARAMS.copy()
    dummy_kwargs.update(
        {
            "render_mode": None,
            "action_scale": 0.001,
            "agent_heuristic": False,
            "opponent_heuristic": True,
            "m_max_episode_steps": 10,
            "on_plane": False,
            "simulator": SimulatorType.Local,
            "limit_accel": True,
            "use_diff_obs": True,
            "obs_add_distance": False,
            "obs_window_size": 2,
            "action_window_size": 1,
        }
    )
    env = SatelliteEnv3v1Hvt(**dummy_kwargs)
    obs_space = env.observation_space
    action_space = env.action_space

    name_agents = list(obs_space.keys())
    num_agents = len(name_agents)
    n_actions = [action_space[name].shape[0] for name in name_agents]
    n_obs = [obs_space[name].shape[0] for name in name_agents]
    env.close()
    return n_obs, n_actions, num_agents, name_agents


def plot_results(param_name, values, wins, totals):
    """绘制结果图表（针对二项分布采用正确置信区间）"""
    means, lowers, uppers = [], [], []

    # 计算胜率及其 95% 置信区间 (基于二项分布的正态近似计算法)
    for w, t in zip(wins, totals):
        p = w / t if t > 0 else 0
        means.append(p)
        # 1.96 代表 95% CI
        margin = 1.96 * np.sqrt((p * (1 - p)) / t) if t > 0 else 0
        lowers.append(max(0.0, p - margin))
        uppers.append(min(1.0, p + margin))

    plt.figure(figsize=(8, 5))
    if param_name == "on_plane":
        x = [str(v) for v in values]
        yerr_lower = [means[i] - lowers[i] for i in range(len(means))]
        yerr_upper = [uppers[i] - means[i] for i in range(len(means))]
        plt.bar(
            x,
            means,
            yerr=[yerr_lower, yerr_upper],
            capsize=5,
            alpha=0.7,
            error_kw={"elinewidth": 2, "ecolor": "black"},
        )
    else:
        plt.errorbar(
            values,
            means,
            yerr=[
                np.array(means) - np.array(lowers),
                np.array(uppers) - np.array(means),
            ],
            fmt="o-",
            capsize=5,
            capthick=2,
            elinewidth=2,
            markersize=8,
            label="Mean ± 95% CI",
        )
        plt.grid(True, linestyle="--", alpha=0.6)

    plt.xlabel(param_name)
    plt.ylabel(f"Win Rate ({NUM_TOTAL_EPISODES} Episodes)")
    plt.ylim(0, 1.05)
    plt.title(f"Effect of {param_name} on Win Rate")
    plt.tight_layout()

    os.makedirs("../img/plots", exist_ok=True)
    plt.savefig(f"../img/plots/{param_name}_effect.png", dpi=150)
    plt.close()


def main():
    print(">>> 正在提取环境空间维度...")
    n_obs, n_actions, num_agents, name_agents = get_environment_shapes()

    tasks = []
    base_seed = 10000

    # 动态构建切割任务 (Chunking)
    for param_name, values in PARAM_VALUES.items():
        for pv in values:
            all_seeds = [
                base_seed + hash((param_name, pv, i)) % 1000000
                for i in range(NUM_TOTAL_EPISODES)
            ]
            # 切分为小块，每块 CHUNK_SIZE 局，充分利用多核
            for i in range(0, NUM_TOTAL_EPISODES, CHUNK_SIZE):
                chunk_seeds = all_seeds[i : i + CHUNK_SIZE]
                tasks.append(
                    {"param_name": param_name, "param_value": pv, "seeds": chunk_seeds}
                )

    total_tasks = len(tasks)
    total_episodes_to_run = sum(len(t["seeds"]) for t in tasks)
    print(
        f">>> 任务生成完毕：共计 {total_episodes_to_run} 局评估，被切割为 {total_tasks} 个计算块。"
    )
    print(f">>> 启动多核极速压测 (分配核心数: {mp.cpu_count()}) ...")

    # 结果容器
    results_wins = {param: {v: 0 for v in vals} for param, vals in PARAM_VALUES.items()}
    results_totals = {
        param: {v: 0 for v in vals} for param, vals in PARAM_VALUES.items()
    }

    # 留一个核给操作系统，防止电脑完全卡死
    max_workers = max(1, mp.cpu_count() - 1)

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_worker_process,
        initargs=(MODEL_PATH, n_obs, n_actions, num_agents, name_agents),
    ) as executor:

        futures = [executor.submit(run_episodes_chunk, t) for t in tasks]

        # 进度条按 Episode 数目更新，看起来更直观
        with tqdm(
            total=total_episodes_to_run, desc="Episodes Computed", unit="ep"
        ) as pbar:
            for future in as_completed(futures):
                try:
                    p_name, p_val, wins, total_eps = future.result()
                    results_wins[p_name][p_val] += wins
                    results_totals[p_name][p_val] += total_eps
                    pbar.update(total_eps)
                except Exception as e:
                    print(f"\n[错误] 计算块执行崩溃: {e}")

    # 绘图输出
    print("\n>>> 开始生成统计图表...")
    for param_name, values in PARAM_VALUES.items():
        wins_list = [results_wins[param_name][v] for v in values]
        totals_list = [results_totals[param_name][v] for v in values]
        plot_results(param_name, values, wins_list, totals_list)
        print(f"  ✓ {param_name} 处理完毕并已保存图像。")

    print(
        f"\n>>> 压测全部完成！共耗尽 {total_episodes_to_run} 局模拟。图表存储于 ./plots/ 目录下。"
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()
