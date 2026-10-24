import os
import gc
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from argparse import Namespace

from src.eval_internal.Env2v2 import Env2v2

# ----- 限制 CPU 线程数 -----
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# 已经正确导入 FAMAC 的 RNN Actor
from src.algorithms.Network.Actor.RNNDeterministicActor import (
    RNNDeterministicActor,
)
from src.environment.visualization.SimulatorFactory import SimulatorType

# ================= 共享网络参数配置 =================
task_type = "2v2"

model_path = {
    "2v2": r"D:\Ropes\Laboratory\803\PythonSatellite\output\models\SatelliteEnv2v2NoPlaneLimitAccel\FACMAC\v9_6_27_2_FAMAC_NoPlane-20260518-215412\actor_critic_mixer.pth"
}

num_episodes_per_pair = 100
device = torch.device("cpu")
seed = 8

# FAMAC 网络所需的额外参数
args = Namespace()
args.device = "cpu"
args.use_agent_id = True  # <--- 这行缺失导致了报错
args.hidden_size = 128

# 环境固定参数
env_kwargs = {
    "render_mode": "None",
    "update_interval": 2,
    "action_scale": 2 * 0.0005,
    "collision_threshold": 0.1,
    "agent_heuristic": False,
    "opponent_heuristic": True,
    "m_max_episode_steps": 2000,
    "on_plane": False,
    "simulator": SimulatorType.Local,
    "attacker_max_accel": 1.0,
    "limit_accel": True,
    "obs_add_last_action": True,
    "intercept_time": 50,
    "intercept_distance_threshold_km": 15.0,
    "win_max_angle": 30.0,
    "outside_threshold": 100,
}

# 扇区角度列表（度）
sector_angles = [30, 60, 90, 120, 150, 180]
# 扇区半径列表（km）
sector_radii = [20, 25, 30, 35]

CHUNK_SIZE = 10

# ================= 全局 worker 缓存 =================
_worker_actor = None
_worker_name_agents = None


def init_worker(model_path, obs_dim, action_dim, num_agents, name_agents):
    """子进程初始化：加载 FAMAC 模型"""
    global _worker_actor, _worker_name_agents
    _worker_name_agents = name_agents
    all_model = torch.load(model_path[task_type], map_location="cpu")

    # ========== 修改点 2：实例化 RNNDeterministicActor ==========
    actor_input_dim = obs_dim if not args.use_agent_id else obs_dim + num_agents

    actor = RNNDeterministicActor(
        input_size=actor_input_dim,
        output_size=action_dim,
        hidden_size=args.hidden_size,
    ).to(device)

    actor.load_state_dict(all_model["actor"])
    actor.eval()
    _worker_actor = actor
    torch.set_grad_enabled(False)


def run_episodes_chunk(angle, radius, seeds_chunk):
    """运行一包种子（多局），返回 (angle, radius, win_count, total_episodes)"""
    global _worker_actor, _worker_name_agents

    env = Env2v2(sector_angle_deg=angle, sector_radius_km=radius, **env_kwargs)
    num_agents = len(_worker_name_agents)

    wins = 0
    gc.disable()
    try:
        for ep_seed in seeds_chunk:
            np.random.seed(ep_seed)
            random.seed(ep_seed)
            torch.manual_seed(ep_seed)

            obs, _ = env.reset()
            done = False
            truncated = False

            # ========== 修改点 3：初始化 RNN 的隐藏状态 ==========
            hidden_state = None
            if args.use_agent_id:
                agent_ids_tensor = torch.eye(
                    num_agents, dtype=torch.float32, device=device
                )

            with torch.inference_mode():
                while not (done or truncated):
                    obs_list = [obs[name] for name in _worker_name_agents]
                    obs_tensor = torch.tensor(
                        np.array(obs_list), dtype=torch.float32
                    ).to(device)

                    # 拼接 Agent ID
                    if args.use_agent_id:
                        obs_tensor = torch.cat([obs_tensor, agent_ids_tensor], dim=-1)

                    # 重塑为 (num_agents, seq_len=1, input_dim) 供 RNN 使用
                    obs_tensor = obs_tensor.unsqueeze(1)

                    # 前向推理并更新 hidden_state
                    actions_tensor, hidden_state = _worker_actor(
                        obs_tensor, hidden_state
                    )

                    # 剥离 seq_len 维度
                    actions_tensor = actions_tensor.squeeze(1).cpu().detach().numpy()

                    actions = {}
                    for idx, name in enumerate(_worker_name_agents):
                        actions[name] = actions_tensor[idx]

                    obs, _, done, truncated, info = env.step(actions)

            # TODO 3: 2v2 胜利条件判定！
            if info.get("all_intercept_claimed", False):
                wins += 1
    finally:
        gc.enable()
        env.close()

    return angle, radius, wins, len(seeds_chunk)


def get_env_shapes():
    """获取观测/动作空间维度（单个智能体）"""
    temp_env = Env2v2(sector_angle_deg=360, sector_radius_km=30, **env_kwargs)
    obs_space = temp_env.observation_space
    action_space = temp_env.action_space
    name_agents = list(obs_space.keys())

    obs_dim = obs_space[name_agents[0]].shape[0]
    action_dim = action_space[name_agents[0]].shape[0]
    temp_env.close()
    return obs_dim, action_dim, len(name_agents), name_agents


def main():
    print(">>> 提取 2v2 环境空间维度...")
    obs_dim, action_dim, num_agents, name_agents = get_env_shapes()
    print(f"  智能体: {name_agents}")
    print(f"  观测维度: {obs_dim}")
    print(f"  动作维度: {action_dim}")

    base_seed = 10000
    tasks = []
    for angle in sector_angles:
        for radius in sector_radii:
            all_seeds = [
                base_seed + hash((angle, radius, i)) % 1000000
                for i in range(num_episodes_per_pair)
            ]
            for i in range(0, num_episodes_per_pair, CHUNK_SIZE):
                chunk_seeds = all_seeds[i : i + CHUNK_SIZE]
                tasks.append((angle, radius, chunk_seeds))

    total_episodes = sum(len(chunk) for _, _, chunk in tasks)
    print(f">>> 总测试局数: {total_episodes}, 切分为 {len(tasks)} 个计算块")
    print(f">>> 使用 CPU 核心数: {mp.cpu_count() - 1} (保留一个给系统)")

    win_counts = {(a, r): 0 for a in sector_angles for r in sector_radii}
    total_counts = {(a, r): 0 for a in sector_angles for r in sector_radii}

    with ProcessPoolExecutor(
        max_workers=max(1, mp.cpu_count() - 1),
        initializer=init_worker,
        initargs=(model_path, obs_dim, action_dim, num_agents, name_agents),
    ) as executor:
        futures = [
            executor.submit(run_episodes_chunk, angle, radius, seeds)
            for angle, radius, seeds in tasks
        ]
        with tqdm(total=total_episodes, desc="测试进度", unit="ep") as pbar:
            for future in as_completed(futures):
                angle, radius, wins, total = future.result()
                win_counts[(angle, radius)] += wins
                total_counts[(angle, radius)] += total
                pbar.update(total)

    win_rate_grid = np.zeros((len(sector_radii), len(sector_angles)))
    for i, radius in enumerate(sector_radii):
        for j, angle in enumerate(sector_angles):
            total = total_counts[(angle, radius)]
            rate = win_counts[(angle, radius)] / total if total > 0 else 0
            win_rate_grid[i, j] = rate
            print(
                f"角度 {angle:3f}° | 半径 {radius:3f} km : 胜场 {win_counts[(angle, radius)]:3f}/{total:3f} , 胜率 = {rate:.3f}"
            )

    # ========== 绘图部分 ==========

    # 1. 原始3D曲面图
    X, Y = np.meshgrid(sector_angles, sector_radii)
    Z = win_rate_grid

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.9)
    ax.set_xlabel("Sector Angle (deg)")
    ax.set_ylabel("Sector Radius (km)")
    ax.set_zlabel("Win Rate")
    ax.set_title("2v2 Interceptor Win Rate vs Initial Sector (FAMAC)")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    plt.tight_layout()
    plt.savefig("win_rate_3d_sector_famac_2v2.png", dpi=150)
    plt.close()

    # 3. 等高线图
    fig, ax = plt.subplots(figsize=(10, 6))
    contour = ax.contour(X, Y, Z, levels=10, cmap="plasma", linewidths=1.5)
    ax.clabel(contour, inline=True, fontsize=8)
    ax.set_xlabel("Sector Angle (deg)")
    ax.set_ylabel("Sector Radius (km)")
    ax.set_title("2v2 Win Rate Contour")
    ax.contourf(X, Y, Z, levels=20, cmap="viridis", alpha=0.6)
    plt.tight_layout()
    plt.savefig("win_rate_contour_2v2.png", dpi=150)
    plt.close()

    # 4. 参数关系折线图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for i, radius in enumerate(sector_radii):
        ax1.plot(sector_angles, win_rate_grid[i, :], marker="o", label=f"R={radius} km")
    ax1.set_xlabel("Sector Angle (deg)")
    ax1.set_ylabel("Win Rate")
    ax1.set_title("Win Rate vs Angle (by Radius)")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    for j, angle in enumerate(sector_angles):
        ax2.plot(sector_radii, win_rate_grid[:, j], marker="s", label=f"θ={angle}°")
    ax2.set_xlabel("Sector Radius (km)")
    ax2.set_ylabel("Win Rate")
    ax2.set_title("Win Rate vs Radius (by Angle)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="best", ncol=2)
    plt.tight_layout()
    plt.savefig("win_rate_lineplot_2v2.png", dpi=150)
    plt.close()

    print("所有 2v2 评估图表已生成完毕！")


if __name__ == "__main__":
    mp.freeze_support()
    main()
