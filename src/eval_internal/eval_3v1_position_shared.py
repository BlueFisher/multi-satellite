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

from eval_internal.Env3v1 import Env3v1

# ----- 限制 CPU 线程数 -----
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from algorithms.Network.Actor.SharedMessageDeterministicActor import (
    SharedMessageDeterministicActor,
)
from environment.visualization.SimulatorFactory import SimulatorType

# ================= 共享网络参数配置 =================
task_type = "3v1"
model_path = {
    "3v1": r"D:\Ropes\Laboratory\803\PythonSatellite\output\models\SatelliteEnv3v1HvtNoPlaneLimitAccel\SharedMessageMATD3\SharedMessageMATD3-c200-no-gate-20260406-122807\actor_critic.pth"
}

num_episodes_per_pair = 100
device = torch.device("cpu")
seed = 8

# 共享网络所需的额外参数（与训练时保持一致）
args = Namespace()
args.use_relation = True
args.use_ndq = True
args.use_message_gate = False
args.device = "cpu"
args.relation_topk = 0.4

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
    "use_diff_obs": True,
    "obs_add_distance": False,
    "obs_window_size": 2,
    "action_window_size": 1,
    "attacker_max_accel": 0.8,
    "capture_sustain_steps": 150,
    "limit_accel": True,
    "obs_compatible": False,
}

# 扇区角度列表（度）
sector_angles = [i for i in range(80, 100, 2)]
# sector_angles = [i for i in range(30, 50, 5)]
# 扇区半径列表（km）
sector_radii = [i for i in range(24, 28, 1)]

CHUNK_SIZE = 10

# ================= 全局 worker 缓存 =================
_worker_actor = None
_worker_name_agents = None


def init_worker(model_path, obs_dim, action_dim, num_agents, name_agents):
    """子进程初始化：加载共享模型"""
    global _worker_actor, _worker_name_agents
    _worker_name_agents = name_agents
    all_model = torch.load(model_path[task_type], map_location="cpu")
    actor = SharedMessageDeterministicActor(
        obs_dim=obs_dim,  # 单个智能体的观测维度
        action_dim=action_dim,  # 单个智能体的动作维度
        hidden_dim=128,
        num_agents=num_agents,
        args=args,
    ).to(device)
    actor.load_state_dict(all_model["actor"])
    actor.eval()
    _worker_actor = actor
    torch.set_grad_enabled(False)


def run_episodes_chunk(angle, radius, seeds_chunk):
    """运行一包种子（多局），返回 (angle, radius, win_count, total_episodes)"""
    global _worker_actor, _worker_name_agents

    env = Env3v1(sector_angle_deg=angle, sector_radius_km=radius, **env_kwargs)

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
            with torch.inference_mode():
                while not (done or truncated):
                    # 收集所有智能体的观测
                    obs_list = [obs[name] for name in _worker_name_agents]
                    # 拼接成 (1, num_agents, obs_dim) 张量
                    obs_tensor = torch.tensor(obs_list, dtype=torch.float32).unsqueeze(
                        0
                    )
                    obs_tensor = obs_tensor.to(device)

                    # 共享网络前向推理
                    actions_tensor, _ = _worker_actor(
                        obs_tensor
                    )  # (1, num_agents, action_dim)
                    actions_tensor = (
                        actions_tensor.squeeze(0).cpu().detach()
                    )  # (num_agents, action_dim)

                    # 转换为字典动作
                    actions = {}
                    for idx, name in enumerate(_worker_name_agents):
                        actions[name] = actions_tensor[idx].numpy()

                    obs, _, done, truncated, info = env.step(actions)
            if info.get("capture_reward_claimed", False):
                wins += 1
    finally:
        gc.enable()
        env.close()

    return angle, radius, wins, len(seeds_chunk)


def get_env_shapes():
    """获取观测/动作空间维度（单个智能体）"""
    temp_env = Env3v1(sector_angle_deg=360, sector_radius_km=30, **env_kwargs)
    obs_space = temp_env.observation_space
    action_space = temp_env.action_space
    name_agents = list(obs_space.keys())
    # 假设所有智能体的观测/动作维度相同（对于 3v1 环境通常成立）
    obs_dim = obs_space[name_agents[0]].shape[0]
    action_dim = action_space[name_agents[0]].shape[0]
    temp_env.close()
    return obs_dim, action_dim, len(name_agents), name_agents


def main():
    print(">>> 提取环境空间维度...")
    obs_dim, action_dim, num_agents, name_agents = get_env_shapes()
    print(f"  智能体: {name_agents}")
    print(f"  观测维度: {obs_dim}")
    print(f"  动作维度: {action_dim}")

    # 构建所有 (角度, 半径) 组合的任务
    base_seed = 10000
    tasks = []  # 每个元素为 (angle, radius, seeds_chunk)
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
    print(f">>> 使用 CPU 核心数: {mp.cpu_count()-1} (保留一个给系统)")

    # 结果容器
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

    # 计算胜率矩阵
    win_rate_grid = np.zeros((len(sector_radii), len(sector_angles)))
    for i, radius in enumerate(sector_radii):
        for j, angle in enumerate(sector_angles):
            total = total_counts[(angle, radius)]
            rate = win_counts[(angle, radius)] / total if total > 0 else 0
            win_rate_grid[i, j] = rate
            print(
                f"角度 {angle:3d}° | 半径 {radius:3d} km : 胜场 {win_counts[(angle, radius)]:3d}/{total:3d} , 胜率 = {rate:.3f}"
            )

    # ========== 1. 原始3D曲面图 ==========
    X, Y = np.meshgrid(sector_angles, sector_radii)
    Z = win_rate_grid

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.9)
    ax.set_xlabel("Sector Angle (deg)")
    ax.set_ylabel("Sector Radius (km)")
    ax.set_zlabel("Win Rate")
    ax.set_title("Interceptor Win Rate vs Initial Sector (Shared Network)")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    plt.tight_layout()
    plt.savefig("win_rate_3d_sector_shared.png", dpi=150)
    plt.close()
    print("3D曲面图已保存为 win_rate_3d_sector_shared.png")

    # ========== 3. 等高线图 ==========
    fig, ax = plt.subplots(figsize=(10, 6))
    contour = ax.contour(X, Y, Z, levels=10, cmap="plasma", linewidths=1.5)
    ax.clabel(contour, inline=True, fontsize=8)
    ax.set_xlabel("Sector Angle (deg)")
    ax.set_ylabel("Sector Radius (km)")
    ax.set_title("Win Rate Contour")
    # 可选：填充等高线
    ax.contourf(X, Y, Z, levels=20, cmap="viridis", alpha=0.6)
    plt.tight_layout()
    plt.savefig("win_rate_contour.png", dpi=150)
    plt.close()
    print("等高线图已保存为 win_rate_contour.png")

    # ========== 4. 参数关系折线图 ==========
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：固定半径，胜率 vs 角度
    for i, radius in enumerate(sector_radii):
        ax1.plot(sector_angles, win_rate_grid[i, :], marker="o", label=f"R={radius} km")
    ax1.set_xlabel("Sector Angle (deg)")
    ax1.set_ylabel("Win Rate")
    ax1.set_title("Win Rate vs Angle (by Radius)")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    # 右图：固定角度，胜率 vs 半径
    for j, angle in enumerate(sector_angles):
        ax2.plot(sector_radii, win_rate_grid[:, j], marker="s", label=f"θ={angle}°")
    ax2.set_xlabel("Sector Radius (km)")
    ax2.set_ylabel("Win Rate")
    ax2.set_title("Win Rate vs Radius (by Angle)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="best", ncol=2)

    plt.tight_layout()
    plt.savefig("win_rate_lineplot.png", dpi=150)
    plt.close()
    print("参数关系折线图已保存为 win_rate_lineplot.png")


if __name__ == "__main__":
    mp.freeze_support()
    main()
