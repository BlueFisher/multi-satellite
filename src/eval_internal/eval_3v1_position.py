import os
import gc
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D  # 用于 3D 绘图

from eval_internal.Env3v1 import Env3v1

# ----- 限制 CPU 线程数 -----
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from algorithms.Network.Actor.IntentPredictActor import IntentDeterministicActor
from environment.visualization.SimulatorFactory import SimulatorType

# ================= 参数配置 =================
task_type = "3v1"
model_path = {
    "3v1": r"D:\Ropes\Laboratory\803\PythonSatellite\output\models\SatelliteEnv3v1Hvt\MATD3_Intent\3v1OnPlaneCapture200-20260123-141534\actor_critic.pth"
}

num_episodes_per_pair = 50  # 每个(角度,半径)组合的测试局数（可调小）
device = torch.device("cpu")
seed = 8

# 环境固定参数
env_kwargs = {
    "render_mode": "None",
    "update_interval": 2,
    "action_scale": 2 * 0.0005,
    "collision_threshold": 1,
    "agent_heuristic": False,
    "opponent_heuristic": True,
    "m_max_episode_steps": 2000,
    "on_plane": True,
    "simulator": SimulatorType.Local,
    "use_diff_obs": True,
    "obs_add_distance": True,
    "obs_window_size": 2,
    "action_window_size": 1,
    "attacker_max_accel": 0.4,
    "capture_sustain_steps": 100,
}

# 扇区角度列表（度）
sector_angles = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]
# 扇区半径列表（km）
sector_radii = [20, 25, 30, 35, 40, 45, 50]  # 最小半径20与RHO_MIN一致

CHUNK_SIZE = 10  # 每个 worker 一次处理的局数

# ================= 全局 worker 缓存 =================
_worker_actors = None
_worker_name_agents = None


def init_worker(model_path, n_obs_list, n_actions_list, name_agents):
    """子进程初始化：加载模型"""
    global _worker_actors, _worker_name_agents
    _worker_name_agents = name_agents
    all_model = torch.load(model_path[task_type], map_location="cpu")
    actors = []
    for i, name in enumerate(name_agents):
        actor = IntentDeterministicActor(
            input_dim=n_obs_list[i],
            output_dim=n_actions_list[i],
            hidden_dim=128,
            intent_predictor_count=3,
            intent_predictor_output_dim=n_actions_list[i],
        )
        actor.load_state_dict(all_model[f"actor{i}"])
        actor.eval()
        actors.append(actor)
    _worker_actors = actors
    torch.set_grad_enabled(False)


def run_episodes_chunk(angle, radius, seeds_chunk):
    """运行一包种子（多局），返回 (angle, radius, win_count, total_episodes)"""
    global _worker_actors, _worker_name_agents

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
                    actions = {}
                    for i, name in enumerate(_worker_name_agents):
                        obs_tensor = torch.as_tensor(obs[name], dtype=torch.float32)
                        act = _worker_actors[i](obs_tensor).numpy()
                        actions[name] = act
                    obs, _, done, truncated, info = env.step(actions)
            if info.get("capture_reward_claimed", False):
                wins += 1
    finally:
        gc.enable()
        env.close()

    return angle, radius, wins, len(seeds_chunk)


def get_env_shapes():
    """获取观测/动作空间维度"""
    temp_env = Env3v1(sector_angle_deg=360, sector_radius_km=30, **env_kwargs)
    obs_space = temp_env.observation_space
    action_space = temp_env.action_space
    name_agents = list(obs_space.keys())
    n_obs = [obs_space[name].shape[0] for name in name_agents]
    n_actions = [action_space[name].shape[0] for name in name_agents]
    temp_env.close()
    return n_obs, n_actions, name_agents


def main():
    print(">>> 提取环境空间维度...")
    n_obs, n_actions, name_agents = get_env_shapes()
    num_agents = len(name_agents)
    print(f"  智能体: {name_agents}")
    print(f"  观测维度: {n_obs}")
    print(f"  动作维度: {n_actions}")

    # 构建所有 (角度, 半径) 组合的任务
    base_seed = 10000
    tasks = []  # 每个元素为 (angle, radius, seeds_chunk)
    for angle in sector_angles:
        for radius in sector_radii:
            # 为该组合生成所有局数的随机种子
            all_seeds = [
                base_seed + hash((angle, radius, i)) % 1000000
                for i in range(num_episodes_per_pair)
            ]
            # 切分成 chunk
            for i in range(0, num_episodes_per_pair, CHUNK_SIZE):
                chunk_seeds = all_seeds[i : i + CHUNK_SIZE]
                tasks.append((angle, radius, chunk_seeds))

    total_episodes = sum(len(chunk) for _, _, chunk in tasks)
    print(f">>> 总测试局数: {total_episodes}, 切分为 {len(tasks)} 个计算块")
    print(f">>> 使用 CPU 核心数: {mp.cpu_count()-1} (保留一个给系统)")

    # 结果容器：二维字典 / 2D数组
    win_counts = {(a, r): 0 for a in sector_angles for r in sector_radii}
    total_counts = {(a, r): 0 for a in sector_angles for r in sector_radii}

    with ProcessPoolExecutor(
        max_workers=max(1, mp.cpu_count() - 1),
        initializer=init_worker,
        initargs=(model_path, n_obs, n_actions, name_agents),
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

    # 计算胜率并填入二维数组（用于绘图）
    win_rate_grid = np.zeros((len(sector_radii), len(sector_angles)))
    for i, radius in enumerate(sector_radii):
        for j, angle in enumerate(sector_angles):
            total = total_counts[(angle, radius)]
            rate = win_counts[(angle, radius)] / total if total > 0 else 0
            win_rate_grid[i, j] = rate
            print(
                f"角度 {angle:3d}° | 半径 {radius:3d} km : 胜场 {win_counts[(angle, radius)]:3d}/{total:3d} , 胜率 = {rate:.3f}"
            )

    # 绘制 3D 曲面图
    X, Y = np.meshgrid(sector_angles, sector_radii)  # X:角度, Y:半径
    Z = win_rate_grid

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.9)
    ax.set_xlabel("Sector Angle (deg)")
    ax.set_ylabel("Sector Radius (km)")
    ax.set_zlabel("Win Rate")
    ax.set_title("Interceptor Win Rate vs Initial Sector (Angle & Radius)")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    plt.tight_layout()
    plt.savefig("win_rate_3d_sector.png", dpi=150)
    plt.show()
    print("3D 图表已保存为 win_rate_3d_sector.png")


if __name__ == "__main__":
    mp.freeze_support()
    main()
