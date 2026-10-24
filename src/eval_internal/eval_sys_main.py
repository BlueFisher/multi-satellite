"""
！！！ 注意 ！！！
这个只是一个内部快速评估模型的脚本，不需要启动Unity, 快速的对模型进行测试。实际脚本中很多内容和环境强耦合
如果需要使用该脚本对模型进行测试，请仔细阅读代码并针对需要测试的环境做出修改
如果不想修改代码，Unity也集成了测试功能（Unity 会调用 cmd_show_sys.py这个脚本，该脚本对所有环境做出了适配），请使用Unity进行测试
"""

import random
from argparse import Namespace
from typing import Dict

import numpy as np
import torch

from Evaluation.draw import convert_to_lvlh, plot_lvlh_trajectories
from algorithms.Network.Actor.DeterministicActor import DeterministicActor
from algorithms.Network.Actor.IntentPredictActor import IntentDeterministicActor
from environment import SimulatorType

task_type = "3v1"
model_path = {
    # "3v1": r"D:\Ropes\Laboratory\803\PythonSatellite\output\models\SatelliteEnv3v1HvtNoPlane\MATD3_Intent\Intent-20260122-114445\actor_critic.pth"
    "3v1": r"D:\Ropes\Laboratory\803\PythonSatellite\output\models\SatelliteEnv3v1Hvt\MATD3_Intent\3v1OnPlaneCapture200-20260123-141534\actor_critic.pth"
    # "2v2": "pretrain_model/MATD3-2v2/V7/V7_7_3w_2000_300time/episode_27800/actor_critic.pth",
    # "3v1": "pretrain_model/MATD3-3v1Hvt/episode_19700/actor_critic.pth",
    # "3v1": "pretrain_model/MATD3-3v1Hvt/episode_5900/actor_critic.pth",
    # "3v1": "pretrain_model/MATD3-3v1Hvt/episode_35000/actor_critic.pth",
}
num_episodes = 100


generator = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed = 8

if seed != -1:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

update_interval = 2
attacker_max_accel = 5 / 6

env_kwargs = {
    "render_mode": "None",  # None,human
    "update_interval": update_interval,
    "action_scale": update_interval * 0.0005,
    "collision_threshold": 1,
    "agent_heuristic": False,
    "opponent_heuristic": True,
    "m_max_episode_steps": 2000,
    "on_plane": True,
    "simulator": SimulatorType.Local,
    # "config_name": "测试1042.yaml",
}

logger_callback = None

args = Namespace(
    **{"eval_episode": num_episodes, "just_eval": True, "debug_mode": True}
)
if task_type == "3v1":
    from src.utils.Logger.LoggerCallback import Satellite3v1HvtLoggerCallback
    from src.environment.SatelliteEnv3v1Hvt import SatelliteEnv3v1Hvt

    env_kwargs.update(
        {
            "use_diff_obs": True,
            "obs_add_distance": True,
            "obs_window_size": 2,
            "action_window_size": 1,
            "attacker_max_accel": attacker_max_accel,
            "capture_sustain_steps": 400,
        }
    )
    env = SatelliteEnv3v1Hvt(**env_kwargs)
    logger_callback = Satellite3v1HvtLoggerCallback(None, args)
elif task_type == "2v2":
    from src.utils.Logger.LoggerCallback import Satellite2v2LoggerCallback
    from src.environment.SatelliteEnv2v2 import SatelliteEnv2v2

    env_kwargs.update({"attacker_max_accel": attacker_max_accel, "intercept_time": 300})
    env = SatelliteEnv2v2(**env_kwargs)
    logger_callback = Satellite2v2LoggerCallback(None, args)
else:
    raise ValueError("Invalid task type")

obs_space = env.observation_space
action_space = env.action_space

name_agents = list(obs_space.keys())
num_agents = len(name_agents)

n_actions = [action_space[name].shape[0] for name in name_agents]
n_obs = [obs_space[name].shape[0] for name in name_agents]

# 计算输入维度
obs_dim_sum = sum(n_obs)
actions_dim_sum = sum(n_actions)
print(obs_dim_sum, actions_dim_sum)

actors = []
all_model = torch.load(model_path[task_type], map_location=device)
for i in range(num_agents):
    actor = IntentDeterministicActor(
        input_dim=n_obs[i],
        output_dim=n_actions[i],
        hidden_dim=128,
        intent_predictor_count=3,
        intent_predictor_output_dim=n_actions[i],
    )
    actor.load_state_dict(all_model[f"actor{i}"])
    actors.append(actor)


def generate_image(traj_dict: Dict, ref: str, output_dir: str, eci=False) -> None:
    # 转换为LVLH坐标系
    lvlh_dict = convert_to_lvlh(traj_dict, ref) if eci else traj_dict
    # 检查轨迹范围，自动调整视图大小
    max_range = 0
    for key, traj in lvlh_dict.items():
        lvlh_dict[key] = np.array(traj)
        max_val = np.max(np.abs(lvlh_dict[key][:, :2]))
        if max_val > max_range:
            max_range = max_val

    # 设置视图大小为最大值的1.5倍，至少0.1km
    view_size = max(0.1, max_range * 1.5)
    print(f"Auto-adjusted view size: ±{view_size:.2f} km")

    # 生成LVLH轨迹图像
    plot_lvlh_trajectories(lvlh_dict, view_size=view_size, output_dir=output_dir)


episode_rewards = []
for episode in range(num_episodes):
    # 重置环境
    observations, info = env.reset()
    total_reward = 0
    done = {name: False for name in name_agents}
    truncated = {name: False for name in name_agents}

    if generator:
        trail = {}

    while True:
        actions = {}
        with torch.no_grad():
            for i, name in enumerate(name_agents):
                obs = torch.FloatTensor(observations[name])
                action = actors[i](obs)
                actions[name] = action.numpy()
        next_observations, rewards, done, truncated, info = env.step(actions)
        observations = next_observations
        env.render()

        if generator:
            state = env.unwrapped.get_all_satellite_state()
            for key, value in state.items():
                if key not in trail:
                    trail[key] = []
                trail[key].append(value)

        if done or truncated:
            logger_callback.after_eval_episode([], info)
            # print(info)

            if generator:
                generate_image(trail, "HVT_0", f"output/image{task_type}", False)
            break
logger_callback.on_eval_log_interval(None)
env.close()
