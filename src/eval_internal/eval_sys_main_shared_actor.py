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

from algorithms.Network.Actor.SharedMessageDeterministicActor import (
    SharedMessageDeterministicActor,
)
from environment import SimulatorType

task_type = "2v2"
model_path = {
    # "3v1": R"C:\Users\computer\Desktop\new_train\3.25-140328-c100\actor_critic.pth"
    "3v1": R"C:\Users\computer\Desktop\new_train\3.25-140328-c200\actor_critic.pth",
    "2v2": r"C:\Users\computer\Desktop\new_train\4.3-185944-i200\actor_critic.pth",
}
num_episodes = 500


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
    "collision_threshold": 0.1,
    "agent_heuristic": False,
    "opponent_heuristic": True,
    "m_max_episode_steps": 2000,
    "on_plane": False,
    "simulator": SimulatorType.Local,
    "limit_accel": False,
    "attacker_max_accel": attacker_max_accel,
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
            "obs_compatible": False,
            "obs_add_distance": False,
            "use_diff_obs": True,
            "obs_window_size": 2,
            "action_window_size": 1,
            "capture_sustain_steps": 50,
        }
    )
    env = SatelliteEnv3v1Hvt(**env_kwargs)
    logger_callback = Satellite3v1HvtLoggerCallback(None, args)
elif task_type == "2v2":
    from src.utils.Logger.LoggerCallback import Satellite2v2LoggerCallback
    from src.environment.SatelliteEnv2v2 import SatelliteEnv2v2

    env_kwargs.update({"intercept_time": 100})
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

args = Namespace()
args.use_relation = True
args.use_ndq = True
args.use_message_gate = True
args.device = "cuda"
args.relation_topk = 1
actor = SharedMessageDeterministicActor(
    obs_dim=n_obs[0], action_dim=n_actions[0], hidden_dim=128, num_agents=2, args=args
).to(device)
actor.load_state_dict(all_model[f"actor"])

episode_rewards = []
for episode in range(num_episodes):
    # 重置环境
    observations, info = env.reset()
    total_reward = 0
    done = {name: False for name in name_agents}
    truncated = {name: False for name in name_agents}

    while True:
        actions = {}

        obs = [observations[name] for name in name_agents]
        obs = torch.FloatTensor(obs).unsqueeze(0).to(device)
        act, _ = actor(obs)
        act = act.cpu().detach().squeeze(0).numpy()

        actions.update({name: act[index] for index, name in enumerate(name_agents)})
        next_observations, rewards, done, truncated, info = env.step(actions)

        observations = next_observations
        env.render()

        if done or truncated:
            logger_callback.after_eval_episode([], info)

            break
logger_callback.on_eval_log_interval(None)
env.close()
