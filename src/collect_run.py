import random
from argparse import Namespace
from copy import deepcopy
from typing import Dict

import gymnasium as gym
import numpy as np
import torch
from gymnasium import VectorizeMode

from Evaluation.draw import convert_to_lvlh, plot_lvlh_trajectories
from algorithms.Network.Actor.DeterministicActor import DeterministicActor
from environment import SimulatorType
from src.ExperienceCollection.ExperienceDatasetWriter import ExperienceDatasetWriter
from src.environment.RegisterEnv import register_env
from src.utils.utils_func import load_config, load_args, get_alg_manager


def overwrite_terminal_next_obs(
    experience_next_obs,
    next_observations,
    info,
    dones_env,
    name_agents,
):
    num_envs = len(dones_env)

    if "final_obs" not in info:
        return experience_next_obs

    for env_idx in range(num_envs):
        if dones_env[env_idx]:
            final_obs = info["final_obs"][env_idx]
            for i, agent in enumerate(name_agents):
                experience_next_obs[i][env_idx] = final_obs[agent]

    return experience_next_obs


def build_eval_components(
    task_type: str,
    model_path: Dict[str, str],
    num_episodes: int,
    seed: int,
    device: torch.device,
):
    """
    构建评估所需的所有组件：
    env, actors, logger_callback, agent metadata
    """

    # ------------------ seed ------------------
    if seed != -1:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    # ------------------ env kwargs ------------------
    update_interval = 2
    attacker_max_accel = 5 / 6

    env_kwargs = {
        "render_mode": "None",
        "update_interval": update_interval,
        "action_scale": update_interval * 0.0005,
        "collision_threshold": 0.1,
        "agent_heuristic": False,
        "opponent_heuristic": True,
        "m_max_episode_steps": 2000,
        "on_plane": False,
        "simulator": SimulatorType.Local,

    }

    args = Namespace(
        **{"eval_episode": num_episodes, "just_eval": True, "debug_mode": True}
    )

    args.seed = seed




    # ------------------ env & logger ------------------
    if task_type == "3v1":
        from src.environment.SatelliteEnv3v1Hvt import SatelliteEnv3v1Hvt
        from src.utils.Logger.LoggerCallback import Satellite3v1HvtLoggerCallback

        env_kwargs.update({
            "id": "SatelliteEnv3v1Hvt",
            "use_diff_obs": True,
            "obs_add_distance": True,
            "attacker_max_accel": attacker_max_accel,
        })

    #    env = SatelliteEnv3v1Hvt(**env_kwargs)
        logger_callback = Satellite3v1HvtLoggerCallback(None, args)

    elif task_type == "2v2":
        from src.environment.SatelliteEnv2v2 import SatelliteEnv2v2
        from src.utils.Logger.LoggerCallback import Satellite2v2LoggerCallback

        env_kwargs.update({
            "id": "SatelliteEnv2v2",
            "attacker_max_accel": attacker_max_accel,
            "intercept_time": 300
        })

       # env = SatelliteEnv2v2(**env_kwargs)
        logger_callback = Satellite2v2LoggerCallback(None, args)

    else:
        raise ValueError(f"Invalid task type: {task_type}")

    #创建并行环境
    env = gym.make_vec(
        vectorization_mode=VectorizeMode.ASYNC,
        vector_kwargs={
            "autoreset_mode": gym.vector.AutoresetMode.SAME_STEP,
            "context": "spawn",
        },
        num_envs=num_envs,
        **env_kwargs
    )

    # ------------------ actor loading ------------------
    obs_space = env.observation_space
    action_space = env.action_space
    name_agents = list(obs_space.keys())
    num_agents = len(name_agents)

    args.name_agents = name_agents

    n_obs = [obs_space[name].shape[1] for name in name_agents]
   # n_actions = [action_space[name].shape[0] for name in name_agents]

    n_actions = [action_space[name].shape[1] for name in name_agents]

    actors = []
    all_model = torch.load(model_path[task_type], map_location=device)

    for i in range(num_agents):
        actor = DeterministicActor(
            input_dim=n_obs[i],
            output_dim=n_actions[i],
            hidden_dim=128,
        )
        actor.load_state_dict(all_model[f"actor{i}"])
        actor.to(device)
        actor.eval()
        actors.append(actor)

    #构建数据集存储对象
    writer = ExperienceDatasetWriter(
        save_path=f"dataset/{model_path[task_type]}_{max_step}.h5",
        max_steps=max_step // env.num_envs + 1,
    )


    return {
        "env": env,
        "actors": actors,
        "logger": logger_callback,
        "name_agents": name_agents,
        "num_agents": num_agents,
        "args": args,
        "writer":writer
    }

def eval_parallel_collect_dataset(
    args: Namespace,
    env: gym.vector.AsyncVectorEnv,
    actors: list,
    device: torch.device,
    target_step: int,
    writer: ExperienceDatasetWriter,
):
    observations, info = env.reset(seed=args.seed + 1000)

    num_envs = env.num_envs
    num_agents = len(args.name_agents)

    obs_list = [observations[agent] for agent in args.name_agents]

    episode_count = np.zeros(num_envs, dtype=int)
    total_step = 0

    while total_step < target_step:
        # ---------- choose action ----------
        actions = {}
        with torch.no_grad():
            for i, agent in enumerate(args.name_agents):
                obs_tensor = torch.as_tensor(
                    obs_list[i], dtype=torch.float32, device=device
                )
                actions[agent] = actors[i](obs_tensor).cpu().numpy()

        # ---------- env step ----------


        next_observations, rewards, terminations, truncations, info = env.step(actions)

        dones_env = np.logical_or(terminations, truncations)

        # ---------- next_obs ----------
        next_obs_list = [next_observations[agent] for agent in args.name_agents]
        if dones_env.any():
            obs_list = next_obs_list
            continue
        print(total_step)

        experience_next_obs = [obs.copy() for obs in next_obs_list]

        experience_next_obs = overwrite_terminal_next_obs(
            experience_next_obs,
            next_observations,
            info,
            dones_env,
            args.name_agents,
        )

        # ---------- pack experience ----------
        obs_np = np.stack(obs_list, axis=0)
        actions_np = np.stack([actions[a] for a in args.name_agents], axis=0)



        # store_dones = np.repeat(
        #     terminations[:, None], num_agents, axis=1
        # )
        # store_dones = np.repeat(
        #     store_dones[:, :, None], num_agents, axis=2
        # )
        store_dones = np.repeat(terminations[:, np.newaxis], num_agents, axis=1)

        next_obs_np = np.stack(experience_next_obs, axis=0)

        writer.append({
            "obs": obs_np,
            "actions": actions_np,
            "rewards": rewards,
            "next_obs": next_obs_np,
            "dones": store_dones,
        })

        # ---------- update episode count ----------
        for env_idx in range(num_envs):
            total_step += 1
            if dones_env[env_idx]:
                episode_count[env_idx] += 1



        obs_list = next_obs_list

    env.close()
    writer.close()


#配置变量，之后需要修改
num_envs = 10
task_type = "2v2"
model_path = {
    # "3v1": "C:/Users/computer/Desktop/14001/actor_critic.pth"
    #"2v2": "pretrain_model/MATD3-2v2/V7/V7_7_3w_2000_300time/episode_27800/actor_critic.pth",
    # "3v1": "pretrain_model/MATD3-3v1Hvt/episode_19700/actor_critic.pth",
    # "3v1": "pretrain_model/MATD3-3v1Hvt/episode_5900/actor_critic.pth",
    # "3v1": "pretrain_model/MATD3-3v1Hvt/episode_35000/actor_critic.pth",
    #"2v2": "output/models/SatelliteEnv2v2NoPlane/MATD3/HighOrbitNoPlane-20251225-094239/actor_critic.pth"
    "2v2": "pretrain_model/MATD3-2v2/V8/V8_13_02_01_2000_200time_offplane/15201/actor_critic.pth"
}
max_step = 100000
seed = 8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
save_path = "dataset/eval_parallel.h5"


def main() -> None:
    register_env()
    component_dict = build_eval_components(task_type, model_path, max_step, seed, device)


    eval_parallel_collect_dataset(component_dict["args"],component_dict["env"],component_dict["actors"],device,max_step,component_dict["writer"])
   # eval_collect_parallel(train_env ,component_dict["actors"],alg ,args,device, )



if __name__ == "__main__":
   main()


