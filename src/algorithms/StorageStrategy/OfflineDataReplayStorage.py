from argparse import Namespace
from typing import List, Tuple

import numpy as np

from src.ExperienceCollection.ExperienceDatasetReader import ExperienceDatasetReader
from src.algorithms.StorageStrategy.IStorageStrategy import IStorageStrategy
from src.algorithms.StorageStrategy.ReplayBuffer.ReplayBuffer import StandardReplayBuffer


class OfflineDataReplayStorage(IStorageStrategy):
    def __init__(self,
                 obs_shapes: List[Tuple[int]],
                 action_shapes: List[Tuple[int]],
                 num_agents : int,
                 args: Namespace,
                 dataset_path: str,
                 action_continuous: bool = False,
                 ) -> None:
        experienceDatasetReader = ExperienceDatasetReader(dataset_path)
        assert len(experienceDatasetReader) != 0 , "数据集为空"
        assert num_agents == experienceDatasetReader.get_step(0)["obs"].shape[0] , "数据集不符合要求----智能体数量不符合"

        num_env = experienceDatasetReader.get_step(0)['obs'].shape[1]
        dataset_size = len(experienceDatasetReader)
        buffer_size = num_env * dataset_size
        self.args = args
        self.replay_buffer = StandardReplayBuffer(
            capacity=buffer_size,
            obs_shapes=obs_shapes,
            action_shapes=action_shapes,
            num_agents=num_agents,
            action_continuous=action_continuous,
        )
        #向缓冲池中存储经验数据
        for i in range(dataset_size):
            data = experienceDatasetReader.get_step(i)
            self.replay_buffer.add(
                obs=data["obs"],
                actions=data["actions"],
                rewards=data["rewards"],
                next_obs=data["next_obs"],
                dones=data["dones"]
            )





    def store_experience(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
        **kwargs
    ) -> None:
        pass




    def can_update(self) -> bool:
        """判断是否可以进行模型更新"""
        return True




