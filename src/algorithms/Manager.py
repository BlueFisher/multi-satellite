from argparse import Namespace

from .Action.IActorStrategy import IActorStrategy
from .StorageStrategy.IStorageStrategy import IStorageStrategy
from .Trainer.ITrainer import ITrainer


class Manager:
    def __init__(
        self,
        actor_strategy: IActorStrategy,
        storage_strategy: IStorageStrategy,
        trainer: ITrainer,
        args: Namespace,
    ):
        self.action = actor_strategy
        self.storage = storage_strategy
        self.trainer = trainer
        self.args = args

        if args.load_model:
            self.load_model(args.load_path)

    def choose_action(self, agent_list, obs, global_state=None, eval=False):
        self.action.before_action_selection()
        ret = self.action.select_action(
            agent_list, obs, global_state=global_state, evaluation=eval
        )
        self.action.after_action_selection()
        return ret

    def store_experience(self, *args, **kwargs):
        self.storage.before_storage()
        self.storage.store_experience(*args, **kwargs)
        self.storage.after_storage()

    def train(self):
        self.trainer.before_training()
        ret = self.trainer.train()
        self.trainer.after_training()
        return ret

    def can_train(self):
        return self.storage.can_update()

    def save_model(self, directory: str):
        self.trainer.save_models(directory)

    def load_model(self, directory: str):
        self.trainer.load_models(directory)

    def before_episode(self, env_idx: int):
        pass

    def after_episode(self, env_idx: int):
        pass
