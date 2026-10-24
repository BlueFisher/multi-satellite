import importlib

from .RewardBase import RewardBase


class RewardComputerFactory:
    @staticmethod
    def get_reward_computer(computer_type: str) -> RewardBase:

        if not isinstance(computer_type, str):
            raise TypeError("RewardComputer type must be str")

        try:
            module = importlib.import_module(
                ".reward." + computer_type, package=__package__
            )
            class_instance = getattr(module, computer_type)
            return class_instance()

        except:
            raise


if __name__ == "__main__":
    computer_type = "RewardComputer3v1"

    reward_computer = RewardComputerFactory.get_reward_computer(computer_type)
