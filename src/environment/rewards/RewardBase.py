import abc


class RewardBase(abc.ABC):
    def __init__(self):
        """
        RewardBase在被初始化的时候环境会把 env 赋值，也就是说 self.env 始终指向环境
        """
        self.env = None

    def reset(self):
        """
        如果奖励函数需要reset，在子类中重写该方法
        """
        pass

    @abc.abstractmethod
    def compute_reward(self):
        """
        子类必须实现该方法，计算每一个智能体的奖励, 返回格式为 Python 字典
        返回格式示例:
            {
                "agent_1": 1.0f,
                "agent_2": 2.0f,
                ...
            }
        """
        pass
