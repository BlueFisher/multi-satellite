import torch


def soft_update(target_network, source_network, tau):
    """
    对目标网络进行软更新（Soft Update）。

    使用参数 τ（tau）控制目标网络向源网络缓慢逼近，
    通常用于目标 Q 网络更新，以提高训练稳定性。

    Args:
        target_network: 需要被更新的目标网络。
        source_network: 当前学习到的源网络。
        tau (float): 更新速率系数，范围通常为 0 < tau <= 1。

    Example:
        >>> soft_update(target_actor, actor, tau=0.005)
        >>> soft_update(target_critic, critic, tau=0.01)
    """
    for target_param, param in zip(
        target_network.parameters(), source_network.parameters()
    ):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class Noise:
    """
    为动作或输入添加高斯噪声的工具类。

    该类常用于连续动作空间的探索，为输入添加
    均值为 mean、标准差为 std 的随机噪声，并支持裁剪范围。

    Attributes:
        mean (float): 噪声均值。
        std (float): 噪声标准差。

    Example:
        >>> noise = Noise(mean=0, std=0.1)
        >>> action = torch.tensor([0.5, -0.2])
        >>> noisy_action = noise(action, min=-1.0, max=1.0)
    """

    def __init__(self, mean=0, std=0.10):
        """
        初始化噪声生成器。

        Args:
            mean (float): 高斯噪声的均值。
            std (float): 高斯噪声的标准差。

        Example:
            >>> noise = Noise(mean=0, std=0.2)
        """
        self.mean = mean
        self.std = std

    def __call__(self, x, min: int, max: int):
        """
        为输入添加高斯噪声并进行范围裁剪。

        噪声按照 N(mean, std) 进行采样，并对结果进行 min-max 裁剪，
        多用于连续动作空间中的随机探索。

        Args:
            x (Tensor): 输入张量，例如动作值。
            min (int): 下界裁剪值。
            max (int): 上界裁剪值。

        Returns:
            Tensor: 添加噪声并裁剪后的结果。

        Example:
            >>> noise = Noise(0, 0.1)
            >>> action = torch.tensor([0.3, -0.5])
            >>> noisy_action = noise(action, -1, 1)
        """
        noise = torch.randn_like(x) * self.std + self.mean
        x = x + noise
        x = torch.clip(x, min, max)
        return x


class OUNoise:
    """
    Ornstein-Uhlenbeck 噪声生成器，用于 DDPG 等算法中的动作探索。

    OU 过程满足随机微分方程：dx = theta * (mu - x) * dt + sigma * dW，
    其中 dW 是维纳过程。离散化后用于生成时间上相关的噪声，使探索更加平滑。

    Attributes:
        mu (float): 长期均值，噪声最终会围绕该值波动。
        theta (float): 均值回归速度，越大则状态越快回到 mu。
        sigma (float): 波动率，控制噪声的随机强度。
        dt (float): 时间步长，通常取 1 或较小的值（如 0.01）。
        state (Tensor or None): 当前噪声状态，形状与输入动作一致。

    Example:
        >>> noise = OUNoise(mu=0.0, theta=0.15, sigma=0.2, dt=1e-2)
        >>> action = torch.tensor([0.5, -0.2])
        >>> noisy_action = noise(action, min=-1.0, max=1.0)
        >>> # 重置噪声过程（例如每个 episode 开始时）
        >>> noise.reset()
    """

    def __init__(self, mu=0.0, theta=0.15, sigma=0.2, dt=1e-2):
        """
        初始化 OU 噪声生成器。

        Args:
            mu (float): 均值参数。
            theta (float): 均值回归系数。
            sigma (float): 噪声标准差。
            dt (float): 离散时间步长。
        """
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.state = None  # 将在第一次调用时根据输入形状初始化

    def reset(self):
        """重置噪声状态为 None，下次调用时重新初始化为零向量。"""
        self.state = None

    def __call__(self, x, min=None, max=None):
        """
        为输入添加 OU 噪声并进行范围裁剪。

        根据当前状态更新 OU 过程，将更新后的状态加到输入上，
        然后裁剪到指定范围。

        Args:
            x (Tensor): 输入张量，通常为动作值。
            min (float, optional): 裁剪下界，若不提供则不裁剪。
            max (float, optional): 裁剪上界，若不提供则不裁剪。

        Returns:
            Tensor: 添加噪声并裁剪后的结果。

        Example:
            >>> noise = OUNoise()
            >>> action = torch.tensor([0.3, -0.5])
            >>> noisy_action = noise(action, min=-1.0, max=1.0)
        """
        # 若状态未初始化，则根据 x 的形状创建零状态
        if self.state is None:
            self.state = torch.zeros_like(x)

        # 生成 OU 增量
        dx = self.theta * (self.mu - self.state) * self.dt + self.sigma * torch.sqrt(
            torch.tensor(self.dt)
        ) * torch.randn_like(x)
        self.state = self.state + dx

        # 将噪声加到输入上
        x_noisy = x + self.state

        # 可选裁剪
        if min is not None and max is not None:
            x_noisy = torch.clip(x_noisy, min, max)
        return x_noisy
