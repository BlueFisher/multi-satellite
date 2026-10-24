import numba
import numpy as np
from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit

from .IGenerator import IGenerator


# 核心计算优化 - 使用Numba JIT编译
@numba.jit(nopython=True, cache=True)
def rk4_integrate(
    state_vector: np.ndarray, k: float, accel_val: np.ndarray, step: float
) -> np.ndarray:
    """
    使用四阶 Runge-Kutta (RK4) 方法对轨道动力学方程进行积分，
    根据输入状态、引力常数和加速度，计算积分步后的新状态。

    此函数通常用于轨道动力学数值积分，可处理外加加速度的轨道传播。

    Args:
        state_vector (np.ndarray): 初始状态向量，格式为 [x, y, z, vx, vy, vz]。
        k (float): 引力参数（如 μ = GM）。
        accel_val (np.ndarray): 外加加速度向量。
        step (float): 积分步长（单位：秒）。

    Returns:
        np.ndarray: 积分后的状态向量。

    示例:
        >>> next_state = rk4_integrate(
        ...     state_vector=np.array([7000, 0, 0, 0, 7.5, 0]),
        ...     k=398600.4418,
        ...     accel_val=np.array([0, 0, 0]),
        ...     step=10.0
        ... )
        >>> print(next_state)
    """
    h = step
    k1 = dynamics(0, state_vector, k, accel_val)
    k2 = dynamics(0, state_vector + 0.5 * h * k1, k, accel_val)
    k3 = dynamics(0, state_vector + 0.5 * h * k2, k, accel_val)
    k4 = dynamics(0, state_vector + h * k3, k, accel_val)
    return state_vector + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


@numba.jit(nopython=True, cache=True)
def dynamics(t: float, y: np.ndarray, k: float, accel_val: np.ndarray) -> np.ndarray:
    """
    定义轨道运动的动力学方程，用于计算位置与速度的一阶导数。
    方程包含中心引力模型以及外加加速度项。

    本函数通常作为数值积分器（如 RK4）的一部分使用。

    Args:
        t (float): 时间参数（通常不使用，保留用于积分器兼容性）。
        y (np.ndarray): 当前状态向量，格式为 [x, y, z, vx, vy, vz]。
        k (float): 引力常数（μ = GM）。
        accel_val (np.ndarray): 外加加速度向量。

    Returns:
        np.ndarray: 状态导数向量，格式为 [vx, vy, vz, ax, ay, az]。

    示例:
        >>> dy = dynamics(
        ...     t=0.0,
        ...     y=np.array([7000, 0, 0, 0, 7.5, 0]),
        ...     k=398600.4418,
        ...     accel_val=np.array([0, 0, 0])
        ... )
        >>> print(dy)
    """

    r = y[:3]
    v = y[3:]
    r_norm = np.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
    r_norm3 = r_norm * r_norm * r_norm
    gravity_accel = -k * r / r_norm3
    total_accel = gravity_accel + accel_val
    return np.array([v[0], v[1], v[2], total_accel[0], total_accel[1], total_accel[2]])


class OrbitGeneratorF(IGenerator):
    """
    高性能轨道状态更新生成器。

    本类用于在有推力和无推力条件下更新航天器轨道状态，
    包含以下特性：

    1. 对于无推力情况，使用解析传播方法以获得极高运行速度。
    2. 对于有推力情况，使用固定步长的 RK4 数值积分方法。
    3. 核心数值计算基于 Numba JIT 优化以提升执行效率。
    4. 支持连续调用以进行实时轨道传播。

    Attributes:
        rtol (float): 数值计算的相对误差容限。
        atol (float): 数值计算的绝对误差容限。
        zero_accel (np.ndarray): 表示无推力情况下的加速度向量。

    示例:
        >>> gen = OrbitGeneratorF()
        >>> state = np.array([7000, 0, 0, 0, 7.5, 0])
        >>> accel = np.array([0.0, 0.0, 0.0])
        >>> next_state = gen(state, accel, step=10)
        >>> print(next_state)
    """

    def __init__(self, rtol=1e-8, atol=1e-10):
        """
        初始化轨道生成器配置，包括误差容限和默认无推力向量。

        Args:
            rtol (float): 数值传播的相对误差限制。
            atol (float): 数值传播的绝对误差限制。

        示例:
            >>> gen = OrbitGeneratorF(rtol=1e-9, atol=1e-12)
        """
        self.rtol = rtol
        self.atol = atol
        self.zero_accel = np.array([0.0, 0.0, 0.0])  # 无推力状态（单位：km/s²）

    def __call__(
        self,
        state: np.ndarray[(6,), np.float64],  # 前3维位置(km)，后3维速度(km/s)
        accel: np.ndarray,  # 输入加速度（单位：km/s²，纯数值数组）
        step: float = 10,  # 步长（单位：秒）
    ) -> np.ndarray[(6,), np.float64]:
        """
        根据输入状态与加速度计算下一步轨道状态。

        若加速度近似为零，则自动选择解析轨道传播方法；
        否则使用固定步长的 RK4 数值积分方法。

        Args:
            state (np.ndarray): 当前轨道状态向量 [x, y, z, vx, vy, vz]。
            accel (np.ndarray): 加速度向量（单位：km/s²）。
            step (float): 传播时间步长（单位：秒）。

        Returns:
            np.ndarray: 更新后的轨道状态向量。

        示例:
            >>> gen = OrbitGeneratorF()
            >>> next_state = gen(
            ...     state=np.array([7000, 0, 0, 0, 7.5, 0]),
            ...     accel=np.array([0.0, 0.0, 0.0]),
            ...     step=10
            ... )
            >>> print(next_state)
        """

        # 检查是否为无推力状态（允许微小数值误差）
        if np.allclose(accel, self.zero_accel, atol=1e-12):
            # 无推力时使用解析解法
            return self._propagate_analytical(state, step)
        else:
            # 有推力时使用固定步长RK4
            return self._propagate_with_rk4(state, accel, step)

    def _propagate_analytical(self, state: np.ndarray, step: float) -> np.ndarray:
        """
        在无推力情况下，使用轨道力学解析解法更新状态。

        利用 poliastro 的解析传播能力，在忽略外推力时实现高效轨道演化。

        Args:
            state (np.ndarray): 当前状态向量 [x, y, z, vx, vy, vz]。
            step (float): 传播时间（秒）。

        Returns:
            np.ndarray: 传播后的更新状态。

        示例:
            >>> gen = OrbitGeneratorF()
            >>> gen._propagate_analytical(
            ...     state=np.array([7000, 0, 0, 0, 7.5, 0]),
            ...     step=20
            ... )
        """
        r = state[:3]
        v = state[3:]

        # 创建临时轨道对象
        orbit = Orbit.from_vectors(Earth, r * u.km, v * u.km / u.s)

        # 解析传播
        propagated_orbit = orbit.propagate(step * u.s)

        # 转换回状态数组
        return np.concatenate([propagated_orbit.r.value, propagated_orbit.v.value])

    def _propagate_with_rk4(
        self, state: np.ndarray, accel: np.ndarray, step: float
    ) -> np.ndarray:
        """
        在存在外加推力时，使用基于固定步长的 RK4 数值积分方法更新轨道状态。

        积分核心使用 Numba 优化以提升执行效率。

        Args:
            state (np.ndarray): 当前状态向量 [x, y, z, vx, vy, vz]。
            accel (np.ndarray): 外加加速度（单位：km/s²）。
            step (float): 积分时间步长（秒）。

        Returns:
            np.ndarray: 传播后的状态向量。

        示例:
            >>> gen = OrbitGeneratorF()
            >>> next_state = gen._propagate_with_rk4(
            ...     state=np.array([7000, 0, 0, 0, 7.5, 0]),
            ...     accel=np.array([0.0001, 0, 0]),
            ...     step=10
            ... )
            >>> print(next_state)
        """
        # 地球引力常数（km³/s²）
        k = Earth.k.to(u.km**3 / u.s**2).value

        # 执行RK4积分
        new_state = rk4_integrate(state, k, accel, step)

        return new_state


if __name__ == "__main__":
    r0 = [4e8, 0, 0] << u.m
    v0 = [0, 1022, 0] << (u.m / u.s)
    initial = Orbit.from_vectors(Earth, r0, v0)
    generator = OrbitGeneratorF()
    res = generator(initial, None)
    # res = initial.propagate(3 << u.day, method=CowellPropagator())
    print(res)
