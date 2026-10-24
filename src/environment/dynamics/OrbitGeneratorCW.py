import numpy as np
from astropy import units as u
from poliastro.bodies import Earth

from .IGenerator import IGenerator


def eci2lvlh(Xi_t: np.ndarray[float], Xi_c: np.ndarray[float]):
    """
    目标星相对于追踪星的轨道动力学（前者相对于后者）
    这是803提供的版本，但是修改了输出格式
    参数:
    Xi_t: 追踪星的状态向量 [x, y, z, vx, vy, vz]  惯性系下
    Xi_c: 目标星的状态向量 [x, y, z, vx, vy, vz]  惯性系下

    返回:
    X: 相对状态向量 [delta_Ro; delta_Vo] 经过特定旋转后的结果
    后者可以是追踪星也可以是虚拟高价值目标点，即LVLH的原点
    目标星相对于追踪星的轨道动力学（前者相对于后者）
    """

    # miu = 3.9860044e14
    miu = 3.986004418e5

    delta_Xi = Xi_t - Xi_c
    Coi_c = solve_Co2i(Xi_c)
    r_c = np.linalg.norm(Xi_c[0:3])
    v_c = np.linalg.norm(Xi_c[3:6])
    ener = v_c**2 / 2 - miu / r_c
    a_c = -miu / (2 * ener)
    woi_c = np.array([0, -np.sqrt(miu / a_c**3), 0])  # n

    delta_Ro = Coi_c @ delta_Xi[0:3]  # 矩阵相乘
    delta_Vo = Coi_c @ delta_Xi[3:6] - np.cross(woi_c, delta_Ro)
    delta_Ro = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]]) @ delta_Ro
    delta_Vo = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]]) @ delta_Vo

    return np.concatenate((delta_Ro, delta_Vo))


def lvlh2eci(X_lvlh: np.ndarray[float], Xi_c: np.ndarray[float]) -> np.ndarray[float]:
    """
    将LVLH坐标系下的相对状态转换为惯性系（ECI）下的相对状态

    参数:
    X_lvlh: LVLH坐标系下的相对状态向量 [delta_Ro; delta_Vo]
           （前3元素为相对位置，后3为相对速度）
    Xi_c: 目标星（LVLH原点）的惯性系状态向量 [x, y, z, vx, vy, vz]

    返回:
    delta_Xi: 惯性系下的相对状态向量 [delta_r; delta_v]
              （追踪星相对于目标星的惯性系位置和速度）
    """
    # 引力常数（与原函数保持一致，单位：km³/s²）
    miu = 3.986004418e5

    # 1. 解析LVLH下的相对位置和速度
    delta_Ro = X_lvlh[0:3]  # LVLH相对位置
    delta_Vo = X_lvlh[3:6]  # LVLH相对速度

    # 2. 定义旋转矩阵M及其逆（正交矩阵，逆=转置）
    M = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])
    M_inv = M.T  # M的逆矩阵

    # 3. 从LVLH位置恢复轨道系中间位置r1
    r1 = M_inv @ delta_Ro

    # 4. 从LVLH速度恢复轨道系中间速度v1
    v1 = M_inv @ delta_Vo

    # 5. 计算目标星的轨道参数和转换矩阵
    Coi_c = solve_Co2i(Xi_c)  # 轨道系→惯性系转换矩阵
    r_c = np.linalg.norm(Xi_c[0:3])  # 目标星位置模长
    v_c = np.linalg.norm(Xi_c[3:6])  # 目标星速度模长
    ener = v_c**2 / 2 - miu / r_c  # 比机械能
    a_c = -miu / (2 * ener)  # 半长轴
    woi_c = np.array([0, -np.sqrt(miu / a_c**3), 0])  # 轨道角速度向量

    # 6. 计算惯性系相对位置delta_r
    delta_r = Coi_c.T @ r1  # Coi_c的逆=转置，轨道系→惯性系

    # 7. 计算惯性系相对速度delta_v（修正牵连速度）
    delta_v = Coi_c.T @ (v1 + np.cross(woi_c, r1))

    eci_r = Xi_c[:3] + delta_r
    eci_v = Xi_c[3:6] + delta_v

    # 8. 拼接惯性系相对状态
    return np.concatenate((eci_r, eci_v))


def solve_Co2i(X):
    """
    从轨道坐标系到惯性坐标系的转换矩阵
    """
    x, y, z, vx, vy, vz = X

    r = np.sqrt(x**2 + y**2 + z**2)
    H = np.cross(X[0:3], X[3:6])
    ff = np.zeros((3, 3))

    # 第二列是角动量方向的单位向量
    ff[:, 1] = -H / np.linalg.norm(H)

    # 第三列是位置向量的负方向单位向量
    ff[:, 2] = np.array([-x / r, -y / r, -z / r])

    # 第一列是第二列和第三列的叉积的单位向量
    temp = np.cross(ff[:, 1], ff[:, 2])
    ff[:, 0] = temp / np.linalg.norm(temp)

    # 转置以匹配 MATLAB 的输出形式
    return ff.T


class CwSolution:
    """
    Clohessy–Wiltshire (CW) 相对运动解析解的封装类。

    该类根据给定的平均角速度 n 与传播步长 t 构造
    CW 状态转移矩阵 Phi 和加速度影响矩阵 Psi，用于线性化轨道动力学下的
    相对位置与速度更新。

    Example:
        >>> n = 0.0011
        >>> t = 10
        >>> cw = CwSolution(n, t)
        >>> state = np.array([10, 0, 0, 0, 0.1, 0])
        >>> accel = np.array([0, 0, 0])
        >>> next_state = cw.update(state, accel)
    """

    def __init__(self, n, t):
        self.Phi = np.array(
            [
                [
                    4 - 3 * np.cos(n * t),
                    0,
                    0,
                    np.sin(n * t) / n,
                    (2 - 2 * np.cos(n * t)) / n,
                    0,
                ],
                [
                    6 * (np.sin(n * t) - n * t),
                    1,
                    0,
                    (2 * np.cos(n * t) - 2) / n,
                    (4 * np.sin(n * t) / n) - 3 * t,
                    0,
                ],
                [0, 0, np.cos(n * t), 0, 0, np.sin(n * t) / n],
                [3 * n * np.sin(n * t), 0, 0, np.cos(n * t), 2 * np.sin(n * t), 0],
                [
                    6 * n * (np.cos(n * t) - 1),
                    0,
                    0,
                    -2 * np.sin(n * t),
                    4 * np.cos(n * t) - 3,
                    0,
                ],
                [0, 0, -n * np.sin(n * t), 0, 0, np.cos(n * t)],
            ]
        )
        self.Psi = np.array(
            [
                [
                    (1 - np.cos(n * t)) / (n * n),
                    2 * (n * t - np.sin(n * t)) / (n * n),
                    0,
                ],
                [
                    2 * (np.sin(n * t) - n * t) / (n * n),
                    4 * (1 - np.cos(n * t)) / (n * n) - 3 * t * t / 2,
                    0,
                ],
                [0, 0, (1 - np.cos(n * t)) / (n * n)],
                [np.sin(n * t) / n, 2 * (1 - np.cos(n * t)) / n, 0],
                [2 * (np.cos(n * t) - 1) / n, 4 * np.sin(n * t) / n - 3 * t, 0],
                [0, 0, np.sin(n * t) / n],
            ]
        )

    def update(self, state, accel):
        """
        使用 CW 状态转移方程更新相对运动状态。

        该方法使用 Phi 和 Psi 矩阵，根据输入的状态与加速度，
        输出在当前时间步后的新相对状态。

        Args:
            state (np.ndarray): 相对运动状态向量 [rx, ry, rz, vx, vy, vz]。
            accel (np.ndarray): 相对坐标系下的加速度向量。

        Returns:
            np.ndarray: 更新后的 6 维相对状态向量。

        Example:
            >>> cw = CwSolution(0.0011, 10)
            >>> state = np.zeros(6)
            >>> accel = np.zeros(3)
            >>> new_state = cw.update(state, accel)
        """

        state.reshape(6, 1)
        accel.reshape(3, 1)
        return self.Phi @ state + self.Psi @ accel


class OrbitGeneratorCW(IGenerator):
    """
    使用 Clohessy–Wiltshire (CW) 方程进行目标卫星相对运动更新的生成器。

    该类依赖参考轨道（origin_state），并基于 CW 线性化模型，利用
    参考轨道平均角速度 n，对目标卫星的相对位置和速度进行传播。
    常用于编队飞行、相对导航、交会对接等场景。

    Example:
        >>> gen = OrbitGeneratorCW()
        >>> origin = np.array([7000, 0, 0, 0, 7.5, 0])
        >>> gen.set_origin(origin)
        >>> state = np.array([10, 0, 0, 0, 0.1, 0])
        >>> accel = np.zeros(3)
        >>> next_state = gen(state, accel, step=10)
    """

    def __init__(self):
        super().__init__()
        self.origin_state: np.ndarray = None  # 存储参考轨道状态数组
        self.cw_solution: CwSolution = None
        self.n: float = 0.0  # 参考轨道平均角速度 (rad/s)

    def set_origin(self, origin_state: np.ndarray[(6,), np.float64]) -> None:
        """
        设置参考轨道状态，并依据该状态计算 CW 模型所需的平均角速度 n。

        该方法需要在进行任何传播前调用，用于初始化 CW 解算器，
        并将参考轨道存储起来，以便后续相对运动计算。

        Args:
            origin_state (np.ndarray): 参考轨道在 ECI 坐标系下的状态向量。

        Example:
            >>> gen = OrbitGeneratorCW()
            >>> origin = np.array([7000, 0, 0, 0, 7.5, 1.0])
            >>> gen.set_origin(origin)
        """

        self.origin_state = origin_state
        # 从参考状态计算平均角速度n (rad/s)
        # 1. 提取参考轨道位置
        r_origin = origin_state[:3]
        # 2. 计算轨道半径大小
        r_mag = np.linalg.norm(r_origin)
        # 3. 计算平均角速度 (n = sqrt(μ/r³), μ为地球引力常数 km³/s²)
        self.n = np.sqrt(Earth.k.to(u.km**3 / u.s**2).value / (r_mag**3))
        # 重置CW解算器
        self.cw_solution = None

    def __call__(
        self,
        state: np.ndarray[
            (6,), np.float64
        ],  # 目标卫星当前状态 [r_x, r_y, r_z, v_x, v_y, v_z] (km, km/s)
        accel: np.ndarray[(3,), np.float64],  # LVLH坐标系下的加速度 (km/s²)
        step: float = 10,  # 时间步长 (s)
    ) -> np.ndarray[(6,), np.float64]:
        """
        使用 CW 方程推进目标卫星在相对坐标系下的一个时间步状态。

        该方法基于已设定的参考轨道 origin_state，通过 CW 线性动力学模型
        更新目标卫星的相对位置与速度。

        Args:
            state (np.ndarray): 目标卫星当前相对状态向量。
            accel (np.ndarray): 相对坐标系下的加速度输入。
            step (float): 推进的时间步长（单位：秒）。

        Returns:
            np.ndarray: 更新后的相对状态向量。

        Example:
            >>> gen = OrbitGeneratorCW()
            >>> gen.set_origin(np.array([7000, 0, 0, 0, 7.5, 0]))
            >>> state = np.array([10, 0, 0, 0, 0.1, 0])
            >>> accel = np.zeros(3)
            >>> new_state = gen(state, accel, step=10)
        """

        # 初始化检查：确保已设置参考轨道
        if self.origin_state is None:
            raise ValueError("请先通过set_origin()设置参考轨道状态")

        # 初始化CW解算器
        if self.cw_solution is None:
            self.cw_solution = CwSolution(self.n, step)

        # 2. 使用CW方程更新LVLH坐标系下的状态
        new_lvlh_state = self.cw_solution.update(state, accel)

        return new_lvlh_state
