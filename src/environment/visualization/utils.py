import numpy as np


def eci2lvlh(Xi_t: np.ndarray[(6,), np.float64], Xi_c: np.ndarray[(6,), np.float64]):
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


def lvlh2eci(
    X_lvlh: np.ndarray[(6,), np.float64], Xi_c: np.ndarray[(6,), np.float64]
) -> np.ndarray[(6,), np.float64]:
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
    计算轨道坐标系（LVLH/RTN）到惯性坐标系（ECI）的方向余弦矩阵（DCM）。

    本函数根据位置向量与速度向量构建轨道坐标系的三个正交单位轴，
    并返回从轨道坐标系转换到惯性坐标系的方向矩阵。

    Args:
        X (np.ndarray): 
            包含卫星在惯性系下的位置与速度的 6 元向量，
            格式为 `[x, y, z, vx, vy, vz]`。

    Returns:
        np.ndarray: 
            形状为 `(3, 3)` 的方向余弦矩阵，用于将向量从轨道坐标系转换到惯性坐标系。

    Example:
        >>> import numpy as np
        >>> X = np.array([7000, 0, 0, 0, 7.5, 1.0])
        >>> C_o2i = solve_Co2i(X)
        >>> print(C_o2i)
        # 输出为 3x3 转换矩阵，可用于：
        # v_eci = C_o2i @ v_lvlh
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
