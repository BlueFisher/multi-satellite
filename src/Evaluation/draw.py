import os

import matplotlib.colors as mcolors
import numpy as np
from matplotlib import pyplot as plt


def solve_Co2i(X):
    """从轨道坐标系到惯性坐标系的转换矩阵"""
    x, y, z, vx, vy, vz = X
    r = np.sqrt(x**2 + y**2 + z**2)
    H = np.cross(X[0:3], X[3:6])
    ff = np.zeros((3, 3))

    ff[:, 1] = -H / np.linalg.norm(H)
    ff[:, 2] = np.array([-x / r, -y / r, -z / r])
    temp = np.cross(ff[:, 1], ff[:, 2])
    ff[:, 0] = temp / np.linalg.norm(temp)

    return ff.T


def eci2lvlh(Xi_t: np.ndarray, Xi_c: np.ndarray):
    """将ECI坐标转换为LVLH坐标系"""
    miu = 3.986004418e5  # 地球引力常数 (km^3/s^2)

    r_c = np.linalg.norm(Xi_c[0:3])
    v_c = np.linalg.norm(Xi_c[3:6])
    ener = v_c**2 / 2 - miu / r_c
    a_c = -miu / (2 * ener)
    woi_c = np.array([0, -np.sqrt(miu / a_c**3), 0])

    Coi_c = solve_Co2i(Xi_c)
    delta_Xi = Xi_t - Xi_c
    delta_Ro = Coi_c @ delta_Xi[0:3]
    delta_Vo = Coi_c @ delta_Xi[3:6] - np.cross(woi_c, delta_Ro)

    R = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])
    delta_Ro = R @ delta_Ro
    delta_Vo = R @ delta_Vo

    return np.concatenate((delta_Ro, delta_Vo))


def convert_to_lvlh(trajectory_dict, reference_sat):
    """转换所有轨迹到LVLH坐标系"""
    lvlh_dict = {}
    ref_traj = trajectory_dict[reference_sat]

    for sat_name, traj in trajectory_dict.items():
        lvlh_traj = []
        for i in range(len(traj)):
            ref_state = ref_traj[i]
            sat_state = traj[i]
            lvlh_state = eci2lvlh(sat_state, ref_state)
            lvlh_traj.append(lvlh_state[:3])  # 仅保留位置信息
        lvlh_dict[sat_name] = np.array(lvlh_traj)

    return lvlh_dict


# def plot_lvlh_trajectories(lvlh_dict, view_size=0.5, output_dir="lvlh_trajectories"):
#     """
#     简洁版LVLH轨迹绘制，专注于运动趋势展示
#
#     参数:
#     lvlh_dict: LVLH坐标系下的轨迹字典
#     view_size: 视图范围（km）
#     output_dir: 输出目录
#     """
#     # 设置中文字体
#     plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
#     plt.rcParams["axes.unicode_minus"] = False
#
#     os.makedirs(output_dir, exist_ok=True)
#
#     # 简单颜色方案
#     colors = list(mcolors.TABLEAU_COLORS.values())
#
#     # 创建图形
#     fig, ax = plt.subplots(figsize=(8, 8))
#     ax.set_title("LVLH坐标系轨迹")
#     ax.set_xlabel("径向 (km)")
#     ax.set_ylabel("沿迹 (km)")
#     ax.set_xlim(-view_size, view_size)
#     ax.set_ylim(-view_size, view_size)
#     ax.set_aspect("equal")
#
#     # 基础网格和原点
#     ax.grid(True, linestyle="--", alpha=0.7)
#     ax.plot(0, 0, "ko", markersize=6, label="参考星")
#
#     # 绘制坐标轴
#     ax.axhline(0, color="gray", linewidth=1)
#     ax.axvline(0, color="gray", linewidth=1)
#
#     # 绘制轨迹
#     for i, (name, traj) in enumerate(lvlh_dict.items()):
#         # 跳过参考星自身
#         # if np.allclose(traj, 0, atol=1e-6).any():
#         #     continue
#
#         # 提取径向和沿迹数据
#         radial = traj[:, 0]
#         along_track = traj[:, 1]
#
#         # 绘制主轨迹
#         ax.plot(
#             radial, along_track, color=colors[i % len(colors)], linewidth=2, label=name
#         )
#
#         # 标记起点和终点（简洁版）
#         ax.plot(
#             radial[0],
#             along_track[0],
#             color=colors[i % len(colors)],
#             marker="o",
#             markersize=6,
#         )
#         ax.plot(
#             radial[-1],
#             along_track[-1],
#             color=colors[i % len(colors)],
#             marker="s",
#             markersize=6,
#         )
#
#     # 简洁图例
#     ax.legend(loc="best", frameon=True, framealpha=0.9)
#
#     # 保存图像
#     plt.tight_layout()
#     save_path = os.path.join(output_dir, "lvlh_trajectory_simple.png")
#     plt.savefig(save_path, dpi=200)
#     plt.close(fig)
#     print(f"轨迹图已保存至: {save_path}")
#
#     # 额外生成法向视图（径向-法向平面）
#     fig, ax = plt.subplots(figsize=(8, 8))
#     ax.set_title("LVLH坐标系轨迹（径向-法向）")
#     ax.set_xlabel("径向 (km)")
#     ax.set_ylabel("法向 (km)")
#     ax.set_xlim(-view_size, view_size)
#     ax.set_ylim(-view_size, view_size)
#     ax.set_aspect("equal")
#
#     ax.grid(True, linestyle="--", alpha=0.7)
#     ax.plot(0, 0, "ko", markersize=6, label="参考星")
#     ax.axhline(0, color="gray", linewidth=1)
#     ax.axvline(0, color="gray", linewidth=1)
#
#     for i, (name, traj) in enumerate(lvlh_dict.items()):
#         # if np.allclose(traj, 0, atol=1e-6).any():
#         #     continue
#
#         radial = traj[:, 0]
#         cross_track = traj[:, 2]
#
#         ax.plot(
#             radial, cross_track, color=colors[i % len(colors)], linewidth=2, label=name
#         )
#
#         ax.plot(
#             radial[0],
#             cross_track[0],
#             color=colors[i % len(colors)],
#             marker="o",
#             markersize=6,
#         )
#         ax.plot(
#             radial[-1],
#             cross_track[-1],
#             color=colors[i % len(colors)],
#             marker="s",
#             markersize=6,
#         )
#
#     ax.legend(loc="best", frameon=True, framealpha=0.9)
#     plt.tight_layout()
#     save_path = os.path.join(output_dir, "lvlh_trajectory_cross.png")
#     plt.savefig(save_path, dpi=200)
#     plt.close(fig)
#     print(f"法向视图已保存至: {save_path}")


def plot_lvlh_trajectories(
    lvlh_dict,
    view_size=0.5,
    output_dir="lvlh_trajectories",
    three_d=True,
    show_z_axis=True,
):
    """
    增强版LVLH轨迹绘制，支持2D和3D视图，可选择是否显示Z轴

    参数:
    lvlh_dict: LVLH坐标系下的轨迹字典
    view_size: 视图范围（km）
    output_dir: 输出目录
    three_d: 是否绘制3D轨迹图
    show_z_axis: 在2D图中是否显示Z轴信息
    """
    # 设置中文字体
    plt.rcParams["font.family"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    os.makedirs(output_dir, exist_ok=True)

    # 简单颜色方案
    colors = list(mcolors.TABLEAU_COLORS.values())

    # 创建2D图形 - 径向和沿迹平面
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("LVLH坐标系轨迹 (径向-沿迹平面)")
    ax.set_xlabel("径向 (km)")
    ax.set_ylabel("沿迹 (km)")
    ax.set_xlim(-view_size, view_size)
    ax.set_ylim(-view_size, view_size)
    ax.set_aspect("equal")

    # 基础网格和原点
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.plot(0, 0, "ko", markersize=6, label="参考星")

    # 绘制坐标轴
    ax.axhline(0, color="gray", linewidth=1)
    ax.axvline(0, color="gray", linewidth=1)

    # 绘制轨迹
    for i, (name, traj) in enumerate(lvlh_dict.items()):
        # 提取径向和沿迹数据
        radial = traj[:, 0]
        along_track = traj[:, 1]

        # 绘制主轨迹
        ax.plot(
            radial, along_track, color=colors[i % len(colors)], linewidth=2, label=name
        )

        # 标记起点和终点
        ax.plot(
            radial[0],
            along_track[0],
            color=colors[i % len(colors)],
            marker="o",
            markersize=6,
        )
        ax.plot(
            radial[-1],
            along_track[-1],
            color=colors[i % len(colors)],
            marker="s",
            markersize=6,
        )

    # 简洁图例
    ax.legend(loc="best", frameon=True, framealpha=0.9)

    # 保存图像
    plt.tight_layout()
    save_path = os.path.join(output_dir, "lvlh_trajectory_radial_along.png")
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"径向-沿迹轨迹图已保存至: {save_path}")

    # 如果显示Z轴，创建径向-法向平面图
    if show_z_axis:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title("LVLH坐标系轨迹（径向-法向平面）")
        ax.set_xlabel("径向 (km)")
        ax.set_ylabel("法向 (km)")
        ax.set_xlim(-view_size, view_size)
        ax.set_ylim(-view_size, view_size)
        ax.set_aspect("equal")

        ax.grid(True, linestyle="--", alpha=0.7)
        ax.plot(0, 0, "ko", markersize=6, label="参考星")
        ax.axhline(0, color="gray", linewidth=1)
        ax.axvline(0, color="gray", linewidth=1)

        for i, (name, traj) in enumerate(lvlh_dict.items()):
            radial = traj[:, 0]
            cross_track = traj[:, 2]

            ax.plot(
                radial,
                cross_track,
                color=colors[i % len(colors)],
                linewidth=2,
                label=name,
            )

            ax.plot(
                radial[0],
                cross_track[0],
                color=colors[i % len(colors)],
                marker="o",
                markersize=6,
            )
            ax.plot(
                radial[-1],
                cross_track[-1],
                color=colors[i % len(colors)],
                marker="s",
                markersize=6,
            )

        ax.legend(loc="best", frameon=True, framealpha=0.9)
        plt.tight_layout()
        save_path = os.path.join(output_dir, "lvlh_trajectory_radial_cross.png")
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
        print(f"径向-法向轨迹图已保存至: {save_path}")

    # 如果启用3D绘图，创建3D轨迹图
    if three_d:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title("LVLH坐标系3D轨迹")
        ax.set_xlabel("径向 (km)")
        ax.set_ylabel("沿迹 (km)")
        ax.set_zlabel("法向 (km)")

        # 设置3D视图范围
        ax.set_xlim(-view_size, view_size)
        ax.set_ylim(-view_size, view_size)
        ax.set_zlim(-view_size, view_size)

        # 绘制参考星
        ax.scatter(0, 0, 0, color="black", s=50, label="参考星")

        # 绘制轨迹
        for i, (name, traj) in enumerate(lvlh_dict.items()):
            radial = traj[:, 0]
            along_track = traj[:, 1]
            cross_track = traj[:, 2]

            # 绘制3D轨迹
            ax.plot(
                radial,
                along_track,
                cross_track,
                color=colors[i % len(colors)],
                linewidth=2,
                label=name,
            )

            # 标记起点和终点
            ax.scatter(
                radial[0],
                along_track[0],
                cross_track[0],
                color=colors[i % len(colors)],
                marker="o",
                s=50,
            )
            ax.scatter(
                radial[-1],
                along_track[-1],
                cross_track[-1],
                color=colors[i % len(colors)],
                marker="s",
                s=50,
            )

        ax.legend(loc="best")
        plt.tight_layout()
        save_path = os.path.join(output_dir, "lvlh_trajectory_3d.png")
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
        print(f"3D轨迹图已保存至: {save_path}")


# 使用示例：
# 假设 trajectory_dict 是包含卫星轨迹的字典
# lvlh_data = convert_to_lvlh(trajectory_dict, reference_sat="参考星名称")
# plot_lvlh_trajectories(lvlh_data, view_size=1.0, plot_3d=False)  # 2D视图
# plot_lvlh_trajectories(lvlh_data, view_size=1.0, plot_3d=True)   # 3D视图
