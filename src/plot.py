import os

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体为黑体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示为方块的问题

_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_FILE_DIR)
_SAVE_DIR = os.path.join(_PROJECT_DIR, "save_data", "image")

if not os.path.exists(_SAVE_DIR):
    os.makedirs(_SAVE_DIR)


def plot_end(data):
    """
    绘制卫星轨迹的 3D 图形。

    参数:
        data (List[Dict]): 包含每个时间步卫星数据的列表。
            每个时间步的数据结构如下:
            {
                "data": [
                    {
                        "name": str,  # 卫星名称
                        "r": List[float],  # 卫星位置 [x, y, z]
                    },
                    ...
                ]
            }
    """
    # 如果您将数据保存在文件中，请取消注释以下两行并注释掉 json_data 字符串
    #     with open('data.json', 'r') as f:
    #         data = json.load(f)

    # data = json.loads(json_data)
    name_dict = {
        "attacker_0": "攻击者卫星",
        "interceptor_0": "拦截者卫星1",
        "interceptor_1": "拦截者卫星2",
        "interceptor_2": "拦截者卫星3",
    }
    # 1. 重组数据结构：从“按时间步”转为“按卫星名”
    # 目标结构: {'sat_name': {'x': [], 'y': [], 'z': []}, ...}
    trajectories = {}

    for step in data:
        for sat in step["data"]:
            name = name_dict[sat["name"]]
            position = sat["r"]  # [x, y, z]

            if name not in trajectories:
                trajectories[name] = {"x": [], "y": [], "z": []}

            trajectories[name]["x"].append(position[0])
            trajectories[name]["y"].append(position[1])
            trajectories[name]["z"].append(position[2])

    # 2. 创建 3D 图形
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # 3. 绘制每个卫星的轨迹
    colors = ["blue", "red", "red", "red"]  # 自动分配颜色

    for (name, coords), color in zip(trajectories.items(), colors):
        # 绘制连线
        ax.plot(
            coords["x"], coords["y"], coords["z"], label=name, color=color, linewidth=2
        )

        # 标记起点 (三角形 ^)
        ax.scatter(
            coords["x"][0],
            coords["y"][0],
            coords["z"][0],
            color=color,
            marker="^",
            s=100,
            label=f"{name} 开始点",
        )

        # 标记终点 (叉号 x)
        ax.scatter(
            coords["x"][-1],
            coords["y"][-1],
            coords["z"][-1],
            color=color,
            marker="x",
            s=100,
        )

    # 4. (可选) 绘制地球模型以增加参考感
    # 半径 6371km
    # u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    # earth_x = 6371 * np.cos(u) * np.sin(v)
    # earth_y = 6371 * np.sin(u) * np.sin(v)
    # earth_z = 6371 * np.cos(v)
    # 绘制线框地球，透明度设低一点
    # ax.plot_wireframe(earth_x, earth_y, earth_z, color="grey", alpha=0.3)

    # 5. 设置标签和图例
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.set_title("博弈轨迹 (3D)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1))

    # 6. 设置坐标轴比例一致 (防止地球看起来像椭圆)
    # 获取所有数据的最大最小值，统一坐标轴范围
    all_x = [x for t in trajectories.values() for x in t["x"]]
    all_y = [y for t in trajectories.values() for y in t["y"]]
    all_z = [z for t in trajectories.values() for z in t["z"]]

    # 计算中心点和最大跨度
    max_range = (
        np.array(
            [max(all_x) - min(all_x), max(all_y) - min(all_y), max(all_z) - min(all_z)]
        ).max()
        / 2.0
    )
    mid_x = (max(all_x) + min(all_x)) * 0.5
    mid_y = (max(all_y) + min(all_y)) * 0.5
    mid_z = (max(all_z) + min(all_z)) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    #     plt.show(block=False)
    save_path = os.path.join(_SAVE_DIR, "trajectory.png")
    plt.savefig(save_path, dpi=300)
