import dataclasses
import sys

import numpy as np
import pyvista as pv

pv.global_theme.allow_empty_mesh = True


@dataclasses.dataclass
class SatelliteActorData:
    """
    用于存放单个卫星在 PyVista 中的渲染数据。

    Attributes:
        satellite_actor (pv.Actor):
            卫星球体模型的渲染 Actor，用于显示卫星本体。

        trail_actor (pv.Actor):
            卫星轨迹线的渲染 Actor。

        trail_polydata (pv.PolyData):
            存储轨迹点和线段的底层几何数据对象，用于动态更新轨迹。
    """

    satellite_actor: pv.Actor
    trail_actor: pv.Actor
    trail_polydata: pv.PolyData


class LocalSatelliteAnimationWindow:
    """
    基于 PyVista 的本地卫星可视化窗口。

    功能特性：
        - 支持显示多个卫星及其历史轨迹。
        - 支持设置某个卫星为“中心卫星”，相机将自动跟随其移动。
        - 支持动态缩放视野（上下键）。
        - 支持按下 'q' 键关闭窗口并退出程序。
        - 适用于局部尺度（如 ±几公里）下的相对位置可视化。

    Attributes:
        plotter (pv.Plotter):
            PyVista 渲染窗口对象。

        actors (list[SatelliteActorData]):
            保存所有卫星的渲染数据，包括卫星本体与轨迹。

        center_satellite_index (int):
            当前的中心卫星索引，相机将跟随该卫星移动。
            若为 -1，则不跟随任何卫星。

        view_range (float):
            相机视野范围（越大相机越远），单位为 km 的缩放量。

        initial_camera_set (bool):
            标记是否已经初始化相机视角，避免重复初始化。

        default_camera_direction (np.ndarray):
            初始相机相对卫星的方向向量，实现 45 度透视视图。
    """

    def __init__(self, center_satellite_index: int = -1):
        """
        初始化可视化窗口。

        Args:
            center_satellite_index (int):
                设置初始中心卫星（默认为 -1，不跟随任何卫星）。
        """
        self.actors: list[SatelliteActorData] = []
        self.center_satellite_index = center_satellite_index
        self.plotter = pv.Plotter(window_size=(800, 600))
        self._setup_scene()
        self.view_range = 0.6  # 初始5公里范围
        self.initial_camera_set = False
        self.default_camera_direction = np.array([1, 1, 1])  # 45度视角

    def _setup_scene(self) -> None:
        """设置渲染窗口的初始场景。"""
        self.plotter.set_background("black")
        self.plotter.add_axes()
        self.plotter.camera.up = (0, 1, 0)
        self.plotter.camera.enable_parallel_projection()

    def add_satellite(self, color: str = "blue") -> int:
        """
        向场景中添加一个新的卫星及其轨迹。

        Args:
            color: 卫星和轨迹的颜色。

        Returns:
            新添加卫星的索引。
        """
        satellite = pv.Sphere(radius=0.002)
        satellite_actor = self.plotter.add_mesh(satellite, color=color)

        trail_polydata = pv.PolyData()
        trail_actor = self.plotter.add_mesh(
            trail_polydata,
            color=color,
            line_width=2,  # 适度加宽轨迹线
            render_lines_as_tubes=True,
            style="surface",  # 确保线宽设置生效
        )

        self.actors.append(
            SatelliteActorData(
                satellite_actor=satellite_actor,
                trail_actor=trail_actor,
                trail_polydata=trail_polydata,
            )
        )

        return len(self.actors) - 1

    def update_satellite(
        self, index: int, pos: list[float], sampled_positions: list[list[float]]
    ) -> None:
        """
        更新指定卫星的位置和轨迹。

        Args:
            index: 卫星的索引。
            pos: 卫星的当前位置。
            sampled_positions: 轨迹点的列表。
        """
        actor = self.actors[index]
        actor.satellite_actor.SetPosition(pos)

        if len(sampled_positions) > 1:
            points = np.array(sampled_positions)

            # 优化点：使用NumPy向量化操作高效创建lines和cells数组，
            # 避免了低效的Python循环。
            # 正确构建pyvista期望的线条连接格式
            lines = np.arange(len(points), dtype=np.int32)
            lines = np.c_[[2] * (len(points) - 1), lines[:-1], lines[1:]]
            cells = lines.ravel()

            actor.trail_polydata.points = points
            actor.trail_polydata.lines = cells
            actor.trail_polydata.Modified()
        else:
            # 如果没有轨迹点，则清空轨迹
            actor.trail_polydata.clean()
            actor.trail_polydata.Modified()

        # 如果是中心卫星，更新相机视图
        if index == self.center_satellite_index:
            self._update_camera_view(pos)

    def _update_camera_view(self, center_pos: list[float]) -> None:
        """
        更新相机位置，使其始终跟随中心卫星。

        Args:
            center_pos (list[float]): 中心卫星的位置。
        """
        center_pos_np = np.array(center_pos)
        if not self.initial_camera_set:
            # 初始设置
            camera_pos = center_pos_np + self.default_camera_direction * self.view_range
            self.plotter.camera.position = camera_pos
            self.plotter.camera.focal_point = center_pos_np
            self.plotter.camera.parallel_scale = self.view_range * 0.6
            self.initial_camera_set = True
        else:
            # 保持当前相机相对位置
            current_vector = np.array(self.plotter.camera.position) - np.array(
                self.plotter.camera.focal_point
            )
            current_distance = np.linalg.norm(current_vector)

            direction = current_vector / current_distance
            new_position = center_pos_np + direction * current_distance

            self.plotter.camera.position = new_position.tolist()
            self.plotter.camera.focal_point = center_pos_np

    def set_view_range(self, new_range: float):
        """
        通过键盘调整相机视野范围。

        Args:
            new_range (float): 新的视野范围比例。
        """

        self.view_range = new_range
        # 优化点：只在有中心卫星时才重置相机
        if self.initial_camera_set and self.center_satellite_index >= 0:
            self.initial_camera_set = False
            center_pos = self.actors[
                self.center_satellite_index
            ].satellite_actor.position
            self._update_camera_view(center_pos)

    def set_center_satellite(self, index: int):
        """
        设置新的中心卫星，相机会自动跟随该卫星。

        Args:
            index (int): 卫星索引。
        """
        if 0 <= index < len(self.actors):
            self.center_satellite_index = index
            self.initial_camera_set = False
            # 设置后立即重置相机
            pos = self.actors[self.center_satellite_index].satellite_actor.position
            self._update_camera_view(pos)

    def _handle_quit(self):
        """处理 'q' 键按下的回调函数，用于退出程序。"""
        print("Detected 'q' key press. Closing window and exiting program...")
        self.plotter.close()
        sys.exit()

    def show(self) -> None:
        """显示渲染窗口并添加交互式键盘事件。"""
        self.plotter.show(
            auto_close=False,
            interactive_update=True,
        )

        self.plotter.add_key_event("q", self._handle_quit)

        # 添加上下键缩放功能
        self.plotter.add_key_event(
            "Up", lambda: self.set_view_range(self.view_range * 0.8)
        )
        self.plotter.add_key_event(
            "Down", lambda: self.set_view_range(self.view_range * 1.2)
        )

    def render(self) -> None:
        """渲染当前帧。"""
        self.plotter.update()

    def close(self) -> None:
        """关闭渲染窗口。"""
        self.plotter.close()

    def reset(self):
        """重置模拟器状态，清空所有演员和数据。"""
        self.plotter.clear()
        self.actors.clear()
        self.initial_camera_set = False
        self._setup_scene()
