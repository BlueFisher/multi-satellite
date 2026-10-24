import dataclasses

import numpy as np
import pyvista as pv

pv.global_theme.allow_empty_mesh = True


@dataclasses.dataclass
class SatelliteActorData:
    """
    用于存储单个卫星在可视化中的相关渲染数据。

    Attributes:
        satellite_actor (pv.Actor): 卫星本体的渲染对象。
        trail_actor (pv.Actor): 卫星轨迹线的渲染对象。
        trail_polydata (pv.PolyData): 轨迹的几何数据容器。

    Example:
        >>> data = SatelliteActorData(satellite_actor, trail_actor, trail_polydata)
        >>> print(data.satellite_actor)
    """
    satellite_actor: pv.Actor
    trail_actor: pv.Actor
    trail_polydata: pv.PolyData


class SatelliteAnimationWindow:
    """
    一个用于显示地球、卫星及其轨迹的 PyVista 可视化窗口。

    此类提供添加卫星、更新轨迹、显示窗口、渲染动画以及重置场景的功能。

    Attributes:
        actors (list[SatelliteActorData]): 存储所有卫星渲染相关数据。
        plotter (pv.Plotter): PyVista 绘图窗口对象。

    Example:
        >>> window = SatelliteAnimationWindow()
        >>> idx = window.add_satellite("yellow")
        >>> window.show()
    """
    def __init__(self):
        self.actors = []
        self.plotter = pv.Plotter(window_size=(800, 600))
        self._setup_scene()

    def _setup_scene(self) -> None:
        """
        初始化可视化窗口场景，设置背景、坐标轴和初始相机视角。

        Example:
            >>> window = SatelliteAnimationWindow()
            >>> window._setup_scene()
        """
        self.plotter.set_background("black")
        self.plotter.add_axes()
        self.plotter.camera.position = (-255, 0, 0)
        self.earth_actor = self.plotter.add_mesh(pv.Sphere(radius=8), color="blue")

    def add_satellite(self, color: str = "blue") -> int:
        """
        向窗口中添加一个卫星及其轨迹数据结构。

        Args:
            color (str): 卫星与其轨迹的颜色。

        Returns:
            int: 添加后的卫星索引。

        Example:
            >>> window = SatelliteAnimationWindow()
            >>> sat_id = window.add_satellite(color="red")
            >>> print(sat_id)
        """
        satellite = pv.Sphere(radius=0.0001)  # 尺寸别太小
        satellite_actor = self.plotter.add_mesh(satellite, color=color)

        trail_polydata = pv.PolyData()
        trail_actor = self.plotter.add_mesh(trail_polydata, color=color, line_width=2)

        self.actors.append(
            SatelliteActorData(
                satellite_actor=satellite_actor,
                trail_actor=trail_actor,
                trail_polydata=trail_polydata,
            )
        )

        return len(self.actors) - 1

    def update_satellite(self, index, pos, sampled_positions) -> None:
        """
        根据指定位置与轨迹点更新某个卫星的显示状态。

        Args:
            index (int): 卫星索引。
            pos (list[float]): 卫星当前位置坐标。
            sampled_positions (list[list[float]]): 轨迹点列表。

        Example:
            >>> window = SatelliteAnimationWindow()
            >>> idx = window.add_satellite()
            >>> window.update_satellite(idx, [1, 2, 3], [[0,0,0],[1,2,3]])
        """
        actor = self.actors[index]
        actor.satellite_actor.SetPosition(pos)

        if len(sampled_positions) >= 2:
            points = np.array(sampled_positions)
            lines = np.array(
                [[i, i + 1] for i in range(len(points) - 1)], dtype=np.int32
            )
            cells = np.hstack([[2, i, j] for i, j in lines])

            actor.trail_polydata.points = points
            actor.trail_polydata.lines = cells
            actor.trail_polydata.Modified()

    def show(self) -> None:
        """
        显示可视化窗口，保持交互状态。

        Example:
            >>> window = SatelliteAnimationWindow()
            >>> window.show()
        """
        self.plotter.show(auto_close=False, interactive_update=True)

    def render(self) -> None:
        """
        用于刷新当前帧渲染，常用于动画循环中。

        Example:
            >>> window = SatelliteAnimationWindow()
            >>> window.render()
        """
        self.plotter.update()

    def close(self) -> None:
        """
        关闭并销毁可视化窗口。

        Example:
            >>> window = SatelliteAnimationWindow()
            >>> window.close()
        """
        self.plotter.close()

    def reset(self):
        """
        重置场景，清除所有添加的卫星和轨迹数据。

        Example:
            >>> window = SatelliteAnimationWindow()
            >>> window.reset()
        """
        self.plotter.clear()
        self.actors.clear()
        self._setup_scene()
