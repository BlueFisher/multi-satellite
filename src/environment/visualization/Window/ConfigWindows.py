import sys

import numpy as np
import pyvista as pv
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QListWidget,
    QListWidgetItem,
    QGroupBox,
    QFormLayout,
    QLineEdit,
)
from poliastro.bodies import Earth
from pyvistaqt import QtInteractor


class Satellite:
    """
    卫星类，存储卫星的轨道参数和位置，并支持更新轨道参数。

    属性:
        name: 卫星名称
        a: 半长轴 (km)
        e: 偏心率
        i: 轨道倾角 (deg)
        Omega: 升交点赤经 (deg)
        omega: 近地点幅角 (deg)
        M: 平近点角 (deg)
        position: 卫星位置 (x, y, z)
        actor: PyVista用于显示的actor对象
    """

    def __init__(
        self, name, a=7000, e=0.1, i=0, Omega=0, omega=0, M=0, position=(0, 0, 0)
    ):
        """
        初始化卫星对象。

        Args:
            name: 卫星名称
            a: 半长轴 (km)，默认7000
            e: 偏心率，默认0.1
            i: 轨道倾角 (deg)，默认0
            Omega: 升交点赤经 (deg)，默认0
            omega: 近地点幅角 (deg)，默认0
            M: 平近点角 (deg)，默认0
            position: 卫星位置 (x, y, z)，默认(0,0,0)

        示例:
            >>> sat = Satellite("Sat1", a=7000, e=0.1)
        """
        self.name = name
        self.a = a  # 半长轴 (km)
        self.e = e  # 偏心率
        self.i = i  # 轨道倾角 (deg)
        self.Omega = Omega  # 升交点赤经 (deg)
        self.omega = omega  # 近地点幅角 (deg)
        self.M = M  # 平近点角 (deg)
        self.position = position  # 卫星位置 (x, y, z)
        self.actor = None  # PyVista的actor对象

    def update_orbit_params(self, params):
        """
        更新卫星的轨道参数。

        Args:
            params: 字典，包含可能的轨道参数键值，如 "a", "e", "i", "Omega", "omega", "M"

        示例:
            >>> sat.update_orbit_params({"a":7100, "i":45})
        """
        self.a = float(params.get("a", self.a))
        self.e = float(params.get("e", self.e))
        self.i = float(params.get("i", self.i))
        self.Omega = float(params.get("Omega", self.Omega))
        self.omega = float(params.get("omega", self.omega))
        self.M = float(params.get("M", self.M))


class SatelliteConfigWindow(QMainWindow):
    """
    卫星轨道配置主窗口，包含3D视图和卫星参数控制面板，支持卫星添加、删除、选中和参数/位置更新。

    属性:
        calculate_orbit: 可选的轨道计算回调函数
        satellites: 存储所有卫星对象的字典
        selected_satellite: 当前选中的卫星
    """

    def __init__(self, calculate_orbit_callback=None):
        """
        初始化卫星配置窗口，创建3D视图、控制面板和地球模型。

        Args:
            calculate_orbit_callback: 可选函数，用于根据轨道参数计算卫星位置

        示例:
            >>> window = SatelliteConfigWindow(calculate_orbit_callback=my_orbit_func)
        """
        super().__init__()
        self.setWindowTitle("卫星轨道配置器")
        self.resize(1200, 800)

        # 轨道计算回调函数
        self.calculate_orbit = calculate_orbit_callback

        # 创建中心部件和布局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # 创建3D视图
        self.create_3d_view()

        # 创建控制面板
        self.create_control_panel()

        # 卫星列表和当前选中的卫星
        self.satellites = {}
        self.selected_satellite = None

        # 拖拽相关状态
        self.is_dragging = False
        self.dragging_satellite = None
        self.last_mouse_pos = None

        # 初始化地球
        self.create_earth()

        # 设置自定义鼠标交互
        self.setup_custom_interactor()

    def create_3d_view(self):
        """
        创建3D视图区域并添加到主布局中。

        示例:
            >>> window.create_3d_view()
        """
        self.plotter = QtInteractor(self)

        self.main_layout.addWidget(self.plotter.interactor)

    def setup_custom_interactor(self):
        """
        设置自定义鼠标交互逻辑。

        示例:
            >>> window.setup_custom_interactor()
        """
        pass

        self.plotter.pick_mouse_position()
        # """设置自定义鼠标交互样式"""
        # # 获取交互器样式
        # self.iren_style = self.plotter.iren.get_interactor_style()
        #
        # # 保存原始的鼠标事件处理函数
        # self.orig_left_button_press = self.iren_style.OnLeftButtonDown
        # self.orig_left_button_release = self.iren_style.OnLeftButtonUp
        # self.orig_mouse_move = self.iren_style.OnMouseMove
        #
        # # 替换为自定义处理函数
        # self.iren_style.OnLeftButtonDown = self.on_left_button_down
        # self.iren_style.OnLeftButtonUp = self.on_left_button_up
        # self.iren_style.OnMouseMove = self.on_mouse_move
        # mouse = pv._vtk.vtkInteractorStyleTrackballCamera()
        # mouse.SetDefaultRenderer(self.plotter.ren_win.GetRenderers().GetFirstRenderer())
        # self.plotter.SetInteractorStyle(mouse)

        # def clicked(m, event):
        #     try:
        #         modifiers = QApplication.keyboardModifiers()
        #         if modifiers == QtCore.Qt.ControlModifier:
        #             pass
        #         else:
        #             clickPos = m.GetInteractor().GetEventPosition()
        #             picker = pv._vtk.vtkPointPicker()
        #             picker.Pick(clickPos[0], clickPos[1], 0, m.GetDefaultRenderer())
        #             actor = picker.GetActor()
        #             if actor != None:
        #                 print(actor)
        #             m.OnLeftButtonDown()
        #     except Exception as e:
        #         print(e)
        #
        # mouse.AddObserver("LeftButtonPressEvent", clicked)

    def create_control_panel(self):
        """
        创建控制面板，包括卫星列表、轨道参数编辑和位置编辑区域。

        示例:
            >>> window.create_control_panel()
        """
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_panel.setMaximumWidth(400)
        self.main_layout.addWidget(control_panel)

        # 卫星列表区域
        satellite_list_group = QGroupBox("卫星列表")
        satellite_list_layout = QVBoxLayout(satellite_list_group)

        self.satellite_list = QListWidget()
        self.satellite_list.itemClicked.connect(self.on_satellite_selected)
        satellite_list_layout.addWidget(self.satellite_list)

        buttons_layout = QHBoxLayout()
        self.add_satellite_btn = QPushButton("添加卫星")
        self.add_satellite_btn.clicked.connect(self.on_add_satellite)
        self.remove_satellite_btn = QPushButton("删除卫星")
        self.remove_satellite_btn.clicked.connect(self.on_remove_satellite)
        buttons_layout.addWidget(self.add_satellite_btn)
        buttons_layout.addWidget(self.remove_satellite_btn)
        satellite_list_layout.addLayout(buttons_layout)

        control_layout.addWidget(satellite_list_group)

        # 卫星参数编辑区域
        self.satellite_params_group = QGroupBox("卫星轨道参数")
        params_layout = QFormLayout(self.satellite_params_group)

        self.param_edits = {}
        param_names = [
            ("a", "半长轴 (km)"),
            ("e", "偏心率"),
            ("i", "轨道倾角 (deg)"),
            ("Omega", "升交点赤经 (deg)"),
            ("omega", "近地点幅角 (deg)"),
            ("M", "平近点角 (deg)"),
        ]

        for param_key, param_label in param_names:
            edit = QLineEdit()
            edit.setReadOnly(True)
            params_layout.addRow(param_label, edit)
            self.param_edits[param_key] = edit

        self.update_params_btn = QPushButton("更新参数")
        self.update_params_btn.clicked.connect(self.on_update_params)
        self.update_params_btn.setEnabled(False)
        params_layout.addRow(self.update_params_btn)

        control_layout.addWidget(self.satellite_params_group)

        # 位置编辑区域
        self.position_edit_group = QGroupBox("卫星位置 (x, y, z)")
        position_layout = QFormLayout(self.position_edit_group)

        self.position_edits = {}
        for axis in ["x", "y", "z"]:
            edit = QLineEdit()
            edit.setReadOnly(True)
            position_layout.addRow(axis.upper(), edit)
            self.position_edits[axis] = edit

        self.update_position_btn = QPushButton("更新位置")
        self.update_position_btn.clicked.connect(self.on_update_position)
        self.update_position_btn.setEnabled(False)
        position_layout.addRow(self.update_position_btn)

        control_layout.addWidget(self.position_edit_group)

    def create_earth(self):
        """
        在3D视图中创建地球模型。

        示例:
            >>> window.create_earth()
        """
        earth_radius = 6371  # 地球半径 (km)
        earth = pv.Sphere(radius=earth_radius, center=(0, 0, 0))
        self.earth_actor = self.plotter.add_mesh(
            earth, color="blue", smooth_shading=True
        )
        self.plotter.add_axes()
        self.plotter.set_background("black")

    def add_satellite(self, name="Satellite"):
        """
        添加卫星到场景和列表，并生成默认位置。

        Args:
            name: 卫星名称前缀，自动生成唯一名称

        Returns:
            Satellite: 新创建的卫星对象

        示例:
            >>> sat = window.add_satellite("Sat")
        """
        # 生成唯一名称
        count = 1
        unique_name = name
        while unique_name in self.satellites:
            unique_name = f"{name}{count}"
            count += 1

        # 默认位置在地球轨道上
        default_position = (7000, 0, 0)

        # 创建卫星
        satellite = Satellite(unique_name, position=default_position)
        self.satellites[unique_name] = satellite

        # 在列表中显示
        item = QListWidgetItem(unique_name)
        self.satellite_list.addItem(item)

        # 显示卫星
        self.update_satellite_display(satellite)

        return satellite

    def update_satellite_display(self, satellite):
        """
        更新指定卫星在3D视图中的显示位置。

        Args:
            satellite: 要更新显示的卫星对象

        示例:
            >>> window.update_satellite_display(sat)
        """
        x, y, z = satellite.position

        if satellite.actor is None:
            satellite_mesh = pv.Sphere(radius=100, center=(x, y, z))
            satellite.actor = self.plotter.add_mesh(satellite_mesh, color="red")
        else:
            satellite.actor.SetPosition(x, y, z)

        self.plotter.update()

    def on_satellite_selected(self, item):
        """
        处理卫星列表选中事件，填充参数和位置编辑框，并高亮选中卫星。

        Args:
            item: QListWidgetItem对象

        示例:
            >>> window.on_satellite_selected(list_item)
        """
        satellite_name = item.text()
        self.selected_satellite = self.satellites.get(satellite_name)

        if self.selected_satellite:
            # 填充参数编辑框
            self.param_edits["a"].setText(str(self.selected_satellite.a))
            self.param_edits["e"].setText(str(self.selected_satellite.e))
            self.param_edits["i"].setText(str(self.selected_satellite.i))
            self.param_edits["Omega"].setText(str(self.selected_satellite.Omega))
            self.param_edits["omega"].setText(str(self.selected_satellite.omega))
            self.param_edits["M"].setText(str(self.selected_satellite.M))

            # 填充位置编辑框
            x, y, z = self.selected_satellite.position
            self.position_edits["x"].setText(str(x))
            self.position_edits["y"].setText(str(y))
            self.position_edits["z"].setText(str(z))

            # 启用编辑和更新按钮
            for edit in self.param_edits.values():
                edit.setReadOnly(False)
            for edit in self.position_edits.values():
                edit.setReadOnly(False)
            self.update_params_btn.setEnabled(True)
            self.update_position_btn.setEnabled(True)

            # 高亮显示选中的卫星
            for name, sat in self.satellites.items():
                if sat == self.selected_satellite:
                    sat.actor.GetProperty().SetColor((1.0, 0.0, 0.0))  # 红色
                else:
                    sat.actor.GetProperty().SetColor((0.5, 0.5, 0.5))  # 灰色
            self.plotter.update()

    def on_add_satellite(self):
        """
        处理添加卫星按钮点击事件，调用add_satellite方法。

        示例:
            >>> window.on_add_satellite()
        """
        self.add_satellite()

    def on_remove_satellite(self):
        """
        处理删除卫星按钮点击事件，从3D视图、列表和字典中移除选中卫星。

        示例:
            >>> window.on_remove_satellite()
        """
        if self.selected_satellite:
            # 从场景中移除
            if self.selected_satellite.actor:
                self.plotter.remove_actor(self.selected_satellite.actor)

            # 从列表中移除
            for i in range(self.satellite_list.count()):
                if self.satellite_list.item(i).text() == self.selected_satellite.name:
                    self.satellite_list.takeItem(i)
                    break

            # 从字典中移除
            del self.satellites[self.selected_satellite.name]

            # 重置选中状态
            self.selected_satellite = None
            for edit in list(self.param_edits.values()) + list(
                self.position_edits.values()
            ):
                edit.clear()
                edit.setReadOnly(True)
            self.update_params_btn.setEnabled(False)
            self.update_position_btn.setEnabled(False)

            self.plotter.update()

    def on_update_params(self):
        """
        处理更新卫星轨道参数按钮点击事件，并调用轨道计算函数更新位置。

        示例:
            >>> window.on_update_params()
        """
        if self.selected_satellite:
            # 获取参数值
            params = {
                "a": self.param_edits["a"].text(),
                "e": self.param_edits["e"].text(),
                "i": self.param_edits["i"].text(),
                "Omega": self.param_edits["Omega"].text(),
                "omega": self.param_edits["omega"].text(),
                "M": self.param_edits["M"].text(),
            }

            # 更新卫星参数
            self.selected_satellite.update_orbit_params(params)

            # 如果提供了轨道计算函数，则使用它计算新位置
            if self.calculate_orbit:
                position = self.calculate_orbit(
                    self.selected_satellite.a,
                    self.selected_satellite.e,
                    self.selected_satellite.i,
                    self.selected_satellite.Omega,
                    self.selected_satellite.omega,
                    self.selected_satellite.M,
                )
                self.selected_satellite.position = position

                # 更新位置显示
                x, y, z = position
                self.position_edits["x"].setText(str(x))
                self.position_edits["y"].setText(str(y))
                self.position_edits["z"].setText(str(z))

            # 更新卫星显示
            self.update_satellite_display(self.selected_satellite)

    def on_update_position(self):
        """
        处理更新卫星位置按钮点击事件，将编辑框中的值更新到卫星位置。

        示例:
            >>> window.on_update_position()
        """

        if self.selected_satellite:
            try:
                # 获取位置值
                x = float(self.position_edits["x"].text())
                y = float(self.position_edits["y"].text())
                z = float(self.position_edits["z"].text())

                # 更新卫星位置
                self.selected_satellite.position = (x, y, z)

                # 更新卫星显示
                self.update_satellite_display(self.selected_satellite)

            except ValueError:
                print("位置值必须是数字")

    def on_left_button_down(self):
        """
        左键按下事件处理，支持卫星选中和拖拽开始。

        示例:
            >>> window.on_left_button_down()
        """

        # 获取当前鼠标位置
        iren = self.plotter.iren
        x, y = iren.GetEventPosition()

        # 进行拾取操作
        self.plotter.enable_point_picking(show_message=False, point_size=10)
        self.plotter.pick_mouse_position()

        # 获取拾取的actor
        picked_actor = self.plotter.actor

        # 检查是否拾取到卫星
        if picked_actor is not None:
            for satellite in self.satellites.values():
                if satellite.actor == picked_actor:
                    self.is_dragging = True
                    self.dragging_satellite = satellite
                    self.last_mouse_pos = (x, y)

                    # 如果点击的卫星不是当前选中的卫星，则选中它
                    if satellite != self.selected_satellite:
                        for i in range(self.satellite_list.count()):
                            if self.satellite_list.item(i).text() == satellite.name:
                                self.satellite_list.setCurrentRow(i)
                                break
                    break

        # 调用原始的左键按下处理
        self.orig_left_button_press()

    def on_left_button_up(self):
        """
        左键释放事件处理，结束卫星拖拽。

        示例:
            >>> window.on_left_button_up()
        """

        self.is_dragging = False
        self.dragging_satellite = None
        self.last_mouse_pos = None

        # 调用原始的左键释放处理
        self.orig_left_button_release()

    def on_mouse_move(self):
        """
        鼠标移动事件处理，支持拖拽卫星更新位置。

        示例:
            >>> window.on_mouse_move()
        """

        # 如果正在拖拽卫星
        if self.is_dragging and self.dragging_satellite:
            iren = self.plotter.iren
            x, y = iren.GetEventPosition()

            if self.last_mouse_pos:
                dx = x - self.last_mouse_pos[0]
                dy = y - self.last_mouse_pos[1]

                # 将屏幕坐标变化转换为世界坐标变化
                # 使用相机的变换矩阵来近似计算
                camera = self.plotter.camera
                focal_point = camera.GetFocalPoint()
                position = camera.GetPosition()
                view_direction = [
                    position[0] - focal_point[0],
                    position[1] - focal_point[1],
                    position[2] - focal_point[2],
                ]

                # 归一化视图方向
                norm = np.sqrt(sum([v**2 for v in view_direction]))
                view_direction = [v / norm for v in view_direction]

                # 获取相机的向上方向和向右方向
                up = camera.GetViewUp()
                right = np.cross(view_direction, up)
                right = [r / np.sqrt(sum([r**2 for r in right])) for r in right]

                # 根据鼠标移动计算世界坐标的变化
                # 这里使用一个缩放因子来控制拖拽灵敏度
                scale = 50.0
                world_dx = [r * dx * scale for r in right]
                world_dy = [u * dy * scale for u in up]

                # 计算总位移
                displacement = [
                    world_dx[0] + world_dy[0],
                    world_dx[1] + world_dy[1],
                    world_dx[2] + world_dy[2],
                ]

                # 更新卫星位置
                x, y, z = self.dragging_satellite.position
                new_x = x + displacement[0]
                new_y = y + displacement[1]
                new_z = z + displacement[2]

                self.dragging_satellite.position = (new_x, new_y, new_z)
                self.update_satellite_display(self.dragging_satellite)

                # 更新位置编辑框
                self.position_edits["x"].setText(str(new_x))
                self.position_edits["y"].setText(str(new_y))
                self.position_edits["z"].setText(str(new_z))

                # 更新最后鼠标位置
                self.last_mouse_pos = (x, y)

        # 调用原始的鼠标移动处理
        self.orig_mouse_move()


class SatelliteSimulator:
    """
    卫星模拟器类，封装PyQt窗口的显示和关闭功能。

    属性:
        app: QApplication对象
        window: SatelliteConfigWindow对象
    """


    def __init__(self, calculate_orbit_callback=None):
        """
        初始化卫星模拟器。

        Args:
            calculate_orbit_callback: 可选函数，用于计算轨道位置

        示例:
            >>> sim = SatelliteSimulator(calculate_orbit)
        """

        self.app = QApplication(sys.argv)
        self.window = SatelliteConfigWindow(calculate_orbit_callback)

    def show(self):
        """
        显示卫星配置窗口并启动事件循环。

        Returns:
            int: QApplication.exec_()返回值

        示例:
            >>> sim.show()
        """

        self.window.show()
        return self.app.exec_()

    def close(self):
        """
        关闭卫星配置窗口。

        示例:
            >>> sim.close()
        """

        self.window.close()


# 使用示例
if __name__ == "__main__":
    # 这里应该使用你的轨道计算函数
    def calculate_orbit(a, e, i, Omega, omega, M):
        # 这是一个占位函数，实际使用时应替换为你的轨道计算实现
        # 返回值应为 (x, y, z) 元组
        from poliastro.twobody import Orbit
        from astropy import units as u

        return tuple(
            Orbit.from_classical(
                Earth,
                a * u.km,
                e * u.one,
                i * u.deg,
                Omega * u.deg,
                omega * u.deg,
                M * u.deg,
            ).r.value.tolist()
        )

    simulator = SatelliteSimulator(calculate_orbit)
    simulator.show()
