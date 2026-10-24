from src.environment.SatelliteEnv2v2 import SatelliteEnv2v2


class Env2v2(SatelliteEnv2v2):
    """2v2: 可配置初始扇区角度 + 扇区半径的评估专用环境"""

    def __init__(self, sector_angle_deg=360, sector_radius_km=30, **kwargs):
        self.sector_angle_deg = sector_angle_deg
        self.sector_radius_km = sector_radius_km
        super().__init__(**kwargs)

    def _get_scenario_config(self):
        config = super()._get_scenario_config()

        # 假设 2v2 中也是 20km 作为最近距离，你可以根据实际需求调整
        RHO_MIN = 20.0

        # 遍历配置字典，修改双方出生位置
        for group in config["satellite_groups"]:

            if group["prefix"] == "拦截者":
                group["generation_mode"] = "sector_around_axis"
                group["sector_half_angle_deg"] = self.sector_angle_deg
                group["sector_distance_range_km"] = [RHO_MIN, self.sector_radius_km]

            # 如果你们的 2v2 环境里敌人的前缀也叫 "攻击者" (或者 "目标")
            # if group["prefix"] == "攻击者":
            #     group["polar_theta_range_deg"] = [0, self.sector_angle_deg]
            #     group["polar_rho_range_km"] = [RHO_MIN, self.sector_radius_km]
            #
            # # 如果己方的前缀叫 "拦截者"
            # elif group["prefix"] == "拦截者":
            #     # 让己方两个智能体也根据扇区角度编队生成（如果基础环境支持编队生成的话）
            #     group["generation_mode"] = "formation_flying"
            #     group["formation_angle_range_deg"] = [0, self.sector_angle_deg]

        return config
