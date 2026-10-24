from environment.SatelliteEnv3v1Hvt import SatelliteEnv3v1Hvt


class Env3v1(SatelliteEnv3v1Hvt):
    """可配置初始扇区角度 + 扇区半径的环境"""

    def __init__(self, sector_angle_deg=360, sector_radius_km=30, **kwargs):
        self.sector_angle_deg = sector_angle_deg
        self.sector_radius_km = sector_radius_km
        super().__init__(**kwargs)

    def _get_scenario_config(self):
        config = super()._get_scenario_config()
        RHO_MIN = 20.0  # 攻击者距离 HVT 的最小距离（km）
        for group in config["satellite_groups"]:
            if group["prefix"] == "拦截者":

                group["generation_mode"] = "sector_around_axis"
                group["sector_half_angle_deg"] = self.sector_angle_deg
                group["sector_distance_range_km"] = [RHO_MIN, self.sector_radius_km]

            # if group["prefix"] == "攻击者":
            #     group["polar_theta_range_deg"] = [0, self.sector_angle_deg]
            #     group["polar_rho_range_km"] = [RHO_MIN, self.sector_radius_km]
            # elif group["prefix"] == "拦截者":
            #     group["generation_mode"] = "formation_flying"
            #     group["formation_angle_range_deg"] = [0, self.sector_angle_deg]
        return config
