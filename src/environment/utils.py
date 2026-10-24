import numpy as np
from astropy import units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit

from .utils_harbin import create_scenario_from_harbin_config


def _get_lvlh_to_eci_rotation_matrix(r_eci_vec, v_eci_vec):
    """
    一个辅助函数，用于计算从LVLH到ECI的旋转矩阵。
    输入为无单位的numpy数组。
    """
    r_norm = np.linalg.norm(r_eci_vec)
    if r_norm == 0:
        raise ValueError("位置矢量模长不能为零。")
    r_u = r_eci_vec / r_norm

    h_vec = np.cross(r_eci_vec, v_eci_vec)
    h_norm = np.linalg.norm(h_vec)
    if h_norm == 0:
        raise ValueError("角动量模长不能为零，可能r和v共线。")
    n_u = h_vec / h_norm

    t_u = np.cross(n_u, r_u)

    # 列向量分别是 R, T, N 在ECI中的表示
    return np.stack([r_u, t_u, n_u], axis=1)


def func_natural_ellipse_formation(b, phase, a_orbital):
    """
    自然椭圆绕飞函数（Python版本）
    基于MATLAB代码 func_ziran_raofei1 转换而来

    参数:
        b: 椭圆短半轴 (km)
        phase: 相位角 (弧度)
        a_orbital: 目标卫星的轨道半长轴 (km)

    返回:
        X: 6维状态向量 [Rx, Ry, Rz, Vx, Vy, Vz] (LVLH坐标系)
           位置单位: km
           速度单位: km/s
    """
    # 地球引力常数 (m^3/s^2 转换为 km^3/s^2)
    mu = 3.986005e5  # km^3/s^2

    # 轨道角速度
    n = np.sqrt(mu / (a_orbital**3))

    # 位置 (LVLH坐标系, km)
    Rx = b * np.sin(phase)
    Ry = 2 * b * np.cos(phase)
    Rz = 0

    # 速度 (LVLH坐标系, km/s)
    Vx = n * b * np.cos(phase)
    Vy = -2 * n * b * np.sin(phase)
    Vz = 0

    X = np.array([Rx, Ry, Rz, Vx, Vy, Vz])

    return X


import numpy as np
import astropy.units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit


def create_scenario(simulator, config):
    """
    创建2v2和3V1场景，新版本要求：
    1. 敌方敌方卫星：极坐标 ρ∈[20,30]km, θ∈[0,360°] 随机生成
    2. 我方n颗拦截卫星：支持两种模式
       - 模式1：自然椭圆绕飞模式
       - 模式2：沿着HVT运动方向伴飞，速度方向与HVT一致
    3. 所有卫星在同一轨道平面，但不在同一轨道上
    4. 我方卫星与敌方卫星的距离在观测阈值内（默认50km）
    """
    if config.get("is_harbin") is not None:
        return create_scenario_from_harbin_config(simulator, config)

    # 1. 加载所有卫星组的配置
    satellite_groups = config.get("satellite_groups", [])
    hvt_config = next((g for g in satellite_groups if g["group_name"] == "hvt"), None)
    attack_config = next(
        (g for g in satellite_groups if g["group_name"] == "attackers"), None
    )
    interceptor_config = next(
        (g for g in satellite_groups if g["group_name"] == "interceptors"), None
    )

    if not all([hvt_config, attack_config, interceptor_config]):
        raise ValueError(
            "配置文件必须包含 'HVT', 'Red_Interceptor', 和 'Blue_Attacker' 的分组。"
        )
    observation_threshold = config.get("observation_threshold_km", 50)

    # --- 步骤 1: 创建HVT并锁定轨道平面 ---
    hvt_state, hvt_a_val, rot_lvlh_to_eci = _create_hvt(simulator, hvt_config)

    # --- 步骤 2: 生成敌方攻击卫星 ---
    generated_attacker_positions = _generate_attackers(
        simulator, attack_config, hvt_state, rot_lvlh_to_eci, observation_threshold
    )

    # --- 步骤 3: 创建我方拦截卫星（根据模式选择） ---
    interceptor_count = _generate_interceptors(
        simulator,
        interceptor_config,
        hvt_state,
        hvt_a_val,
        rot_lvlh_to_eci,
        generated_attacker_positions,
        observation_threshold,
    )

    attacker_count = len(generated_attacker_positions)
    return interceptor_count, attacker_count


def _create_hvt(simulator, hvt_config):
    """生成 HVT 卫星，设置 simulator 原点，返回状态、半长轴和 LVLH→ECI 旋转矩阵"""
    hvt_ranges = hvt_config["ranges"]
    hvt_a = np.random.uniform(hvt_ranges["a"][0], hvt_ranges["a"][1]) * u.km
    hvt_ecc = np.random.uniform(hvt_ranges["ecc"][0], hvt_ranges["ecc"][1]) * u.one
    shared_inc = np.random.uniform(hvt_ranges["inc"][0], hvt_ranges["inc"][1]) * u.deg
    shared_raan = (
        np.random.uniform(hvt_ranges["raan"][0], hvt_ranges["raan"][1]) * u.deg
    )
    hvt_argp = np.random.uniform(hvt_ranges["argp"][0], hvt_ranges["argp"][1]) * u.deg
    hvt_nu = np.random.uniform(hvt_ranges["nu"][0], hvt_ranges["nu"][1]) * u.deg

    hvt_orbit = Orbit.from_classical(
        Earth, hvt_a, hvt_ecc, shared_inc, shared_raan, hvt_argp, hvt_nu
    )
    r_hvt_eci_val = hvt_orbit.r.to(u.km).value
    v_hvt_eci_val = hvt_orbit.v.to(u.km / u.s).value
    hvt_a_val = hvt_a.to(u.km).value

    hvt_states = np.hstack([r_hvt_eci_val, v_hvt_eci_val])
    simulator.store_lvlh = True
    simulator.set_origin_satellite_by_vector(
        states=hvt_states,
        prefix=hvt_config["prefix"],
        color=hvt_config["color"],
    )
    simulator.position_generator.set_origin(hvt_states)

    rot_lvlh_to_eci = _get_lvlh_to_eci_rotation_matrix(r_hvt_eci_val, v_hvt_eci_val)
    return hvt_states, hvt_a_val, rot_lvlh_to_eci


def _generate_attackers(
    simulator, attack_config, hvt_state, rot_lvlh_to_eci, observation_threshold
):
    """生成敌方攻击卫星，返回其 ECI 位置列表"""
    r_hvt_eci_val = hvt_state[:3]
    v_hvt_eci_val = hvt_state[3:]

    rho_range = attack_config.get("polar_rho_range_km", [25, 35])
    theta_range = attack_config.get("polar_theta_range_deg", [0, 360])
    vel_pert_range_km_s = attack_config.get(
        "relative_velocity_perturbation_km_s", [-0.01, 0.01]
    )
    min_peer_separation = attack_config.get("min_peer_separation_km", 5.0)
    attacker_count = attack_config["count"]

    generated_positions = []
    for i in range(attacker_count):
        retry_count = 0
        while True:
            rho_attacker = np.random.uniform(rho_range[0], rho_range[1])
            theta_attacker = np.random.uniform(
                np.deg2rad(theta_range[0]), np.deg2rad(theta_range[1])
            )
            r_rel_lvlh = np.array(
                [
                    rho_attacker * np.cos(theta_attacker),
                    rho_attacker * np.sin(theta_attacker),
                    0,
                ]
            )
            r_rel_eci = rot_lvlh_to_eci @ r_rel_lvlh
            r_attacker_candidate_eci = r_hvt_eci_val + r_rel_eci

            # 检查与其他敌方卫星的距离约束
            is_separation_ok = True
            for prev_pos in generated_positions:
                if (
                    np.linalg.norm(r_attacker_candidate_eci - prev_pos)
                    < min_peer_separation
                ):
                    is_separation_ok = False
                    break
            if not is_separation_ok:
                retry_count += 1
                if retry_count > 100:
                    raise RuntimeError(
                        f"无法为敌方攻击卫星 {i} 找到满足最小间距约束的位置。"
                    )
                continue
            break

        generated_positions.append(r_attacker_candidate_eci)

        # 速度扰动
        vx_rel = np.random.uniform(vel_pert_range_km_s[0], vel_pert_range_km_s[1])
        vy_rel = np.random.uniform(vel_pert_range_km_s[0], vel_pert_range_km_s[1])
        v_rel_lvlh = np.array([vx_rel, vy_rel, 0])
        v_rel_eci = rot_lvlh_to_eci @ v_rel_lvlh
        v_attacker_eci_val = v_hvt_eci_val + v_rel_eci

        attacker_states = np.hstack([r_attacker_candidate_eci, v_attacker_eci_val])
        simulator.add_satellite_by_vector(
            states=attacker_states,
            accel=np.zeros(3),
            prefix=attack_config["prefix"],
            color=attack_config["color"],
        )
    return generated_positions


def _generate_interceptors(
    simulator,
    interceptor_config,
    hvt_state,
    hvt_a_val,
    rot_lvlh_to_eci,
    attacker_positions,
    observation_threshold,
):
    """根据 generation_mode 分发到对应的生成函数"""
    generation_mode = interceptor_config.get("generation_mode", "natural_ellipse")
    if generation_mode == "natural_ellipse":
        return _generate_interceptors_natural_ellipse(
            simulator,
            interceptor_config,
            hvt_state,
            hvt_a_val,
            rot_lvlh_to_eci,
            attacker_positions,
            observation_threshold,
        )
    elif generation_mode == "formation_flying":
        return _generate_interceptors_formation_flying(
            simulator,
            interceptor_config,
            hvt_state,
            rot_lvlh_to_eci,
            attacker_positions,
            observation_threshold,
        )
    elif generation_mode == "sector_around_axis":
        return _generate_interceptors_sector_around_axis(
            simulator,
            interceptor_config,
            hvt_state,
            rot_lvlh_to_eci,
            attacker_positions,
            observation_threshold,
        )
    else:
        raise ValueError(f"不支持的生成模式: {generation_mode}")


def _generate_interceptors_natural_ellipse(
    simulator,
    interceptor_config,
    hvt_state,
    hvt_a_val,
    rot_lvlh_to_eci,
    attacker_positions,
    observation_threshold,
):
    """自然椭圆绕飞模式生成拦截卫星"""
    r_hvt_eci_val = hvt_state[:3]
    v_hvt_eci_val = hvt_state[3:]

    interceptor_count = interceptor_config["count"]
    min_separation = interceptor_config.get("min_interceptor_separation_km", 3)

    b_range = interceptor_config.get("ellipse_b_range_km", [8, 12])
    phase_range = interceptor_config.get("ellipse_phase_range_deg", [0, 360])
    phases_deg = interceptor_config.get("ellipse_phases_deg", None)

    b_ellipse = np.random.uniform(b_range[0], b_range[1])
    if phases_deg is None:
        phases_deg = [
            np.random.uniform(phase_range[0], phase_range[1])
            for _ in range(interceptor_count)
        ]

    generated_positions = []
    for i in range(interceptor_count):
        retry_count = 0
        while True:
            phase_rad = np.deg2rad(phases_deg[i])
            X_lvlh = func_natural_ellipse_formation(b_ellipse, phase_rad, hvt_a_val)
            r_rel_lvlh_int = X_lvlh[0:3]
            v_rel_lvlh_int = X_lvlh[3:6]

            r_rel_eci_int = rot_lvlh_to_eci @ r_rel_lvlh_int
            v_rel_eci_int = rot_lvlh_to_eci @ v_rel_lvlh_int
            r_interceptor_candidate_eci = r_hvt_eci_val + r_rel_eci_int
            v_interceptor_eci_val = v_hvt_eci_val + v_rel_eci_int

            # 检查与所有敌方卫星的距离
            all_distances_ok = True
            max_dist = 0.0
            for attacker_pos in attacker_positions:
                dist = np.linalg.norm(r_interceptor_candidate_eci - attacker_pos)
                max_dist = max(max_dist, dist)
                if dist > observation_threshold:
                    all_distances_ok = False
                    break
            if not all_distances_ok:
                retry_count += 1
                if retry_count > 200:
                    print(f"    警告: 无法为拦截卫星 {i} 找到满足观测距离约束的位置。")
                    print(
                        f"    当前与敌方最大距离: {max_dist:.2f} km > {observation_threshold} km"
                    )
                    print(f"    建议: 调整配置参数，缩小敌方和我方的距离范围。")
                    raise RuntimeError(
                        f"无法为拦截卫星 {i} 找到满足观测距离约束的位置。"
                    )
                phases_deg[i] = np.random.uniform(phase_range[0], phase_range[1])
                continue

            # 检查与已生成拦截星的最小间距
            separation_ok = True
            for prev_pos in generated_positions:
                if (
                    np.linalg.norm(r_interceptor_candidate_eci - prev_pos)
                    < min_separation
                ):
                    separation_ok = False
                    break
            if not separation_ok:
                retry_count += 1
                if retry_count > 200:
                    raise RuntimeError(
                        f"无法为拦截卫星 {i} 找到满足最小间距约束的位置。"
                    )
                phases_deg[i] = np.random.uniform(phase_range[0], phase_range[1])
                continue
            break

        generated_positions.append(r_interceptor_candidate_eci)
        interceptor_states = np.hstack(
            [r_interceptor_candidate_eci, v_interceptor_eci_val]
        )
        simulator.add_satellite_by_vector(
            states=interceptor_states,
            accel=np.zeros(3),
            prefix=interceptor_config["prefix"],
            color=interceptor_config["color"],
        )
    return interceptor_count


def _generate_interceptors_formation_flying(
    simulator,
    interceptor_config,
    hvt_state,
    rot_lvlh_to_eci,
    attacker_positions,
    observation_threshold,
):
    """伴飞模式（沿运动方向）生成拦截卫星"""
    r_hvt_eci_val = hvt_state[:3]
    v_hvt_eci_val = hvt_state[3:]

    interceptor_count = interceptor_config["count"]
    min_separation = interceptor_config.get("min_interceptor_separation_km", 3)

    distance_range = interceptor_config.get("formation_distance_range_km", [5, 25])
    angle_range_deg = interceptor_config.get("formation_angle_range_deg", [0, 360])
    vel_pert_range = interceptor_config.get(
        "relative_velocity_perturbation_km_s", [0.0, 0.0]
    )

    generated_positions = []
    for i in range(interceptor_count):
        retry_count = 0
        while True:
            distance_along_track = np.random.uniform(
                distance_range[0], distance_range[1]
            )
            angle_in_rt_plane = np.random.uniform(
                np.deg2rad(angle_range_deg[0]), np.deg2rad(angle_range_deg[1])
            )
            r_component = distance_along_track * np.sin(angle_in_rt_plane)
            t_component = distance_along_track * np.cos(angle_in_rt_plane)
            r_rel_lvlh_int = np.array([r_component, t_component, 0])

            r_rel_eci_int = rot_lvlh_to_eci @ r_rel_lvlh_int
            r_interceptor_candidate_eci = r_hvt_eci_val + r_rel_eci_int

            # 与敌方卫星的距离检查
            all_distances_ok = True
            max_dist = 0.0
            for attacker_pos in attacker_positions:
                dist = np.linalg.norm(r_interceptor_candidate_eci - attacker_pos)
                max_dist = max(max_dist, dist)
                if dist > observation_threshold:
                    all_distances_ok = False
                    break
            if not all_distances_ok:
                retry_count += 1
                if retry_count > 200:
                    print(f"    警告: 无法为拦截卫星 {i} 找到满足观测距离约束的位置。")
                    print(
                        f"    当前与敌方最大距离: {max_dist:.2f} km > {observation_threshold} km"
                    )
                    print(f"    建议: 调整配置参数，缩小距离范围。")
                    raise RuntimeError(
                        f"无法为拦截卫星 {i} 找到满足观测距离约束的位置。"
                    )
                continue

            # 拦截星之间最小间距检查
            separation_ok = True
            for prev_pos in generated_positions:
                if (
                    np.linalg.norm(r_interceptor_candidate_eci - prev_pos)
                    < min_separation
                ):
                    separation_ok = False
                    break
            if not separation_ok:
                retry_count += 1
                if retry_count > 200:
                    raise RuntimeError(
                        f"无法为拦截卫星 {i} 找到满足最小间距约束的位置。"
                    )
                continue
            break

        generated_positions.append(r_interceptor_candidate_eci)

        vx_rel = np.random.uniform(vel_pert_range[0], vel_pert_range[1])
        vy_rel = np.random.uniform(vel_pert_range[0], vel_pert_range[1])
        vz_rel = np.random.uniform(vel_pert_range[0], vel_pert_range[1])
        v_rel_lvlh = np.array([vx_rel, vy_rel, vz_rel])
        v_rel_eci = rot_lvlh_to_eci @ v_rel_lvlh
        v_interceptor_eci_val = v_hvt_eci_val + v_rel_eci

        interceptor_states = np.hstack(
            [r_interceptor_candidate_eci, v_interceptor_eci_val]
        )
        simulator.add_satellite_by_vector(
            states=interceptor_states,
            accel=np.zeros(3),
            prefix=interceptor_config["prefix"],
            color=interceptor_config["color"],
        )
    return interceptor_count


def _generate_interceptors_sector_around_axis(
    simulator,
    interceptor_config,
    hvt_state,
    rot_lvlh_to_eci,
    attacker_positions,
    observation_threshold,
):
    """以 HVT-攻击者连线为轴，向两侧扩散的扇形模式生成拦截卫星（使用已有的 attacker_positions）"""
    r_hvt_eci_val = hvt_state[:3]
    v_hvt_eci_val = hvt_state[3:]

    interceptor_count = interceptor_config["count"]
    min_separation = interceptor_config.get("min_interceptor_separation_km", 3)
    half_angle_deg = interceptor_config.get("sector_half_angle_deg", 15)
    half_angle_rad = np.deg2rad(half_angle_deg)
    dist_range_km = interceptor_config.get("sector_distance_range_km", [10, 30])
    vel_pert_range = interceptor_config.get(
        "relative_velocity_perturbation_km_s", [0.0, 0.0]
    )
    per_attacker_counts = interceptor_config.get("per_attacker_counts", None)

    num_attackers = len(attacker_positions)
    if num_attackers == 0:
        raise RuntimeError("未找到任何攻击者，无法使用 sector_around_axis 模式。")

    # 分配每个攻击者的拦截星数量
    if per_attacker_counts is None:
        base_cnt = interceptor_count // num_attackers
        remainder = interceptor_count % num_attackers
        counts = [
            base_cnt + 1 if i < remainder else base_cnt for i in range(num_attackers)
        ]
    else:
        if len(per_attacker_counts) != num_attackers:
            raise ValueError("per_attacker_counts 长度必须与攻击者数量相等。")
        counts = per_attacker_counts
        if sum(counts) != interceptor_count:
            raise ValueError("per_attacker_counts 总和必须等于 interceptor_count。")

    # 计算每个攻击者在 HVT LVLH 坐标系中的方向角
    attacker_angles = []
    for pos in attacker_positions:
        r_rel_eci = pos - r_hvt_eci_val
        r_rel_lvlh = rot_lvlh_to_eci.T @ r_rel_eci  # 转置即逆变换
        angle = np.arctan2(r_rel_lvlh[1], r_rel_lvlh[0])  # 弧度，参考 R 轴
        attacker_angles.append(angle)

    generated_positions = []
    for att_idx, att_angle in enumerate(attacker_angles):
        num_this = counts[att_idx]
        for j in range(num_this):
            retry_count = 0
            while True:
                offset = np.random.uniform(-half_angle_rad, half_angle_rad)
                final_angle = att_angle + offset
                dist = np.random.uniform(dist_range_km[0], dist_range_km[1])
                r_rel_lvlh_int = np.array(
                    [dist * np.cos(final_angle), dist * np.sin(final_angle), 0.0]
                )
                r_rel_eci_int = rot_lvlh_to_eci @ r_rel_lvlh_int
                r_candidate = r_hvt_eci_val + r_rel_eci_int

                # 约束1：与所有敌方卫星的距离 <= observation_threshold
                ok_dist = True
                max_dist = 0.0
                for apos in attacker_positions:
                    d = np.linalg.norm(r_candidate - apos)
                    max_dist = max(max_dist, d)
                    if d > observation_threshold:
                        ok_dist = False
                        break
                if not ok_dist:
                    retry_count += 1
                    if retry_count > 200:
                        raise RuntimeError(
                            f"无法为攻击者 {att_idx} 的第 {j} 个拦截星找到满足观测距离约束的位置。"
                        )
                    continue

                # 约束2：拦截星之间的最小间距
                sep_ok = True
                for prev in generated_positions:
                    if np.linalg.norm(r_candidate - prev) < min_separation:
                        sep_ok = False
                        break
                if not sep_ok:
                    retry_count += 1
                    if retry_count > 200:
                        raise RuntimeError(
                            f"无法为攻击者 {att_idx} 的第 {j} 个拦截星找到满足最小间距约束的位置。"
                        )
                    continue
                break

            generated_positions.append(r_candidate)

            # 速度：加小扰动
            vx_rel = np.random.uniform(vel_pert_range[0], vel_pert_range[1])
            vy_rel = np.random.uniform(vel_pert_range[0], vel_pert_range[1])
            v_rel_lvlh = np.array([vx_rel, vy_rel, 0.0])
            v_rel_eci = rot_lvlh_to_eci @ v_rel_lvlh
            v_interceptor_eci_val = v_hvt_eci_val + v_rel_eci

            interceptor_states = np.hstack([r_candidate, v_interceptor_eci_val])
            simulator.add_satellite_by_vector(
                states=interceptor_states,
                accel=np.zeros(3),
                prefix=interceptor_config["prefix"],
                color=interceptor_config["color"],
            )
    return interceptor_count


# def create_scenario(simulator, config):
#     """
#     创建2v2和3V1场景，新版本要求：
#     1. 敌方敌方卫星：极坐标 ρ∈[20,30]km, θ∈[0,360°] 随机生成
#     2. 我方n颗拦截卫星：支持两种模式
#        - 模式1：自然椭圆绕飞模式
#        - 模式2：沿着HVT运动方向伴飞，速度方向与HVT一致
#     3. 所有卫星在同一轨道平面，但不在同一轨道上
#     4. 我方卫星与敌方卫星的距离在观测阈值内（默认50km）
#     """
#     if config.get("is_harbin") is not None:
#         return create_scenario_from_harbin_config(simulator, config)
#
#     # 1. 加载所有卫星组的配置
#     satellite_groups = config.get("satellite_groups", [])
#     hvt_config = next((g for g in satellite_groups if g["group_name"] == "hvt"), None)
#     attack_config = next(
#         (g for g in satellite_groups if g["group_name"] == "attackers"), None
#     )
#     interceptor_config = next(
#         (g for g in satellite_groups if g["group_name"] == "interceptors"), None
#     )
#
#     if not all([hvt_config, attack_config, interceptor_config]):
#         raise ValueError(
#             "配置文件必须包含 'HVT', 'Red_Interceptor', 和 'Blue_Attacker' 的分组。"
#         )
#     # 【新增】读取观测距离阈值
#     observation_threshold = config.get("observation_threshold_km", 50)
#
#     # --- 步骤 1: 创建HVT并锁定轨道平面 ---
#     # print("--- 步骤1: 创建高价值目标 (HVT) 作为基准 ---")
#     hvt_ranges = hvt_config["ranges"]
#
#     hvt_a = np.random.uniform(hvt_ranges["a"][0], hvt_ranges["a"][1]) * u.km
#     hvt_ecc = np.random.uniform(hvt_ranges["ecc"][0], hvt_ranges["ecc"][1]) * u.one
#     shared_inc = np.random.uniform(hvt_ranges["inc"][0], hvt_ranges["inc"][1]) * u.deg
#     shared_raan = (
#         np.random.uniform(hvt_ranges["raan"][0], hvt_ranges["raan"][1]) * u.deg
#     )
#     hvt_argp = np.random.uniform(hvt_ranges["argp"][0], hvt_ranges["argp"][1]) * u.deg
#     hvt_nu = np.random.uniform(hvt_ranges["nu"][0], hvt_ranges["nu"][1]) * u.deg
#
#     hvt_orbit = Orbit.from_classical(
#         Earth, hvt_a, hvt_ecc, shared_inc, shared_raan, hvt_argp, hvt_nu
#     )
#     r_hvt_eci_val = hvt_orbit.r.to(u.km).value
#     v_hvt_eci_val = hvt_orbit.v.to(u.km / u.s).value
#
#     # 保存HVT的轨道半长轴（用于自然椭圆绕飞计算）
#     hvt_a_val = hvt_a.to(u.km).value
#
#     hvt_states = np.hstack([r_hvt_eci_val, v_hvt_eci_val])
#     simulator.store_lvlh = True
#     simulator.set_origin_satellite_by_vector(
#         states=hvt_states,
#         prefix=hvt_config["prefix"],
#         color=hvt_config["color"],
#     )
#     simulator.position_generator.set_origin(hvt_states)
#     # print(f"HVT 初始位置 (ECI): {r_hvt_eci_val} km,初始速度 (ECI): {v_hvt_eci_val} km/s\n")
#     # --- 步骤 2: 极坐标生成敌方攻击卫星 ---
#     # 获取LVLH坐标系的旋转矩阵
#     rot_hvt_lvlh_to_eci = _get_lvlh_to_eci_rotation_matrix(r_hvt_eci_val, v_hvt_eci_val)
#     # 读取极坐标参数
#     rho_range = attack_config.get("polar_rho_range_km", [25, 35])
#     theta_range = attack_config.get("polar_theta_range_deg", [0, 360])
#     vel_pert_range_km_s = attack_config.get(
#         "relative_velocity_perturbation_km_s", [-0.01, 0.01]
#     )
#     min_peer_separation = attack_config.get("min_peer_separation_km", 5.0)
#     attacker_count = attack_config["count"]
#     generated_attacker_positions = []
#
#     for i in range(attacker_count):
#         # print(f"\n  正在创建敌方攻击卫星 {i}...")
#
#         retry_count = 0
#         while True:
#             # 极坐标随机采样
#             rho_attacker = np.random.uniform(rho_range[0], rho_range[1])
#             theta_attacker = np.random.uniform(
#                 np.deg2rad(theta_range[0]), np.deg2rad(theta_range[1])
#             )
#
#             # 极坐标转换为LVLH坐标系下的笛卡尔坐标 (R-T平面)
#             r_rel_lvlh = np.array(
#                 [
#                     rho_attacker * np.cos(theta_attacker),
#                     rho_attacker * np.sin(theta_attacker),
#                     0,
#                 ]
#             )
#
#             # 转换到ECI坐标系
#             r_rel_eci = rot_hvt_lvlh_to_eci @ r_rel_lvlh
#             r_attacker_candidate_eci = r_hvt_eci_val + r_rel_eci
#
#             # 先根据位置检查与其他敌方卫星的距离约束
#             is_separation_ok = True
#             for prev_pos in generated_attacker_positions:
#                 dist_to_peer = np.linalg.norm(r_attacker_candidate_eci - prev_pos)
#                 if dist_to_peer < min_peer_separation:
#                     is_separation_ok = False
#                     break
#
#             if not is_separation_ok:
#                 retry_count += 1
#                 if retry_count > 100:
#                     raise RuntimeError(
#                         f"无法为敌方攻击卫星 {i} 找到满足最小间距约束的位置。"
#                     )
#                 continue
#
#             # 满足所有约束，退出循环
#             break
#
#         # 记录位置
#         generated_attacker_positions.append(r_attacker_candidate_eci)
#
#         # 添加速度扰动（在LVLH坐标系中）
#         vx_rel = np.random.uniform(vel_pert_range_km_s[0], vel_pert_range_km_s[1])
#         vy_rel = np.random.uniform(vel_pert_range_km_s[0], vel_pert_range_km_s[1])
#         v_rel_lvlh = np.array([vx_rel, vy_rel, 0])
#
#         # 转换到ECI坐标系
#         v_rel_eci = rot_hvt_lvlh_to_eci @ v_rel_lvlh
#         v_attacker_eci_val = v_hvt_eci_val + v_rel_eci
#
#         attacker_states = np.hstack([r_attacker_candidate_eci, v_attacker_eci_val])
#         simulator.add_satellite_by_vector(
#             states=attacker_states,
#             accel=np.zeros(3),
#             prefix=attack_config["prefix"],
#             color=attack_config["color"],
#         )
#
#     # --- 步骤 3: 创建我方拦截卫星（根据模式选择） ---
#     generation_mode = interceptor_config.get("generation_mode", "natural_ellipse")
#     interceptor_count = interceptor_config["count"]
#     min_separation = interceptor_config.get("min_interceptor_separation_km", 3)
#
#     if generation_mode == "natural_ellipse":
#         # --- 步骤3: 创建我方拦截卫星（自然椭圆绕飞模式） ---
#
#         # 读取椭圆参数
#         b_range = interceptor_config.get("ellipse_b_range_km", [8, 12])
#         phase_range = interceptor_config.get("ellipse_phase_range_deg", [0, 360])
#         phases_deg = interceptor_config.get("ellipse_phases_deg", None)
#
#         # 随机生成短半轴 b
#         b_ellipse = np.random.uniform(b_range[0], b_range[1])
#         # 如果未指定相位，则生成随机相位
#         if phases_deg is None:
#             phases_deg = [
#                 np.random.uniform(phase_range[0], phase_range[1])
#                 for _ in range(interceptor_count)
#             ]
#         generated_interceptor_positions = []
#         for i in range(interceptor_count):
#
#             retry_count = 0
#             while True:
#                 # 使用指定的相位角
#                 phase_rad = np.deg2rad(phases_deg[i])
#
#                 # 调用自然椭圆绕飞函数
#                 X_lvlh = func_natural_ellipse_formation(b_ellipse, phase_rad, hvt_a_val)
#
#                 # 提取位置和速度 (LVLH坐标系)
#                 r_rel_lvlh_int = X_lvlh[0:3]  # [Rx, Ry, Rz]
#                 v_rel_lvlh_int = X_lvlh[3:6]  # [Vx, Vy, Vz]
#
#                 # 转换到ECI坐标系
#                 r_rel_eci_int = rot_hvt_lvlh_to_eci @ r_rel_lvlh_int
#                 v_rel_eci_int = rot_hvt_lvlh_to_eci @ v_rel_lvlh_int
#
#                 r_interceptor_candidate_eci = r_hvt_eci_val + r_rel_eci_int
#                 v_interceptor_eci_val = v_hvt_eci_val + v_rel_eci_int
#                 # 检查与所有敌方卫星的距离是否在观测阈值内
#                 max_dist_to_attacker = 0
#                 all_distances_ok = True
#                 for attacker_pos in generated_attacker_positions:
#                     dist_to_attacker = np.linalg.norm(
#                         r_interceptor_candidate_eci - attacker_pos
#                     )
#                     max_dist_to_attacker = max(max_dist_to_attacker, dist_to_attacker)
#                     if dist_to_attacker > observation_threshold:
#                         all_distances_ok = False
#                         break
#
#                 if not all_distances_ok:
#                     retry_count += 1
#                     if retry_count > 200:
#                         print(
#                             f"    警告: 无法为拦截卫星 {i} 找到满足观测距离约束的位置。"
#                         )
#                         print(
#                             f"    当前与敌方最大距离: {max_dist_to_attacker:.2f} km > {observation_threshold} km"
#                         )
#                         print(f"    建议: 调整配置参数，缩小敌方和我方的距离范围。")
#                         raise RuntimeError(
#                             f"无法为拦截卫星 {i} 找到满足观测距离约束的位置。"
#                         )
#                     # 重新随机生成相位
#                     phases_deg[i] = np.random.uniform(phase_range[0], phase_range[1])
#                     continue
#                 # 再根据位置检查与其他拦截卫星的距离约束
#                 is_separation_ok = True
#                 for prev_pos in generated_interceptor_positions:
#                     dist_to_peer = np.linalg.norm(
#                         r_interceptor_candidate_eci - prev_pos
#                     )
#                     if dist_to_peer < min_separation:
#                         is_separation_ok = False
#                         break
#
#                 if not is_separation_ok:
#                     retry_count += 1
#                     if retry_count > 200:
#                         raise RuntimeError(
#                             f"无法为拦截卫星 {i} 找到满足最小间距约束的位置。"
#                         )
#                     # 重新随机生成相位
#                     phases_deg[i] = np.random.uniform(phase_range[0], phase_range[1])
#                     continue
#
#                 # 满足所有约束，退出循环
#                 break
#             # 记录位置
#             generated_interceptor_positions.append(r_interceptor_candidate_eci)
#             # 添加卫星
#             interceptor_states = np.hstack(
#                 [r_interceptor_candidate_eci, v_interceptor_eci_val]
#             )
#             simulator.add_satellite_by_vector(
#                 states=interceptor_states,
#                 accel=np.zeros(3),
#                 prefix=interceptor_config["prefix"],
#                 color=interceptor_config["color"],
#             )
#     elif generation_mode == "formation_flying":
#         # 方案1：从距离范围内随机采样
#         distance_range = interceptor_config.get("formation_distance_range_km", [5, 25])
#         angle_range_deg = interceptor_config.get("formation_angle_range_deg", [0, 360])
#         vel_pert_range_interceptor = interceptor_config.get(
#             "relative_velocity_perturbation_km_s", [0.0, 0.0]
#         )
#
#         generated_interceptor_positions = []
#
#         for i in range(interceptor_count):
#             retry_count = 0
#             while True:
#                 # 在范围内随机采样距离和角度
#                 distance_along_track = np.random.uniform(
#                     distance_range[0], distance_range[1]
#                 )
#                 angle_in_rt_plane = np.random.uniform(
#                     np.deg2rad(angle_range_deg[0]), np.deg2rad(angle_range_deg[1])
#                 )
#
#                 # 在LVLH坐标系的R-T平面内生成位置
#                 # 使用极坐标，以T轴为主方向
#                 r_component = distance_along_track * np.sin(angle_in_rt_plane)
#                 t_component = distance_along_track * np.cos(angle_in_rt_plane)
#
#                 r_rel_lvlh_int = np.array(
#                     [
#                         r_component,  # R轴（径向）
#                         t_component,  # T轴（沿运动方向）
#                         0,  # N轴（法向，保持为0）
#                     ]
#                 )
#
#                 # 转换到ECI坐标系
#                 r_rel_eci_int = rot_hvt_lvlh_to_eci @ r_rel_lvlh_int
#                 r_interceptor_candidate_eci = r_hvt_eci_val + r_rel_eci_int
#
#                 # 检查与所有敌方卫星的距离是否在观测阈值内
#                 max_dist_to_attacker = 0
#                 all_distances_ok = True
#                 for attacker_pos in generated_attacker_positions:
#                     dist_to_attacker = np.linalg.norm(
#                         r_interceptor_candidate_eci - attacker_pos
#                     )
#                     max_dist_to_attacker = max(max_dist_to_attacker, dist_to_attacker)
#                     if dist_to_attacker > observation_threshold:
#                         all_distances_ok = False
#                         break
#
#                 if not all_distances_ok:
#                     retry_count += 1
#                     if retry_count > 200:
#                         print(
#                             f"    警告: 无法为拦截卫星 {i} 找到满足观测距离约束的位置。"
#                         )
#                         print(
#                             f"    当前与敌方最大距离: {max_dist_to_attacker:.2f} km > {observation_threshold} km"
#                         )
#                         print(f"    建议: 调整配置参数，缩小距离范围。")
#                         raise RuntimeError(
#                             f"无法为拦截卫星 {i} 找到满足观测距离约束的位置。"
#                         )
#                     # 重新随机采样
#                     continue
#
#                 # 检查与其他拦截器的距离约束
#                 is_separation_ok = True
#                 for prev_pos in generated_interceptor_positions:
#                     dist_to_peer = np.linalg.norm(
#                         r_interceptor_candidate_eci - prev_pos
#                     )
#                     if dist_to_peer < min_separation:
#                         is_separation_ok = False
#                         break
#
#                 if not is_separation_ok:
#                     retry_count += 1
#                     if retry_count > 200:
#                         raise RuntimeError(
#                             f"无法为拦截卫星 {i} 找到满足最小间距约束的位置。"
#                         )
#                     continue
#
#                 # 满足所有约束，退出循环
#                 break
#
#             # 记录位置
#             generated_interceptor_positions.append(r_interceptor_candidate_eci)
#
#             # 速度：与HVT保持完全一致（伴飞）
#             vx_rel_int = np.random.uniform(
#                 vel_pert_range_interceptor[0], vel_pert_range_interceptor[1]
#             )
#             vy_rel_int = np.random.uniform(
#                 vel_pert_range_interceptor[0], vel_pert_range_interceptor[1]
#             )
#             vz_rel_int = np.random.uniform(
#                 vel_pert_range_interceptor[0], vel_pert_range_interceptor[1]
#             )
#             v_rel_lvlh_int = np.array([vx_rel_int, vy_rel_int, vz_rel_int])
#
#             v_rel_eci_int = rot_hvt_lvlh_to_eci @ v_rel_lvlh_int
#             v_interceptor_eci_val = v_hvt_eci_val + v_rel_eci_int
#
#             interceptor_states = np.hstack(
#                 [r_interceptor_candidate_eci, v_interceptor_eci_val]
#             )
#             simulator.add_satellite_by_vector(
#                 states=interceptor_states,
#                 accel=np.zeros(3),
#                 prefix=interceptor_config["prefix"],
#                 color=interceptor_config["color"],
#             )
#     elif generation_mode == "sector_around_axis":
#         # ---- 新模式：以 HVT-攻击者连线为轴，向两侧扩散 angle_half 度 ----
#         half_angle_deg = interceptor_config.get("sector_half_angle_deg", 15)
#         half_angle_rad = np.deg2rad(half_angle_deg)
#         dist_range_km = interceptor_config.get("sector_distance_range_km", [10, 30])
#         vel_pert_range = interceptor_config.get(
#             "relative_velocity_perturbation_km_s", [0.0, 0.0]
#         )
#         # 如果指定了每个攻击者对应的拦截星数量，否则均匀分配
#         per_attacker_counts = interceptor_config.get("per_attacker_counts", None)
#
#         # 获取所有攻击者的 ECI 位置（已经生成）
#         attacker_positions = []
#         for sat_name, sat in simulator.satellites.items():
#             if sat_name.startswith(attack_config["prefix"]):
#                 attacker_positions.append(sat.initial_state[:3])
#         num_attackers = len(attacker_positions)
#         if num_attackers == 0:
#             raise RuntimeError("未找到任何攻击者，无法使用 sector_around_axis 模式。")
#
#         # 分配每个攻击者应生成的拦截星数量
#         if per_attacker_counts is None:
#             base_cnt = interceptor_count // num_attackers
#             remainder = interceptor_count % num_attackers
#             counts = [
#                 base_cnt + 1 if i < remainder else base_cnt
#                 for i in range(num_attackers)
#             ]
#         else:
#             if len(per_attacker_counts) != num_attackers:
#                 raise ValueError("per_attacker_counts 长度必须与攻击者数量相等。")
#             counts = per_attacker_counts
#             if sum(counts) != interceptor_count:
#                 raise ValueError("per_attacker_counts 总和必须等于 interceptor_count。")
#
#         # 计算每个攻击者在 HVT LVLH 坐标系中的方向角
#         attacker_angles = []
#         for pos in attacker_positions:
#             r_rel_eci = pos - r_hvt_eci_val
#             r_rel_lvlh = rot_hvt_lvlh_to_eci.T @ r_rel_eci  # 转置即逆变换
#             angle = np.arctan2(r_rel_lvlh[1], r_rel_lvlh[0])  # 弧度，参考 R 轴
#             attacker_angles.append(angle)
#
#         generated_interceptor_positions = []
#         # 为每个攻击者生成对应数量的拦截星
#         for att_idx, att_angle in enumerate(attacker_angles):
#             num_this = counts[att_idx]
#             for _ in range(num_this):
#                 retry_count = 0
#                 while True:
#                     # 在扇区内随机偏移
#                     offset = np.random.uniform(-half_angle_rad, half_angle_rad)
#                     final_angle = att_angle + offset
#                     # 随机距离
#                     dist = np.random.uniform(dist_range_km[0], dist_range_km[1])
#                     # LVLH 中位置
#                     r_rel_lvlh_int = np.array(
#                         [dist * np.cos(final_angle), dist * np.sin(final_angle), 0.0]
#                     )
#                     # 转换到 ECI
#                     r_rel_eci_int = rot_hvt_lvlh_to_eci @ r_rel_lvlh_int
#                     r_interceptor_candidate_eci = r_hvt_eci_val + r_rel_eci_int
#
#                     # 约束1：与所有敌方卫星的距离 <= observation_threshold
#                     ok_distance = True
#                     max_dist = 0.0
#                     for apos in attacker_positions:
#                         d = np.linalg.norm(r_interceptor_candidate_eci - apos)
#                         max_dist = max(max_dist, d)
#                         if d > observation_threshold:
#                             ok_distance = False
#                             break
#                     if not ok_distance:
#                         retry_count += 1
#                         if retry_count > 200:
#                             raise RuntimeError(
#                                 f"无法为攻击者 {att_idx} 的第 {_} 个拦截星找到满足观测距离约束的位置。"
#                             )
#                         continue
#
#                     # 约束2：拦截星之间的最小间距
#                     sep_ok = True
#                     for prev_pos in generated_interceptor_positions:
#                         if (
#                             np.linalg.norm(r_interceptor_candidate_eci - prev_pos)
#                             < min_separation
#                         ):
#                             sep_ok = False
#                             break
#                     if not sep_ok:
#                         retry_count += 1
#                         if retry_count > 200:
#                             raise RuntimeError(
#                                 f"无法为攻击者 {att_idx} 的第 {_} 个拦截星找到满足最小间距约束的位置。"
#                             )
#                         continue
#
#                     # 所有约束通过
#                     break
#
#                 generated_interceptor_positions.append(r_interceptor_candidate_eci)
#
#                 # 速度：可以完全与 HVT 相同，或加小扰动
#                 vx_rel = np.random.uniform(vel_pert_range[0], vel_pert_range[1])
#                 vy_rel = np.random.uniform(vel_pert_range[0], vel_pert_range[1])
#                 v_rel_lvlh = np.array([vx_rel, vy_rel, 0.0])
#                 v_rel_eci = rot_hvt_lvlh_to_eci @ v_rel_lvlh
#                 v_interceptor_eci_val = v_hvt_eci_val + v_rel_eci
#
#                 interceptor_states = np.hstack(
#                     [r_interceptor_candidate_eci, v_interceptor_eci_val]
#                 )
#                 simulator.add_satellite_by_vector(
#                     states=interceptor_states,
#                     accel=np.zeros(3),
#                     prefix=interceptor_config["prefix"],
#                     color=interceptor_config["color"],
#                 )
#     else:
#         raise ValueError(f"不支持的生成模式: {generation_mode}")
#
#     return interceptor_count, attacker_count


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
    # 引力常数（单位：km³/s²）
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


def distance(lhs: np.ndarray[float], rhs: np.ndarray[float]) -> float:
    return np.linalg.norm(lhs - rhs)


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """计算两个向量之间的夹角（单位：弧度）。"""
    # 防止除以零
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 < 1e-6 or norm_v2 < 1e-6:
        return 0.0  # 如果向量几乎为零，夹角无意义，返回0
    dot_product = np.dot(v1, v2)
    cos_theta = dot_product / (norm_v1 * norm_v2)
    # clip确保值在[-1, 1]范围内，避免浮点数精度问题
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)


def get_plane_normal(p1, p2, p3):
    """
    计算由三个点确定的平面的单位法向量

    参数:
    p1, p2, p3 (array-like): 三个点的坐标，每个点的形状为(3,)

    返回:
    numpy.ndarray: 平面的单位法向量，形状为(3,)
    """
    # 将输入转换为numpy数组
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    p3 = np.asarray(p3, dtype=float)

    # 计算向量PQ和PR
    pq = p2 - p1
    pr = p3 - p1

    # 计算叉积得到法向量
    normal = np.cross(pq, pr)

    # 归一化法向量
    if np.linalg.norm(normal) == 0:
        raise ValueError("三个点共线，无法确定唯一平面")

    unit_normal = normal / np.linalg.norm(normal)

    return unit_normal


def project_point_to_plane(v, plane_normal, plane_point):
    """
    将三维点v投影到由法向量和平面上一点定义的平面上

    参数:
    v (array-like): 待投影的三维点，形状为(3,)
    plane_normal (array-like): 平面的法向量，形状为(3,)，需为单位向量
    plane_point (array-like): 平面上的一个点，形状为(3,)

    返回:
    numpy.ndarray: 投影后的点坐标，形状为(3,)
    """
    # 将输入转换为numpy数组
    v = np.asarray(v, dtype=float)
    plane_normal = np.asarray(plane_normal, dtype=float)
    plane_point = np.asarray(plane_point, dtype=float)

    # 确保法向量是单位向量
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    # 计算向量v到平面的距离d
    d = np.dot(v - plane_point, plane_normal)

    # 计算投影点
    v_proj = v - d * plane_normal

    return v_proj
