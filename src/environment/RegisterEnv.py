try:
    import gymnasium as gym
except ModuleNotFoundError:
    import gym


def register_env():
    if not gym.envs.registry.get("SatelliteEnv3v1"):
        gym.register(
            id="SatelliteEnv3v1",
            entry_point="environment.SatelliteEnv3v1:SatelliteEnv3v1",
        )

    if not gym.envs.registry.get("SatelliteEnv2v2"):
        gym.register(
            id="SatelliteEnv2v2",
            entry_point="environment.SatelliteEnv2v2:SatelliteEnv2v2",
        )

    if not gym.envs.registry.get("SatelliteEnv3v1Hvt"):
        gym.register(
            id="SatelliteEnv3v1Hvt",
            entry_point="environment.SatelliteEnv3v1Hvt:SatelliteEnv3v1Hvt",
        )

    if not gym.envs.registry.get("SatelliteEnvPreyNvN"):
        gym.register(
            id="SatelliteEnvPreyNvN",
            entry_point="environment.SatelliteEnvPreyNvN:SatelliteEnvPreyNvN",
        )

    if not gym.envs.registry.get("SatelliteEnvInterceptNvN"):
        gym.register(
            id="SatelliteEnvInterceptNvN",
            entry_point="environment.SatelliteEnvInterceptNvN:SatelliteEnvInterceptNvN",
        )
