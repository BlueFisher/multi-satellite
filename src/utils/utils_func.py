import argparse
import importlib
import os
import platform
from argparse import Namespace

import yaml


def load_config(file_path: str) -> argparse.Namespace:
    """
    加载 YAML 配置文件并返回命名空间对象。

    Args:
        file_path (str): 配置文件的路径。

    Returns:
        argparse.Namespace: 包含配置的命名空间对象。

    Example:
        >>> config = load_config("config.yaml")
        >>> print(config.some_key)
    """
    arg = {}
    if os.path.exists(file_path):
        file = open(file_path, "r", encoding="utf-8")
        arg = yaml.load(file, Loader=yaml.FullLoader)
        if arg is None:
            arg = {}
    else:
        print(f"Warning: config file {file_path} not found")
    return argparse.Namespace(**arg)


def update_args(config: argparse.Namespace, path) -> argparse.Namespace:
    """
    更新命名空间对象的参数。

    Args:
        config (argparse.Namespace): 原始配置。
        path (str): 新配置文件的路径。

    Returns:
        argparse.Namespace: 更新后的配置。

    Example:
        >>> updated_config = update_args(config, "new_config.yaml")
    """
    new_cfg = load_config(path)
    for key, value in vars(new_cfg).items():
        setattr(config, key, value)

    return config


def str2bool(v):
    """
    将字符串转换为布尔值。

    Args:
        v: 输入值，可以是字符串或布尔值。

    Returns:
        bool: 转换后的布尔值。

    Raises:
        argparse.ArgumentTypeError: 如果输入值不是布尔值。

    Example:
        >>> result = str2bool("true")
        >>> print(result)
        True
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def load_args(cfg: Namespace):
    """
    加载命令行参数并更新配置。

    Args:
        cfg (Namespace): 原始配置。

    Returns:
        Namespace: 更新后的配置。

    Example:
        >>> args = load_args(cfg)
    """
    parser = argparse.ArgumentParser()

    # 支持任意已有 cfg 字段
    for key, value in vars(cfg).items():
        if isinstance(value, bool):
            parser.add_argument(f"--{key}", default=value, type=str2bool)
        elif isinstance(value, type(None)):
            parser.add_argument(f"--{key}", default=value, type=str)
        else:
            parser.add_argument(f"--{key}", default=value, type=type(value))

    # 解析所有参数，包括未知参数
    args, unknown = parser.parse_known_args()

    # 用命令行参数更新 cfg
    for key, value in vars(args).items():
        setattr(cfg, key, value)

    # 解析未知参数（--key value 形式），加入到 cfg
    i = 0
    while i < len(unknown):
        if unknown[i].startswith("--"):
            key = unknown[i][2:]
            if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                val = unknown[i + 1]
                # 尝试推断类型
                if val.lower() in ("true", "false"):
                    val = str2bool(val)
                else:
                    try:
                        val = int(val)
                    except ValueError:
                        try:
                            val = float(val)
                        except ValueError:
                            pass  # 保持字符串
                setattr(cfg, key, val)
                i += 2
            else:
                setattr(cfg, key, True)  # 单独的 flag 参数
                i += 1
        else:
            i += 1

    return cfg


def get_alg_manager(alg_name: str):
    """
    根据算法名称加载算法管理器。

    Args:
        alg_name (str): 算法名称。

    Returns:
        object: 算法管理器实例。

    Example:
        >>> manager = get_alg_manager("MADDPG")
    """
    try:
        alg_manger_name = "algorithms." + alg_name
        alg_manger = importlib.import_module(alg_manger_name)
        alg = getattr(alg_manger, alg_name)
    except Exception as e:
        print("Could not load factory class " + alg_name)
        print(e.with_traceback(e.__traceback__))
        exit(1)

    return alg


def get_env_class(env_group: str):
    """
    根据环境组名称加载环境类。

    Args:
        env_group (str): 环境组名称。

    Returns:
        object: 环境类。

    Example:
        >>> env_class = get_env_class("SatelliteEnv")
    """
    try:
        env_lib = importlib.import_module("environment." + env_group)
        env_class = getattr(env_lib, env_group)
    except Exception as e:
        print("Could not load environment class " + env_group)
        exit(1)

    return env_class


def is_linux() -> bool:
    """
    检查当前操作系统是否为 Linux。

    Returns:
        bool: 如果是 Linux 返回 True，否则返回 False。

    Example:
        >>> result = is_linux()
        >>> print(result)
        True
    """
    return platform.system() == "Linux"
