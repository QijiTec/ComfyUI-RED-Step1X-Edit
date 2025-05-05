import platform
import sys
import importlib.util

import torch

def is_flash_attn_available():
    """检查 flash_attn 是否可用"""
    spec = importlib.util.find_spec("flash_attn")
    return spec is not None

def get_available_attention_mode(requested_mode="flash"):
    """
    根据请求的注意力模式和系统可用性返回实际可用的注意力模式
    
    Args:
        requested_mode (str): 请求使用的注意力模式 ("flash", "torch", "vanilla")
    
    Returns:
        str: 实际可用的注意力模式
    """
    if requested_mode == "flash" and not is_flash_attn_available():
        print("警告：FlashAttention2 不可用，将自动切换到 torch SDPA 模式。要使用 FlashAttention2，请安装 flash-attn 包。")
        return "torch"
    return requested_mode


def get_cuda_version():
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
        return f"cu{cuda_version.replace('.', '')[:2]}"  # 例如：cu121
    return "cpu"


def get_torch_version():
    return f"torch{torch.__version__.split('+')[0]}"[:-2]  # 例如：torch2.2


def get_python_version():
    version = sys.version_info
    return f"cp{version.major}{version.minor}"  # 例如：cp310


def get_abi_flag():
    return "abiTRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "abiFALSE"


def get_platform():
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "linux" and machine == "x86_64":
        return "linux_x86_64"
    elif system == "windows" and machine == "amd64":
        return "win_amd64"
    elif system == "darwin" and machine == "x86_64":
        return "macosx_x86_64"
    else:
        raise ValueError(f"Unsupported platform: {system}_{machine}")


def generate_flash_attn_filename(flash_attn_version="2.7.2.post1"):
    cuda_version = get_cuda_version()
    torch_version = get_torch_version()
    python_version = get_python_version()
    abi_flag = get_abi_flag()
    platform_tag = get_platform()

    filename = (
        f"flash_attn-{flash_attn_version}+{cuda_version}{torch_version}cxx11{abi_flag}-"
        f"{python_version}-{python_version}-{platform_tag}.whl"
    )
    return filename


if __name__ == "__main__":
    try:
        filename = generate_flash_attn_filename()
        print("Generated filename:", filename)
    except Exception as e:
        print("Error generating filename:", e)