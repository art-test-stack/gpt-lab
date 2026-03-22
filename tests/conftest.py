import pytest
import torch


def _available_devices():
    devices = ["cpu"]

    if torch.cuda.is_available():
        devices.append("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append("mps")

    return devices


@pytest.fixture(params=_available_devices())
def device(request):
    return torch.device(request.param)

def pytest_report_header(config):
    from gpt_lib.utils.report import (
        get_banner, 
        get_git_info, 
        get_gpu_info, 
        get_system_info
    )

    banner = get_banner()

    def format_info(title, info):
        lines = [f"{title}:"]
        for k, v in info.items():
            lines.append(f"  {k:<18}: {v}")
        return "\n".join(lines)

    return "\n\n".join([
        "=" * 60,
        banner,
        "=" * 60,
        format_info("Git Information", get_git_info()),
        format_info("System Information", get_system_info()),
        format_info("GPU Information", get_gpu_info()),
    ])