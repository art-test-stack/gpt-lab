import os

import pytest
import torch


@pytest.fixture(scope="session")
def native_extension():
    from gpt_lab.rust_tools.dataloader import _extension

    if _extension(required=False) is None:
        if os.environ.get("GPT_LAB_REQUIRE_NATIVE") == "1":
            pytest.fail("Install the rust-dataloader extra before running native tests")
        pytest.skip("Optional rust-dataloader extra is not installed")


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
    from gpt_lab.utils.common import get_banner
    from gpt_lab.utils.system import (
        get_git_info, 
        get_gpu_info, 
        get_system_info
    )

    banner = get_banner()

    def format_info(title, info):
        lines = [f"{title}:"]
        for k, v in info.items():
            lines.append(f"  {k:<18}: {v:<68}")
        return "\n".join(lines)

    return "\n\n".join([
        "\n" +"=" * 80,
        banner,
        "=" * 80,
        format_info("Git Information", get_git_info()),
        format_info("System Information", get_system_info()),
        format_info("GPU Information", get_gpu_info()),
        "=" * 80 + "\n"
    ])
