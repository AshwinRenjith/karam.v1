from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class DeviceInfo:
    device: torch.device
    is_mps: bool
    is_cuda: bool


def get_default_device() -> DeviceInfo:
    """Selects MPS if available, else CUDA, else CPU."""
    if torch.backends.mps.is_available():
        return DeviceInfo(device=torch.device("mps"), is_mps=True, is_cuda=False)
    if torch.cuda.is_available():
        return DeviceInfo(device=torch.device("cuda"), is_mps=False, is_cuda=True)
    return DeviceInfo(device=torch.device("cpu"), is_mps=False, is_cuda=False)


if __name__ == "__main__":
    info = get_default_device()
    print(f"device={info.device}, is_mps={info.is_mps}, is_cuda={info.is_cuda}")
