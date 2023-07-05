"""
"""

import os

import psutil
import pynvml
import torch


divisors = {
    "B": 1,
    "K": 1024,
    "M": 1024 ** 2,
    "G": 1024 ** 3,
    "T": 1024 ** 4,
}


def cpu_mem(fmt: str = "G", digits: int = 2) -> str:
    b = psutil.Process(os.getpid()).memory_info().rss
    m = b / divisors[fmt]
    return f"{round(m, digits)}{fmt}"


def gpu_summary(device: int = None, abbreviated: bool = True) -> dict[int, str]:
    if not torch.cuda.is_initialized():
        return {-1: "No GPU initialized."}

    if device:
        return {device: torch.cuda.memory_summary(torch.device(device), abbreviated)}
    
    r = {}
    for device in range(torch.cuda.device_count()):
        r[device] = torch.cuda.memory_summary(torch.device(device), abbreviated)
    if not r:
        return {-1: "No GPU found."}
    return r


def gpu_usage(device: int = None) -> str:
    if not torch.cuda.is_initialized():
        return {-1: "No GPU initialized."}

    if device:
        util = torch.cuda.utilization(torch.device(device))
        mem = torch.cuda.memory_usage(torch.device(device))
        return {device: f"{util=}% | {mem=}%"}
    
    r = {}
    for device in range(torch.cuda.device_count()):
        util = torch.cuda.utilization(torch.device(device))
        mem = torch.cuda.memory_usage(torch.device(device))
        r[device] = f"{util=}% | {mem=}%"
    if not r:
        return {-1: "No GPU found."}
    return r


def status(device: int = None) -> str:
    if device == "cpu" or device == torch.device("cpu"):
        device = None
    return (
        f"{cpu_mem()=}\n"
        f"{gpu_summary(device)=}\n"
        f"{gpu_usage(device)=}"
    )
