"""Explicit execution devices and isolated reproducible random streams."""
from contextlib import contextmanager
import re
import torch


def resolve_device(value="cpu"):
    text=str(value)
    if text=="cpu":return torch.device("cpu")
    if not re.fullmatch(r"cuda(?::[0-9]+)?",text):
        raise ValueError("Device must be cpu, cuda or cuda:N.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but this PyTorch installation has no available CUDA device.")
    index=torch.cuda.current_device() if text=="cuda" else int(text.split(":")[1])
    if index>=torch.cuda.device_count():raise ValueError(f"CUDA device index {index} is not available.")
    return torch.device("cuda",index)


def seed_streams(seed,device):
    torch.random.default_generator.manual_seed(int(seed))
    if device.type=="cuda":torch.cuda.default_generators[device.index].manual_seed(int(seed))


@contextmanager
def isolated_rng(device,seed=None):
    device=resolve_device(device)
    with torch.random.fork_rng(devices=[device.index] if device.type=="cuda" else []):
        if seed is not None:seed_streams(seed,device)
        yield device


def cpu_state(state):
    return {name:value.detach().to("cpu").clone() for name,value in state.items()}


def close_posterior(posterior):
    
    try:
        close=getattr(posterior,"close",None)
        if close is not None:close()
    finally:
        model=getattr(posterior,"model",None)
        if model is not None and getattr(model,"_exact_posterior",None) is posterior:
            object.__delattr__(model,"_exact_posterior")
