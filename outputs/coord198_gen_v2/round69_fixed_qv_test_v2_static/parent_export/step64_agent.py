"""Try E once after a failed public D check, sharing the loaded Qwen base."""

from functools import lru_cache
import os
from pathlib import Path
import time

from step63_agent import (ModelSuffix, active_model, adapter_metadata, create_agent as create_d,
                          generate_with)


@lru_cache(maxsize=1)
def backup_model():
    tokenizer, model, base_info = active_model()
    wall, cpu = time.perf_counter(), time.process_time()
    path = Path(os.environ["MP3_BACKUP_ADAPTER_DIR"])
    info = dict(base_info, adapter=adapter_metadata(path, "E"))
    model.load_adapter(path, adapter_name="E", is_trainable=False, local_files_only=True)
    info["parameters"] = sum(p.numel() for p in model.parameters())
    if info["parameters"] > 1_000_000_000:
        raise ValueError("Parameter limit exceeded")
    info["load_wall_seconds"] = time.perf_counter() - wall
    info["load_cpu_seconds"] = time.process_time() - cpu
    return tokenizer, model, info


def generate_e(observation, schema):
    bundle = backup_model()
    model = bundle[1]
    model.set_adapter("E")
    try:
        return generate_with(observation, schema, bundle)
    finally:
        model.set_adapter("default")
        model.requires_grad_(False)


class Agent:
    def __init__(self, baseline, proposer=generate_e):
        self.baseline = baseline
        self.proposer = proposer
        self.backup = None

    def act(self, observation):
        if self.backup is not None:
            return self.backup.act(observation)
        action = self.baseline.act(observation)
        primary = self.baseline.suffix
        if (action["tool"] == "submit" and primary is not None
                and primary.reason == "public_failed" and observation["remaining"]):
            self.backup = ModelSuffix(action, self.proposer)
            return self.backup.act(observation)
        return action


def create_agent(checkpoint=None):
    return Agent(create_d(checkpoint))
