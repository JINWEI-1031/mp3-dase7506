"""Reuse the step23 probe and check timing across both precision policies."""
import json
from pathlib import Path

from step21_agent import create_agent as step21
from step23_agent import Step23Agent
from step24_precision import generate

ARMS = ("exact", "minute")


class Step24Agent(Step23Agent):
    def __init__(self, baseline, arm):
        super().__init__(baseline, "sample")
        self.arm = arm

    def evaluate(self, obs, sample=None):
        options = generate(obs, self.schema, self.probe, sample, self.arm)
        self.audit["pool"] = options
        if not options or not obs["remaining"]:
            return self.resume(obs)
        selected = options[0]
        sql = selected["sql"]
        self.audit.update(checked=[sql], trigger=True, selected_rule=selected["rule"])
        self.phase = "repair"
        return {"tool": "check", "sql": sql}


def create_agent(checkpoint):
    path = Path(checkpoint).resolve()
    arm = json.loads(path.read_text(encoding="utf-8"))["arm"]
    if arm not in ARMS:
        raise ValueError(arm)
    return Step24Agent(step21(path.with_name("step21-scoped.json")), arm)
