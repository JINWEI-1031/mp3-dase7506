"""Retain round 42 and add round 26 before submission on unmatched paths."""
import json
from pathlib import Path

from step4_agent import passed
from step25_agent import create_agent as create_baseline
from step26_agent import Step26Agent
from step42_agent import PairedSuffix


ARMS = {"append": "schema", "replace": "replace"}


class Agent:
    def __init__(self, baseline, arm):
        self.repair = Step26Agent(baseline, ARMS[arm])
        self.suffix = None

    def act(self, observation):
        if self.suffix is not None:
            return self.suffix.act(observation)
        action = self.repair.act(observation)
        if action["tool"] != "submit":
            return action
        if self.repair.audit["checked"] and passed(observation["history"][-1]["observation"]):
            # The added check has already found a matching repair.
            return action
        self.suffix = PairedSuffix(action, "control")
        return self.suffix.act(observation)


def create_agent(checkpoint=None):
    path = Path(checkpoint).resolve() if checkpoint else Path(__file__).with_name("configs") / "step47-replace.json"
    arm = json.loads(path.read_text(encoding="utf-8"))["arm"]
    return Agent(create_baseline(path.with_name("step25-exists.json")), arm)
