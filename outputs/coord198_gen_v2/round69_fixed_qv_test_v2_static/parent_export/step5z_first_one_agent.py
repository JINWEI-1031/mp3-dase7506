"""Cap the additional empty-draft branch at one check."""
from pathlib import Path

from step5_agent import load_policy
from step5x_agent import ExpansionAgent
from step5z_agent import TargetedAgent


class FirstHintAgent(TargetedAgent):
    def __init__(self, base, policy, settings):
        super().__init__(base, policy, settings)
        self.baseline_continues = False

    def act(self, observation):
        if self.baseline_continues:
            return ExpansionAgent.act(self, observation)
        if len(observation["history"]) == 1:
            # Give any existing baseline continuation the whole episode.
            action = ExpansionAgent.act(self, observation)
            if action["tool"] != "submit":
                self.baseline_continues = True
                self.audit["empty"] = {"reason": "baseline continuation preserved"}
                return action
        return super().act(observation)


def create_agent(checkpoint):
    import json

    path = Path(checkpoint).resolve()
    settings = json.loads(path.read_text(encoding="utf-8"))
    if settings != {"empty": "first", "limit": 1}:
        raise ValueError("Expected first-passing hint with a one-check limit")
    return FirstHintAgent(*load_policy(path.with_name("step5-cpu_s0_d0.json")), settings)
