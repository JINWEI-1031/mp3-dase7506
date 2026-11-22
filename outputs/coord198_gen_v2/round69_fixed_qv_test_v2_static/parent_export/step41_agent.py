"""Repair independent filters while preserving the outer SELECT structure."""
from pathlib import Path

from repair_rules import parse
from step4_agent import identity, passed
from step19_agent import public_submission
from step25_agent import create_agent as create_baseline
from step30_multiplicity import candidate_pool, scope_plan


def output_shape(sql):
    tree = parse(sql)
    tree.set("where", None)
    return tree


class FilterAgent:
    def __init__(self, baseline):
        self.baseline = baseline
        self.fallback = None
        self.checked = None
        self.pool = []

    def act(self, observation):
        history = observation["history"]
        if self.checked is not None:
            if passed(history[-1]["observation"]):
                return {"tool": "submit", "sql": self.checked}
            return self.fallback

        action = self.fallback or self.baseline.act(observation)
        remaining = observation["remaining"]
        if action["tool"] != "submit" or not remaining:
            return action
        sql = action["sql"]
        if not public_submission(history, sql):
            return action
        plan = scope_plan(observation, sql, "preserve")
        if plan is None:
            return action
        tables = next((event["observation"] for event in history if event["tool"] == "schema"), None)
        if tables is None:
            if remaining >= 2:
                self.fallback = action
                return {"tool": "schema"}
            return action

        shape = output_shape(sql)
        self.pool = [candidate for candidate in candidate_pool(observation, sql, tables, plan)
                     if output_shape(candidate["sql"]) == shape]
        previous = {identity(event["sql"]) for event in history if event["tool"] in ("check", "execute")}
        for candidate in self.pool:
            if identity(candidate["sql"]) not in previous:
                self.fallback = action
                self.checked = candidate["sql"]
                return {"tool": "check", "sql": self.checked}
        return action


def create_agent(checkpoint=None):
    config = checkpoint or Path(__file__).with_name("configs") / "step25-exists.json"
    return FilterAgent(create_baseline(config))
