"""Run one matched row or entity-scope check in the unpassed branch."""
import json
from pathlib import Path

from step4_agent import identity, passed
from step4_edits import Schema
from step24_agent import create_agent as step24
from step25_absence import candidate_pool, negative_span

ARMS = ("row", "exists")


class Step25Agent:
    def __init__(self, baseline, arm):
        self.baseline = baseline
        self.arm = arm
        self.fallback = None
        self.audit = {"pool": [], "checked": [], "trigger": False, "eligible": False}

    def act(self, obs):
        history, remaining = obs["history"], obs["remaining"]
        checked = self.audit["checked"]
        if checked:
            return {"tool": "submit", "sql": checked[0]} if passed(history[-1]["observation"]) else self.fallback
        action = self.baseline.act(obs)
        inherited = self.baseline.baseline
        replace = (action["tool"] == "check" and remaining == 1 and not self.baseline.audit["eligible"]
                   and inherited.schedule is not None and inherited.audit["checked"][-1:] == [action["sql"]])
        if not (replace or action["tool"] == "submit"):
            return action
        if any(e["tool"] == "check" and passed(e["observation"]) for e in history):
            return action
        if not negative_span(obs["task"]["question"]):
            return action
        tables = next((e["observation"] for e in history if e["tool"] == "schema"), None)
        if tables is None:
            return action
        fallback = inherited.fallback if replace else action
        extra = [*(c["sql"] for c in inherited.audit["pool"][:12]), action["sql"]]
        pool = candidate_pool(obs, Schema(tables), fallback["sql"], self.arm, extra)
        previous = {identity(e["sql"]) for e in history if e["tool"] == "check"}
        known = [c["sql"] for c in pool if identity(c["sql"]) in previous]
        available = [c for c in pool if identity(c["sql"]) not in previous]
        self.audit.update(eligible=True, pool=pool, already_checked=known, available=len(available))
        if not available or not remaining:
            return action
        self.fallback = fallback
        self.audit.update(checked=[available[0]["sql"]], trigger=True)
        if replace:
            self.audit["replaced"] = action["sql"]
        return {"tool": "check", "sql": available[0]["sql"]}


def create_agent(checkpoint):
    path = Path(checkpoint).resolve()
    arm = json.loads(path.read_text(encoding="utf-8"))["arm"]
    if arm not in ARMS:
        raise ValueError(arm)
    return Step25Agent(step24(path.with_name("step24-minute.json")), arm)
