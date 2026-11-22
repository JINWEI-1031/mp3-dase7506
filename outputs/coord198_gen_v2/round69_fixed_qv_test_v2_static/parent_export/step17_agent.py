"""One explicit-conflict check after a publicly passing submission."""
import json
from pathlib import Path

from step4_agent import identity, passed
from step4_edits import Schema
from step16_agent import create_agent as step16
from step17_conflicts import conflicts


class Step17Agent:
    def __init__(self, baseline, settings):
        self.baseline = baseline
        self.settings = settings
        self.fallback = None
        self.audit = {"pool": []}

    def act(self, obs):
        history = obs["history"]
        if self.fallback is not None:
            if passed(history[-1]["observation"]):
                return {"tool": "submit", "sql": self.audit["selected"]["sql"]}
            return self.fallback
        action = self.baseline.act(obs)
        if action["tool"] != "submit":
            return action
        if not any(e["tool"] == "check" and passed(e["observation"])
                   and identity(e["sql"]) == identity(action["sql"]) for e in history):
            self.audit["reason"] = "baseline submission did not pass public check"
            return action
        tables = next((e["observation"] for e in history if e["tool"] == "schema"), None)
        pool = conflicts(obs, action["sql"], self.settings["families"], Schema(tables) if tables else None)
        order = sorted(pool, key=lambda c: -c["priority"]) if self.settings["ranked"] else pool
        self.audit.update(pool=pool, order=[c["sql"] for c in order], remaining=obs["remaining"])
        if not order or not obs["remaining"]:
            self.audit["reason"] = "no conflict candidate" if not order else "budget exhausted"
            return action
        self.audit["selected"] = order[0]
        self.fallback = action
        return {"tool": "check", "sql": order[0]["sql"]}


def create_agent(checkpoint):
    path = Path(checkpoint).resolve()
    settings = json.loads(path.read_text(encoding="utf-8"))
    if (set(settings) != {"families", "ranked"} or not settings["families"]
            or not set(settings["families"]) <= {"ordering", "predicate", "join"} or type(settings["ranked"]) is not bool):
        raise ValueError("Invalid step 17 configuration")
    return Step17Agent(step16(path.with_name("step16-supported.json")), settings)
