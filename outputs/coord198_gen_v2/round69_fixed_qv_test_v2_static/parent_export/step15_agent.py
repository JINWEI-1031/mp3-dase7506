"""Probe one relation-role repair after the fixed step14-exists baseline."""
import json
from pathlib import Path

from step4_agent import passed
from step4_edits import Schema
from step14_agent import create_agent as step14
from step15_roles import relation_pool


class Step15Agent:
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
        if any(e["tool"] == "check" and passed(e["observation"]) for e in history):
            self.audit["reason"] = "existing public pass"
            return action
        tables = next((e["observation"] for e in history if e["tool"] == "schema"), None)
        if tables is None:
            self.audit["reason"] = "schema unavailable"
            return action
        pool, counts = relation_pool(obs, Schema(tables), action["sql"], self.settings["mode"])
        order = sorted(pool, key=lambda c: -c["priority"]) if self.settings["ranked"] else pool
        self.audit.update(pool=pool, order=[c["sql"] for c in order], counts=counts, remaining=obs["remaining"])
        if not order or not obs["remaining"]:
            self.audit["reason"] = "no candidate" if not order else "budget exhausted"
            return action
        self.audit["selected"] = order[0]
        self.fallback = action
        return {"tool": "check", "sql": order[0]["sql"]}


def create_agent(checkpoint):
    path = Path(checkpoint).resolve()
    settings = json.loads(path.read_text(encoding="utf-8"))
    if (set(settings) != {"mode", "ranked"} or settings["mode"] not in {"skeleton", "revision"}
            or type(settings["ranked"]) is not bool):
        raise ValueError("Invalid step 15 configuration")
    return Step15Agent(step14(path.with_name("step14-exists.json")), settings)
