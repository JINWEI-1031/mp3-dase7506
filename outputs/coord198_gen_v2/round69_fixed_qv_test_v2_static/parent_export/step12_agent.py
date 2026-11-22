"""Probe one scoped absence candidate after the fixed projection baseline."""
import json
from pathlib import Path

from step4_agent import passed
from step4_edits import Schema
from step10_agent import create_agent as step10
from step12_absence import classify, pool


class Step12Agent:
    def __init__(self, baseline, settings):
        self.baseline = baseline
        self.settings = settings
        self.fallback = None
        self.audit = {"pool": []}

    def act(self, obs):
        history = obs["history"]
        if self.fallback is not None:
            chosen = self.audit["selected"]
            if passed(history[-1]["observation"]):
                return {"tool": "submit", "sql": chosen["sql"]}
            return self.fallback
        action = self.baseline.act(obs)
        if action["tool"] != "submit":
            return action
        match, reason = classify(obs["task"]["question"])
        self.audit["scope"] = reason
        if match is None:
            self.audit["reason"] = reason
            return action
        if any(e["tool"] == "check" and passed(e["observation"]) for e in history):
            self.audit["reason"] = "existing public pass"
            return action
        tables = next((e["observation"] for e in history if e["tool"] == "schema"), None)
        if tables is None:
            self.audit["reason"] = "schema unavailable"
            return action
        options = pool(obs, Schema(tables), action["sql"], self.settings["mode"], match)
        order = sorted(options, key=lambda c: -c["priority"]) if self.settings["ranked"] else options
        self.audit.update(pool=options, order=[c["sql"] for c in order], remaining=obs["remaining"])
        if not order or not obs["remaining"]:
            self.audit["reason"] = "no candidate" if not order else "budget exhausted"
            return action
        self.audit["selected"] = order[0]
        self.fallback = action
        return {"tool": "check", "sql": order[0]["sql"]}


def create_agent(checkpoint):
    path = Path(checkpoint).resolve()
    settings = json.loads(path.read_text(encoding="utf-8"))
    if (set(settings) != {"mode", "ranked"} or settings["mode"] not in {"exists", "except"}
            or type(settings["ranked"]) is not bool):
        raise ValueError("Invalid step 12 configuration")
    return Step12Agent(step10(path.with_name("step10-projection.json")), settings)
