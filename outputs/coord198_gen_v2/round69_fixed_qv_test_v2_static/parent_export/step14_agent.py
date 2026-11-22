"""Probe at most one ranked-group exclusion after step13-feedback."""
import json
from pathlib import Path

from step4_agent import passed
from step4_edits import Schema
from step13_agent import create_agent as step13
from step14_groups import intent, pool


class Step14Agent:
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
        request = intent(obs["task"]["question"])
        self.audit["trigger"] = request is not None
        if request is None:
            self.audit["reason"] = "outside ranked membership"
            return action
        if any(e["tool"] == "check" and passed(e["observation"]) for e in history):
            self.audit["reason"] = "existing public pass"
            return action
        if self.baseline.audit.get("selected"):
            self.audit["reason"] = "baseline category probe already used"
            return action
        if obs.get("ordered"):
            self.audit["reason"] = "ordered output outside scope"
            return action
        tables = next((e["observation"] for e in history if e["tool"] == "schema"), None)
        if tables is None:
            self.audit["reason"] = "schema unavailable"
            return action
        candidates = pool(obs, Schema(tables), action["sql"], self.settings["mode"], request)
        order = sorted(candidates, key=lambda c: -c["priority"]) if self.settings["ranked"] else candidates
        self.audit.update(pool=candidates, order=[c["sql"] for c in order], remaining=obs["remaining"])
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
        raise ValueError("Invalid step 14 configuration")
    return Step14Agent(step13(path.with_name("step13-feedback.json")), settings)
