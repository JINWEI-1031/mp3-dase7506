"""One ordering probe after the publicly passing step17 submission."""
import json
from pathlib import Path

from step4_agent import identity, passed
from step4_edits import Schema
from step17_agent import create_agent as step17
from step18_ordering import pool


class Step18Agent:
    def __init__(self, baseline, policy):
        self.baseline = baseline
        self.policy = policy
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
        sql = action["sql"]
        if not any(e["tool"] == "check" and passed(e["observation"])
                   and identity(e["sql"]) == identity(sql) for e in history):
            self.audit["reason"] = "baseline submission did not pass public check"
            return action
        tables = next((e["observation"] for e in history if e["tool"] == "schema"), None)
        options = pool(obs, sql, Schema(tables) if tables else None, self.policy != "replay")
        order = sorted(options, key=lambda c: -c["priority"], reverse=self.policy == "event-first")
        self.audit.update(pool=options, order=[c["sql"] for c in order], remaining=obs["remaining"])
        if not order or not obs["remaining"]:
            self.audit["reason"] = "no ordering conflict" if not order else "budget exhausted"
            return action
        self.audit["selected"] = order[0]
        self.fallback = action
        return {"tool": "check", "sql": order[0]["sql"]}


def create_agent(checkpoint):
    path = Path(checkpoint).resolve()
    settings = json.loads(path.read_text(encoding="utf-8"))
    if set(settings) != {"policy"} or settings["policy"] not in {"replay", "row-first", "event-first"}:
        raise ValueError("Invalid step 18 configuration")
    return Step18Agent(step17(path.with_name("step17-predicate.json")), settings["policy"])
