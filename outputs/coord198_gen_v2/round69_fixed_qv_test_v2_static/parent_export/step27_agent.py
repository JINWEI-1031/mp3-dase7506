"""Append one evidence-backed check to a publicly passing step25 submission."""
import json
from pathlib import Path

from repair_rules import parse
from step4_agent import identity, passed
from step19_agent import public_submission
from step25_agent import create_agent as step25
from step27_grain import candidate_pool, schema_signal


class Step27Agent:
    def __init__(self, baseline):
        self.baseline = baseline
        self.fallback = None
        self.audit = {"pool": [], "checked": [], "trigger": False, "eligible": False}

    def act(self, obs):
        history = obs["history"]
        remaining = obs["remaining"]
        if self.audit["checked"]:
            if passed(history[-1]["observation"]):
                return {"tool": "submit", "sql": self.audit["checked"][0]}
            return self.fallback

        action = self.fallback or self.baseline.act(obs)
        if action["tool"] != "submit" or not public_submission(history, action["sql"]):
            return action
        self.audit["eligible"] = True
        sql = action["sql"]
        if not schema_signal(parse(sql), obs["task"]["question"], obs["example"]["columns"]):
            self.audit["reason"] = "no_explicit_grain_signal"
            return action
        tables = next((e["observation"] for e in history if e["tool"] == "schema"), None)
        if tables is None:
            if remaining < 2:
                self.audit["reason"] = "insufficient_budget_for_schema_and_check"
                return action
            self.fallback = action
            self.audit["schema_requested"] = True
            return {"tool": "schema"}
        pool = candidate_pool(obs, sql, tables)
        previous = {identity(e["sql"]) for e in history if e["tool"] == "check"}
        self.audit.update(pool=pool, already_checked=[c["sql"] for c in pool if identity(c["sql"]) in previous])
        choice = next((c for c in pool if identity(c["sql"]) not in previous), None)
        if not choice or not remaining:
            self.audit["reason"] = "budget_exhausted" if choice else "no_unchecked_grounded_candidate"
            return action
        self.fallback = action
        self.audit.update(checked=[choice["sql"]], trigger=True, selected=choice)
        return {"tool": "check", "sql": choice["sql"]}


def create_agent(checkpoint):
    path = Path(checkpoint).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if config != {"policy": "public-grain-check"}:
        raise ValueError("Unknown step27 policy")
    return Step27Agent(step25(path.with_name("step25-exists.json")))
