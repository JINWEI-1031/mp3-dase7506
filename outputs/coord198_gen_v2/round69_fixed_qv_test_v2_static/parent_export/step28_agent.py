"""Check one role repair after a publicly passing baseline submission."""
import json
from pathlib import Path

from step4_agent import identity, passed
from step19_agent import public_submission
from step25_agent import create_agent as step25
from step28_roles import candidate_pool, shape


class Step28Agent:
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
        plan = shape(obs, sql)
        if not plan:
            self.audit["reason"] = "no_observed_role_conflict"
            return action
        self.audit["cues"] = plan[1]
        tables = next((e["observation"] for e in history if e["tool"] == "schema"), None)
        if tables is None:
            if remaining < 2:
                self.audit["reason"] = "insufficient_budget_for_schema_and_check"
                return action
            self.fallback = action
            self.audit["schema_requested"] = True
            return {"tool": "schema"}
        pool = candidate_pool(obs, sql, tables, plan)
        prior = {identity(e["sql"]) for e in history if e["tool"] == "check"}
        self.audit.update(pool=pool, already_checked=[c["sql"] for c in pool if identity(c["sql"]) in prior])
        choice = next((c for c in pool if identity(c["sql"]) not in prior), None)
        if not choice or not remaining:
            self.audit["reason"] = "budget_exhausted" if choice else "no_declared_role_repair"
            return action
        self.fallback = action
        self.audit.update(checked=[choice["sql"]], trigger=True, selected=choice)
        return {"tool": "check", "sql": choice["sql"]}


def create_agent(checkpoint):
    path = Path(checkpoint).resolve()
    if json.loads(path.read_text(encoding="utf-8")) != {"policy": "public-role-check"}:
        raise ValueError("Unknown step28 policy")
    return Step28Agent(step25(path.with_name("step25-exists.json")))
