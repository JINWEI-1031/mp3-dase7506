"""Append bounded semantic repairs to the step21-scoped submission."""
import json
from pathlib import Path

from step4_agent import passed
from step4_edits import Schema
from step19_agent import public_submission
from step21_agent import create_agent as step21
from step22_semantics import candidate_pool, unit_replacements

ARMS = {
    "typed": (("typed",), 1),
    "scope": (("scope",), 1),
    "joint": (("typed", "scope"), 1),
    "joint3": (("typed", "scope"), 3),
    "observed": (("typed", "scope", "observed"), 3),
    "reserve": (("typed", "scope", "observed"), 3),
}


class Step22Agent:
    def __init__(self, baseline, arm):
        self.baseline = baseline
        self.reserve = arm == "reserve"
        self.modes, self.max_checks = ARMS[arm]
        self.fallback = None
        self.audit = {"pool": [], "checked": [], "trigger": False}
        self.generated = False

    def act(self, obs):
        history, remaining = obs["history"], obs["remaining"]
        checked = self.audit["checked"]
        if checked and passed(history[-1]["observation"]):
            return {"tool": "submit", "sql": checked[-1]}
        if checked and self.audit.get("replacement"):
            return self.fallback
        if self.fallback is None:
            action = self.baseline.act(obs)
            if action["tool"] != "submit":
                if (self.reserve and remaining == 1 and action["tool"] == "check"
                        and getattr(self.baseline, "schedule", None) is not None
                        and self.baseline.audit["checked"][-1:] == [action["sql"]]):
                    tables = next((e["observation"] for e in history if e["tool"] == "schema"), None)
                    if tables:
                        candidates = unit_replacements(obs, action["sql"], Schema(tables))
                        if candidates:
                            self.fallback = self.baseline.fallback
                            sql = candidates[0]["sql"]
                            checked.append(sql)
                            self.audit.update(pool=candidates, trigger=True, replacement=action["sql"])
                            return {"tool": "check", "sql": sql}
                return action
            self.fallback = action
        action = self.fallback
        if public_submission(history, action["sql"]):
            self.audit["reason"] = "baseline_public"
            return action
        if not self.generated:
            tables = next((e["observation"] for e in history if e["tool"] == "schema"), None)
            if tables is None and remaining >= 2:
                self.audit.update(trigger=True, schema_requested=True)
                return {"tool": "schema"}
            if tables is None:
                self.audit["reason"] = "schema_unavailable"
                return action
            previous = getattr(self.baseline, "audit", {}).get("pool", [])
            self.audit.update(pool=candidate_pool(obs, action["sql"], Schema(tables), self.modes, previous))
            self.generated = True
        pool = self.audit["pool"]
        if remaining and len(checked) < min(self.max_checks, len(pool)):
            sql = pool[len(checked)]["sql"]
            checked.append(sql)
            self.audit["trigger"] = True
            return {"tool": "check", "sql": sql}
        self.audit["reason"] = ("no_candidate" if not pool else "budget"
                                if not remaining else "checks_completed")
        return action


def create_agent(checkpoint):
    path = Path(checkpoint).resolve()
    arm = json.loads(path.read_text(encoding="utf-8"))["arm"]
    return Step22Agent(step21(path.with_name("step21-scoped.json")), arm)
