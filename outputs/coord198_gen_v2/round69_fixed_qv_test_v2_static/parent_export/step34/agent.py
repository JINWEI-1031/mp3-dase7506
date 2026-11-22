"""Quantified-plan check and isolated optional replacement of the eighth check."""
from itertools import islice
import json
from pathlib import Path

from repair_rules import parse
from step4_agent import identity, passed
from step25_agent import create_agent as step25
from step32_plan import compiled, context
from step34.quantities import discover, plans, schedule_pool

ARMS = ("append", "replace")


class Step34Agent:
    def __init__(self, baseline, arm):
        self.baseline, self.arm = baseline, arm
        self.fallback = None
        self.audit = {"pool": [], "checked": [], "already_checked": [], "eligible": False,
                      "trigger": False, "schema_requested": False, "schema_only_sql": [],
                      "fixed_slot_pool": []}

    def act(self, obs):
        history, remaining = obs["history"], obs["remaining"]
        if self.audit["checked"]:
            if passed(history[-1]["observation"]):
                return {"tool": "submit", "sql": self.audit["checked"][0]}
            return self.fallback
        action = self.fallback or self.baseline.act(obs)
        public = any(e["tool"] == "check" and passed(e["observation"]) for e in history)
        if remaining == 1 and action["tool"] == "check" and not public:
            fixed = schedule_pool(obs)
            self.audit["fixed_slot_pool"] = fixed
            if fixed and self.arm == "replace" and identity(fixed[0]["sql"]) != identity(action["sql"]):
                choice = fixed[0]
                self.fallback = {"tool": "submit", "sql": obs["issue_sql"]}
                self.audit.update(pool=fixed, trigger=True, selected=choice, replaced=action["sql"],
                                  checked=[choice["sql"]])
                return {"tool": "check", "sql": choice["sql"]}
        if action["tool"] != "submit":
            return action
        self.audit["eligible"] = public
        c = context(obs)
        found = discover(obs, action["sql"])
        schema = c["schema"]
        bound = list(islice(compiled(c, parse(action["sql"]), plans(obs, found, schema)), 8))
        if (schema is None and not bound and remaining >= 2
                and any(record[-1] for record in found)):
            self.fallback = action
            self.audit["schema_requested"] = True
            return {"tool": "schema"}
        self.audit.update(pool=[*bound, *self.audit["fixed_slot_pool"]], trigger=bool(found or self.audit["fixed_slot_pool"]),
                          schema_only_sql=[p["sql"] for p in bound] if self.audit["schema_requested"] else [])
        previous = {identity(e["sql"]) for e in history if e["tool"] == "check"}
        self.audit["already_checked"] = [p["sql"] for p in self.audit["pool"] if identity(p["sql"]) in previous]
        choice = next((p for p in bound if p["basis"]["eligible"] and identity(p["sql"]) not in previous), None)
        if choice and remaining:
            self.fallback = action
            self.audit.update(checked=[choice["sql"]], selected=choice)
            return {"tool": "check", "sql": choice["sql"]}
        return action


def create_agent(checkpoint):
    path = Path(checkpoint).resolve()
    arm = json.loads(path.read_text(encoding="utf-8"))["arm"]
    if arm not in ARMS:
        raise ValueError("Unknown step34 arm")
    return Step34Agent(step25(path.parent.parent / "configs/step25-exists.json"), arm)
