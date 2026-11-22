"""Run the cross-database graph source after the step25 baseline."""
from itertools import islice
import json
from pathlib import Path

from repair_rules import parse
from step4_agent import identity, passed
from step25_agent import create_agent as step25
from step32_plan import compiled, context
from step33.relations import plan_options, request_binding

ARMS = ("observed", "schema")
POOL_LIMIT = 6


class Step33Agent:
    def __init__(self, baseline, arm):
        self.baseline = baseline
        self.arm = arm
        self.fallback = None
        self.audit = {"pool": [], "checked": [], "already_checked": [], "eligible": False,
                      "trigger": False, "schema_requested": False, "schema_only_sql": []}

    def act(self, obs):
        history = obs["history"]
        if self.audit["checked"]:
            if passed(history[-1]["observation"]):
                return {"tool": "submit", "sql": self.audit["checked"][0]}
            return self.fallback
        action = self.fallback or self.baseline.act(obs)
        if action["tool"] != "submit":
            return action
        c = context(obs)
        parent = parse(action["sql"])
        public = any(e["tool"] == "check" and passed(e["observation"]) for e in history)
        self.audit["eligible"] = public
        schema = c["schema"]
        if schema is None:
            pending = request_binding(c, parent, public)
            self.audit.update(trigger=bool(pending), pending_binding=pending)
            if pending and self.arm == "schema" and obs["remaining"] >= 2:
                self.fallback = action
                self.audit["schema_requested"] = True
                return {"tool": "schema"}
            return action
        options = plan_options(c, parent, public, schema)
        candidates = compiled(c, parent, options)
        # Put independently supported plans first, then keep a small shadow
        # pool for offline candidate coverage when evidence is weaker.
        proposed = list(islice(candidates, 18))
        proposed.sort(key=lambda p: not bool(p["basis"]["advantage"]))
        pool = proposed[:POOL_LIMIT]
        previous = {identity(e["sql"]) for e in history if e["tool"] == "check"}
        self.audit.update(pool=pool, trigger=bool(pool) or self.audit["schema_requested"],
                          already_checked=[p["sql"] for p in pool if identity(p["sql"]) in previous],
                          schema_only_sql=[p["sql"] for p in pool] if self.audit["schema_requested"] else [])
        choice = next((p for p in pool if p["basis"]["advantage"] and identity(p["sql"]) not in previous), None)
        if choice and obs["remaining"]:
            self.fallback = action
            self.audit.update(checked=[choice["sql"]], selected=choice)
            return {"tool": "check", "sql": choice["sql"]}
        return action


def create_agent(checkpoint):
    path = Path(checkpoint).resolve()
    arm = json.loads(path.read_text(encoding="utf-8"))["arm"]
    if arm not in ARMS:
        raise ValueError(arm)
    return Step33Agent(step25(path.parent.parent / "configs/step25-exists.json"), arm)
