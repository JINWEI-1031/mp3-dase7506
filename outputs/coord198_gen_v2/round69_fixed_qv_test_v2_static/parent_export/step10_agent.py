"""Spend at most one extra check and retain the frozen baseline fallback."""
import json
from pathlib import Path

from mp3_sql.sqlite_tool import validate_sql
from step4_agent import identity, passed
from step4_edits import Schema
from step8_agent import create_agent as step8
from step10_bindings import candidates, public_role_conflict


class Step10Agent:
    def __init__(self, baseline, settings):
        self.baseline = baseline
        self.settings = settings
        self.fallback = None
        self.audit = {"pool": [], "stratum": "baseline in progress"}

    def act(self, obs):
        history = obs["history"]
        if self.fallback is not None:
            chosen = self.audit.get("selected")
            if chosen and passed(history[-1]["observation"]):
                return {"tool": "submit", "sql": chosen["sql"]}
            return self.fallback
        action = self.baseline.act(obs)
        if action["tool"] != "submit":
            return action
        self.fallback = action
        public = any(e["tool"] == "check" and passed(e["observation"]) for e in history)
        self.audit["stratum"] = "public pass" if public else "no public pass"
        if public != self.settings["public"]:
            return action
        tables = next((e["observation"] for e in history if e["tool"] == "schema"), None)
        if tables is None:
            self.audit["reason"] = "no observed schema"
            return action
        schema = Schema(tables)
        pool, counts = candidates(obs, schema, self.settings["families"],
                                  self.settings["refined"], [action["sql"]])
        if public:
            pool = [c for c in pool if c["parent"] == action["sql"] and c["rule"] in {
                "predicate_column", "predicate_operator", "predicate_negation",
                "predicate_boolean", "predicate_boolean_scope"}]
            conflict = public_role_conflict(obs, schema, action["sql"])
            seen = {identity(e["sql"]) for e in history if "sql" in e}
            seen.add(identity(action["sql"]))
            if conflict and identity(conflict["sql"]) not in seen:
                validate_sql(conflict["sql"])
                counts["role_generated"] = 1
                pool.insert(0, conflict)
        self.audit.update(pool=pool, counts=counts, remaining=obs["remaining"])
        if pool and obs["remaining"]:
            self.audit["selected"] = pool[0]
            return {"tool": "check", "sql": pool[0]["sql"]}
        return action


def create_agent(checkpoint):
    path = Path(checkpoint).resolve()
    settings = json.loads(path.read_text(encoding="utf-8"))
    if (set(settings) != {"families", "refined", "public"}
            or not settings["families"] or not set(settings["families"]) <= {
                "predicate", "role", "projection", "coupled"}
            or any(type(settings[k]) is not bool for k in ("refined", "public"))):
        raise ValueError("Invalid step 10 configuration")
    return Step10Agent(step8(path.with_name("step8-group-1.json")), settings)
