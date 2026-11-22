"""Use one grouped probe while preserving the step 8 fallback."""
import json
from pathlib import Path

from mp3_sql.sqlite_tool import validate_sql
from step4_agent import identity, passed
from step4_edits import Schema
from step8_agent import create_agent as step8
from step8_groups import compile_pool
from step9_roles import candidates, fact_priority, preaggregate


class Step9Agent:
    def __init__(self, baseline, settings):
        self.baseline = baseline
        self.settings = settings
        self.probe = None
        self.delayed = None
        self.audit = {"pool": [], "reason": "baseline"}

    def act(self, obs):
        if self.probe is not None:
            feedback = obs["history"][-1]["observation"]
            if passed(feedback):
                self.audit["reason"] = "probe public pass"
                return {"tool": "submit", "sql": self.probe}
            self.probe = None
            self.audit["reason"] = "probe public fail"
            action, self.delayed = self.delayed, None
            if action and (action["tool"] == "submit" or obs["remaining"]):
                return action
        action = self.baseline.act(obs)
        if self.audit.get("selected") or not obs["remaining"]:
            return action
        history = obs["history"]
        if any(e["tool"] == "check" and passed(e["observation"]) for e in history):
            return action
        tables = next((e["observation"] for e in history if e["tool"] == "schema"), None)
        if tables is None:
            return action
        schema = Schema(tables)
        seen = {identity(e["sql"]) for e in history if "sql" in e}
        if "sql" in action:
            seen.add(identity(action["sql"]))
        pool = []
        stage7 = self.baseline.base
        late_expansion = action["tool"] == "check" and stage7.start is not None and self.baseline.fallback is None
        if late_expansion and obs["remaining"] == 1 and self.settings["schedule"]:
            fixed = compile_pool(obs, schema, seen.copy(), ("fact", "detail"))
            question = obs["task"]["question"].lower()
            pool = [c for c in sorted(fixed, key=lambda c: -fact_priority(c, question))
                    if fact_priority(c, question)]
            if pool:
                self.audit["fixed_pool"] = fixed
                self.audit["displaced"] = action
        if self.settings["roles"] and not pool and (late_expansion or action["tool"] == "submit"):
            pool = candidates(obs, schema, seen)
            if late_expansion:
                pool = [c for c in pool if c["rule"] == "typed_role_filter"]
        if self.settings.get("preaggregate") and self.baseline.fallback is not None and action["tool"] == "check":
            split = preaggregate(action["sql"], schema, obs["example"]["columns"])
            if split and identity(split["sql"]) not in seen:
                pool = [split]
        self.audit["pool"] = pool or self.audit["pool"]
        for candidate in pool:
            sql = candidate["sql"]
            try:
                validate_sql(sql)
            except ValueError:
                continue
            self.audit["selected"] = candidate
            self.audit["reason"] = "probe"
            self.probe = sql
            self.delayed = action
            return {"tool": "check", "sql": sql}
        return action


def create_agent(checkpoint):
    path = Path(checkpoint).resolve()
    settings = json.loads(path.read_text(encoding="utf-8"))
    if (not {"schedule", "roles"} <= settings.keys() <= {"schedule", "roles", "preaggregate"}
            or not all(type(x) is bool for x in settings.values())):
        raise ValueError("Invalid step 9 settings")
    return Step9Agent(step8(path.with_name("step8-group-1.json")), settings)

