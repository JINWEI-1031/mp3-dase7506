"""Round 40 public-only witness suffix."""
import json
from pathlib import Path

from step4_agent import identity, passed
from step25_agent import create_agent as step25
from step32_plan import context
from step37.agent import Step37Agent
from step40.witness import analyze


class Suffix:
    def __init__(self, fallback, settings, use_new=True, allow_schema=True):
        self.fallback = fallback
        self.settings = settings
        self.use_new = use_new
        self.allow_schema = allow_schema
        self.audit = {"pool": [], "binding": {}, "request": None, "checked": None}

    def act(self, obs):
        if self.audit["checked"]:
            if passed(obs["history"][-1]["observation"]):
                return {"tool": "submit", "sql": self.audit["checked"]}
            return self.fallback
        c = context(obs)
        plan, binding = analyze(c) if self.use_new else (None, {"stage": "control"})
        old = Step37Agent(None, "membership", replace_slot=False)
        old.fallback = self.fallback
        old_action = old.act(obs)
        request = {}
        if binding.get("request"):
            request["witness_binding"] = binding["request"]
        if old_action["tool"] == "schema":
            request["inherited_binding"] = {"unresolved": old.audit["binding_requests"],
                                            "output_fields": c["columns"], "question": c["question"]}
        if (self.allow_schema and c["schema"] is None and not self.audit["request"]
                and obs["remaining"] >= self.settings["schema_min_remaining"] and request):
            self.audit["request"] = request
            return {"tool": "schema"}
        if old_action["tool"] == "schema":
            old.act({**obs, "remaining": 1})
        pool = {}
        for option in old.audit["pool"][:self.settings["inherited_cap"]]:
            pool.setdefault(identity(option["sql"]), option)
        if plan and self.settings["new_cap"]:
            pool.setdefault(identity(plan["sql"]), plan)
        self.audit.update(pool=list(pool.values()), binding=binding)
        prior = {identity(sql) for sql, _ in c["checks"]}
        choice = next((p for key, p in pool.items() if p["basis"]["eligible"] and key not in prior), None)
        if choice and obs["remaining"] and self.settings["suffix_checks"]:
            self.audit["checked"] = choice["sql"]
            return {"tool": "check", "sql": choice["sql"]}
        return self.fallback


class Agent:
    def __init__(self, baseline, settings):
        self.baseline = baseline
        self.settings = settings
        self.suffix = None

    def act(self, obs):
        if self.suffix is None:
            action = self.baseline.act(obs)
            if action["tool"] != "submit":
                return action
            self.suffix = Suffix(action, self.settings)
        return self.suffix.act(obs)


def create_agent(checkpoint):
    path = Path(checkpoint).resolve()
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    return Agent(step25(path.parent / config["baseline_config"]), config["main"])

