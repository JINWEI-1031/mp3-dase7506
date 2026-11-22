"""Paired public-observation controller for round 38."""
from step4_agent import identity, passed
from repair_rules import parse
from step32_plan import context
from step37.agent import Step37Agent
from step38.event_grain import analyze


class Suffix:
    def __init__(self, fallback, use_new, allow_schema):
        self.fallback = fallback
        self.use_new = use_new
        self.allow_schema = allow_schema
        self.audit = {"pool": [], "new_pool": [], "checked": [], "binding_request": None}

    def act(self, obs):
        history, remaining = obs["history"], obs["remaining"]
        if self.audit["checked"]:
            if passed(history[-1]["observation"]):
                return {"tool": "submit", "sql": self.audit["checked"][0]}
            return self.fallback
        c = context(obs)
        c["baseline"] = self.fallback["sql"]
        plans, binding = analyze(c, parse(c["baseline"])) if self.use_new else ([], {})
        old = Step37Agent(None, "membership", replace_slot=False)
        old.fallback = self.fallback
        old_action = old.act(obs)
        if c["schema"] is None and not self.audit["binding_request"]:
            requests = {}
            if old_action["tool"] == "schema":
                requests["inherited"] = old.audit["binding_requests"]
            if binding.get("request"):
                requests["event_grain"] = binding["request"]
            if self.allow_schema and remaining >= 2 and requests:
                self.audit["binding_request"] = requests
                return {"tool": "schema"}
        if old_action["tool"] == "schema":
            # Keep the inherited no-schema pool for the observation control.
            old.act({**obs, "remaining": min(remaining, 1)})
        known = {identity(p["sql"]) for p in old.audit["pool"]}
        new = [p for p in plans if identity(p["sql"]) not in known]
        pool = [*old.audit["pool"], *new]
        previous = {identity(e["sql"]) for e in history if e["tool"] == "check"}
        self.audit.update(pool=pool, new_pool=new, binding=binding,
                          old_binding=old.audit.get("binding_requests", []))
        choice = next((p for p in pool if p["basis"]["eligible"] and identity(p["sql"]) not in previous), None)
        if choice and remaining:
            self.audit.update(checked=[choice["sql"]], selected=choice)
            return {"tool": "check", "sql": choice["sql"]}
        return self.fallback


class Agent:
    def __init__(self, baseline):
        self.baseline = baseline
        self.suffix = None

    def act(self, obs):
        if self.suffix is None:
            action = self.baseline.act(obs)
            if action["tool"] != "submit":
                return action
            self.suffix = Suffix(action, use_new=True, allow_schema=True)
        return self.suffix.act(obs)


def create_agent(checkpoint):
    from step25_agent import create_agent as step25
    return Agent(step25(checkpoint))

