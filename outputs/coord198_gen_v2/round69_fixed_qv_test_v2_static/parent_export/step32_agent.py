"""Append one check to the unmodified step25 baseline when a plan is supported."""
import json
from pathlib import Path
import re

from sqlglot import exp

from repair_rules import parse
from step4_agent import identity, passed
from step19_agent import public_submission
from step25_agent import create_agent as step25
from step32_plan import (COMPARISON, compiled, context, cpu_plans, decompose,
                         pending_bindings)

POOL_LIMIT = 5
ARMS = ("cpu-observed", "cpu-schema", "model-schema")


def model_evidence(c, parent, candidate):
    old, new = decompose(parent), decompose(parse(candidate["sql"]))
    if not old or not new or old.limit or not new.order or not re.search(r"\balphabetic", c["question"], re.I):
        return False
    return (all(a == b for key, a in vars(old).items() if key != "order" for b in [getattr(new, key)])
            and re.sub(r"\s+ASC$", "", new.order, flags=re.I) == old.output)


class Step32Agent:
    def __init__(self, baseline, arm):
        self.baseline = baseline
        self.arm = arm
        self.fallback = None
        self.audit = {"pool": [], "observed_pool": [], "checked": [], "eligible": False,
                      "trigger": False, "already_checked": [], "schema_requested": False}

    def act(self, obs):
        history, remaining = obs["history"], obs["remaining"]
        if self.audit["checked"]:
            if passed(history[-1]["observation"]):
                return {"tool": "submit", "sql": self.audit["checked"][0]}
            return self.fallback
        action = self.fallback or self.baseline.act(obs)
        if action["tool"] != "submit":
            return action
        sql = action["sql"]
        parent = parse(sql)
        public = public_submission(history, sql)
        ever_public = any(e["tool"] == "check" and passed(e["observation"]) for e in history)
        self.audit["eligible"] = ever_public
        c = context(obs)
        checked = {identity(e["sql"]) for e in history if e["tool"] in ("check", "execute")}
        if self.fallback is None:
            self.audit["pending_bindings"] = pending_bindings(c, parent, ever_public)
            observed = list(compiled(c, parent, cpu_plans(c, parent, ever_public)))[:POOL_LIMIT]
            self.audit["observed_pool"] = observed
            if self.arm != "cpu-observed" and self.audit["pending_bindings"] and remaining >= 2:
                self.fallback = action
                self.audit["schema_requested"] = True
                return {"tool": "schema"}
        pending = self.audit["pending_bindings"]
        cpu = list(compiled(c, parent, cpu_plans(c, parent, ever_public)))[:POOL_LIMIT] \
            if self.audit["schema_requested"] else self.audit["observed_pool"]
        original = {identity(p["sql"]) for p in self.audit["observed_pool"]}
        schema_only = [p["sql"] for p in cpu if identity(p["sql"]) not in original]
        pool = [p | {"source": "cpu", "selection_evidence": True} for p in cpu]
        can_plan = not ever_public or bool(parent and COMPARISON.search(c["question"])
                                           and any(parent.find_all(exp.Subquery)))
        self.audit.update(schema_only_sql=schema_only, schema_key_evidence=bool(schema_only),
                          trigger=bool(pending or cpu or (self.arm == "model-schema" and can_plan)),
                          model_trigger=bool(self.arm == "model-schema" and can_plan))
        # Diagnostic generation can occur with no check slots, to measure lost coverage.
        if self.arm == "model-schema" and can_plan:
            from step32_model import propose

            model = propose(obs, c, sql)
            self.audit["model"] = {key: model[key] for key in
                                   ("valid_plan", "sql", "input_tokens", "new_tokens", "wall_seconds", "cpu_seconds")}
            if model["sql"] and identity(model["sql"]) == identity(sql):
                self.audit["model"]["duplicate_baseline"] = True
            elif model["sql"]:
                proposals = list(compiled(c, parent, [{"plan": model["plan"], "basis": "frozen model proposal"}]))
                for p in proposals:
                    p.update(source="model", selection_evidence=not ever_public or model_evidence(c, parent, p))
                    pool.append(p)
        self.audit.update(public_submission=public, pool=pool, checked=[],
                          already_checked=[p["sql"] for p in pool if identity(p["sql"]) in checked])
        # A known passing submission is retained unless the new plan has independent evidence.
        choice = next((p for p in pool if p["selection_evidence"] and identity(p["sql"]) not in checked), None)
        if choice and remaining:
            self.fallback = action
            self.audit.update(checked=[choice["sql"]], selected=choice)
            return {"tool": "check", "sql": choice["sql"]}
        return action


def create_agent(checkpoint):
    path = Path(checkpoint).resolve()
    arm = json.loads(path.read_text(encoding="utf-8"))["arm"]
    if arm not in ARMS:
        raise ValueError("Unknown step32 arm")
    return Step32Agent(step25(path.with_name("step25-exists.json")), arm)


