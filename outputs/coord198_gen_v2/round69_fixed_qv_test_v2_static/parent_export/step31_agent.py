"""Use one fixed post-pass check for a grounded scope or role repair."""
import json
from pathlib import Path

from step4_agent import identity, passed
from step19_agent import public_submission
from step25_agent import create_agent as step25
from step28_roles import candidate_pool as role_candidates
from step29_agent import filtered_role_shape
from step30_multiplicity import candidate_pool as scope_candidates, scope_plan
from step31_population import candidate_pool as population_candidates, population_plan


class Step31Agent:
    def __init__(self, baseline):
        self.baseline = baseline
        self.fallback = None
        self.audit = {"pool": [], "checked": [], "trigger": False, "eligible": False}

    def act(self, obs):
        history, remaining = obs["history"], obs["remaining"]
        if self.audit["checked"]:
            if passed(history[-1]["observation"]):
                return {"tool": "submit", "sql": self.audit["checked"][0]}
            return self.fallback
        action = self.fallback or self.baseline.act(obs)
        if action["tool"] != "submit" or not public_submission(history, action["sql"]):
            return action
        self.audit["eligible"] = True
        sql = action["sql"]
        roles, suppressed = filtered_role_shape(obs, sql)
        scope = scope_plan(obs, sql, "gated")
        population = population_plan(obs, sql)
        if suppressed:
            self.audit["conventional_keys"] = suppressed
        if not (population or scope or roles):
            self.audit["reason"] = "only_conventional_key" if suppressed else "no_observed_conflict"
            return action
        if population:
            self.audit["population_signal"] = population.evidence
        self.audit["scope_signal"] = scope[0].kind if scope else None
        self.audit["role_cues"] = roles[1] if roles else ()
        tables = next((e["observation"] for e in history if e["tool"] == "schema"), None)
        requires_schema = population.needs_schema if population else bool(scope or roles)
        if tables is None and requires_schema:
            if remaining < 2:
                self.audit["reason"] = "insufficient_budget_for_schema_and_check"
                return action
            self.fallback = action
            self.audit["schema_requested"] = True
            return {"tool": "schema"}
        pool = population_candidates(obs, sql, population, tables) if population else []
        if scope and tables is not None:
            pool.extend(scope_candidates(obs, sql, tables, scope))
        seen = {identity(c["sql"]) for c in pool}
        if roles and tables is not None:
            pool.extend(c for c in role_candidates(obs, sql, tables, roles) if identity(c["sql"]) not in seen)
        previous = {identity(e["sql"]) for e in history if e["tool"] == "check"}
        self.audit.update(pool=pool, already_checked=[c["sql"] for c in pool if identity(c["sql"]) in previous])
        choice = next((c for c in pool if identity(c["sql"]) not in previous), None)
        if not choice or not remaining:
            self.audit["reason"] = "budget_exhausted" if choice else "no_grounded_candidate"
            return action
        self.fallback = action
        self.audit.update(checked=[choice["sql"]], trigger=True, selected=choice)
        return {"tool": "check", "sql": choice["sql"]}


def create_agent(checkpoint):
    path = Path(checkpoint).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if config != {"policy": "quantified-population"}:
        raise ValueError("Unknown step31 policy")
    return Step31Agent(step25(path.with_name("step25-exists.json")))
