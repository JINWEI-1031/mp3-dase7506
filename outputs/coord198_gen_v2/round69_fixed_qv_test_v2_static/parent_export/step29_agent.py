"""Use one fixed post-pass check for a grounded scope or role repair."""
import json
from pathlib import Path

from step4_agent import identity, passed
from step19_agent import observed_schema, public_submission
from step25_agent import create_agent as step25
from step28_roles import link_columns, shape as role_shape, words as role_words
from step7_semantics import top_tables
from step28_roles import candidate_pool as role_candidates
from step29_scope import candidate_pool as scope_candidates, scope_signal


def conventional_direction(source_key, target, target_key):
    """Recognize a foreign key that already names the target entity and key."""
    return (role_words(target_key) in ({"id"}, {"key"}, {"code"})
            and role_words(source_key) == role_words(target) | role_words(target_key))


def filtered_role_shape(obs, sql):
    result = role_shape(obs, sql)
    if not result:
        return None, ()
    tree, cues = result
    if "asymmetric_key" not in cues:
        return result, ()
    observed = observed_schema(obs, tree)
    edges = link_columns(tree, top_tables(tree, observed), observed)
    if not edges:
        return result, ()
    asymmetric = [(a, k, b, ref) for a, k, b, ref in edges
                  if k != ref and "id" in role_words(k) | role_words(ref)]
    ordinary = tuple(e for e in asymmetric if conventional_direction(e[1], e[2], e[3])
                     or conventional_direction(e[3], e[0], e[1]))
    if len(ordinary) == len(asymmetric) and ordinary:
        remaining = tuple(cue for cue in cues if cue != "asymmetric_key")
        return ((tree, remaining) if remaining else None), ordinary
    return result, ()


class Step29Agent:
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
        scope = scope_signal(obs, sql)
        if suppressed:
            self.audit["conventional_keys"] = suppressed
        if not (scope or roles):
            self.audit["reason"] = "only_conventional_key" if suppressed else "no_observed_conflict"
            return action
        self.audit["scope_signal"] = scope.kind if scope else None
        self.audit["role_cues"] = roles[1] if roles else ()
        tables = next((e["observation"] for e in history if e["tool"] == "schema"), None)
        if tables is None:
            if remaining < 2:
                self.audit["reason"] = "insufficient_budget_for_schema_and_check"
                return action
            self.fallback = action
            self.audit["schema_requested"] = True
            return {"tool": "schema"}
        pool = scope_candidates(obs, sql, tables, scope) if scope else []
        seen = {identity(c["sql"]) for c in pool}
        if roles:
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
    if json.loads(path.read_text(encoding="utf-8")) != {"policy": "public-witness-check"}:
        raise ValueError("Unknown step29 policy")
    return Step29Agent(step25(path.with_name("step25-exists.json")))
