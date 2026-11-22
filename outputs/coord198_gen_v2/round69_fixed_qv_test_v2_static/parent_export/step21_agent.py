"""Try scoped predicate recovery before the prior no-public rescue policy."""
import json
from pathlib import Path

from step4_agent import identity, passed
from step4_edits import Schema
from step17_agent import create_agent as step17
from step19_agent import public_submission
from step19_grounding import failed_pool
from step20_agent import projection_used_as_filter
from step20_paths import path_pool
from step21_carry import carry_pool

ARMS = {
    "grounded": ("none", 3),
    "scoped": ("scoped", 3),
    "paths": ("paths", 3),
    "source": ("source", 3),
    "merge1": ("merge", 1),
    "merge": ("merge", 3),
}


def prior_pool(obs, sql, schema):
    joint = failed_pool(obs, sql, schema, True)
    rebuilt = path_pool(obs, sql, schema, 2)
    seen = {}
    for candidate in [*joint[:1], *rebuilt, *joint[1:]]:
        seen.setdefault(identity(candidate["sql"]), candidate)
    question = obs["task"]["question"]
    return sorted(seen.values(), key=lambda item: projection_used_as_filter(item, question))


class Step21Agent:
    def __init__(self, baseline, arm):
        self.baseline = baseline
        self.mode, self.max_carry_checks = ARMS[arm]
        self.fallback = None
        self.schedule = None
        self.audit = {"pool": [], "checked": [], "carry_count": 0, "trigger": False}

    def act(self, obs):
        history, remaining = obs["history"], obs["remaining"]
        checked = self.audit["checked"]
        if checked and passed(history[-1]["observation"]):
            return {"tool": "submit", "sql": checked[-1]}
        if self.fallback is None:
            action = self.baseline.act(obs)
            if action["tool"] != "submit":
                return action
            self.fallback = action
        action = self.fallback
        if public_submission(history, action["sql"]):
            self.audit["reason"] = "baseline_public"
            return action
        if self.schedule is None:
            tables = next((e["observation"] for e in history if e["tool"] == "schema"), None)
            if tables is None and remaining >= 2:
                self.audit.update(trigger=True, schema_requested=True)
                return {"tool": "schema"}
            if tables is None:
                self.audit["reason"] = "schema_unavailable"
                return action
            schema = Schema(tables)
            prior = prior_pool(obs, action["sql"], schema)
            recovered = carry_pool(obs, action["sql"], schema, self.mode) if self.mode != "none" else []
            pool = {}
            for candidate in [*recovered, *prior]:
                pool.setdefault(identity(candidate["sql"]), candidate)
            self.audit.update(pool=list(pool.values()), carry_count=len(recovered))
            self.schedule = list(dict.fromkeys(
                [*(c["sql"] for c in recovered[:self.max_carry_checks]),
                 *(c["sql"] for c in prior)]))
        if remaining and len(checked) < min(len(self.schedule), 3):
            sql = self.schedule[len(checked)]
            checked.append(sql)
            self.audit["trigger"] = True
            return {"tool": "check", "sql": sql}
        self.audit["reason"] = ("no_candidate" if not self.audit["pool"] else
                                "budget" if not remaining else "checks_completed")
        return action


def create_agent(checkpoint):
    path = Path(checkpoint).resolve()
    arm = json.loads(path.read_text(encoding="utf-8"))["arm"]
    return Step21Agent(step17(path.with_name("step17-predicate.json")), arm)
