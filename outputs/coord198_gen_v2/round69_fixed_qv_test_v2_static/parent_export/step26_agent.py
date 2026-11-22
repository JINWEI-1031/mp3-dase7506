"""One fixed candidate check after step25 or in its inherited rescue slot."""
import json
from pathlib import Path
import re

from step4_agent import identity, passed
from step25_agent import create_agent as step25
from step26_relations import candidate_pool

ARMS = ("observed", "schema", "replace")
SCHEMA_SIGNAL = re.compile(r"\b(?:both|without|not|ordered|from|each|who|whose|which)\b", re.I)


class Step26Agent:
    def __init__(self, baseline, arm):
        self.baseline = baseline
        self.arm = arm
        self.deferred = None
        self.fallback = None
        self.audit = {"pool": [], "checked": [], "trigger": False, "eligible": False}

    def act(self, obs):
        history, remaining = obs["history"], obs["remaining"]
        if self.audit["checked"]:
            if passed(history[-1]["observation"]):
                return {"tool": "submit", "sql": self.audit["checked"][0]}
            return self.fallback
        action = self.deferred or self.baseline.act(obs)
        if any(e["tool"] == "check" and passed(e["observation"]) for e in history):
            self.audit["stratum"] = "public"
            return action
        tables = any(e["tool"] == "schema" for e in history)
        precision = self.baseline.baseline
        inherited = precision.baseline
        replace = (self.arm == "replace" and remaining == 1 and tables
                   and action["tool"] == "check" and not self.baseline.audit["trigger"]
                   and not precision.audit["eligible"] and inherited.schedule is not None
                   and inherited.audit["checked"][-1:] == [action["sql"]])
        if not (replace or action["tool"] == "submit"):
            return action
        fallback = inherited.fallback if replace else action
        self.audit.update(eligible=True, stratum="no_public")
        pool = candidate_pool(obs, fallback["sql"])
        previous = {identity(e["sql"]) for e in history if e["tool"] == "check"}
        known = [c["sql"] for c in pool if identity(c["sql"]) in previous]
        reserved = previous | ({identity(action["sql"])} if replace else set())
        available = [c["sql"] for c in pool if identity(c["sql"]) not in reserved]
        self.audit.update(pool=pool, already_checked=known)
        if (self.arm in ("schema", "replace") and not tables and not available and remaining >= 2
                and SCHEMA_SIGNAL.search(obs["task"]["question"])):
            self.deferred = action
            self.audit["schema_requested"] = True
            return {"tool": "schema"}
        if not available or not remaining:
            return action
        self.fallback = fallback
        self.audit.update(checked=available[:1], trigger=True)
        if replace:
            self.audit["replaced"] = action["sql"]
        return {"tool": "check", "sql": available[0]}


def create_agent(checkpoint):
    path = Path(checkpoint).resolve()
    arm = json.loads(path.read_text(encoding="utf-8"))["arm"]
    if arm not in ARMS:
        raise ValueError(arm)
    return Step26Agent(step25(path.with_name("step25-exists.json")), arm)
