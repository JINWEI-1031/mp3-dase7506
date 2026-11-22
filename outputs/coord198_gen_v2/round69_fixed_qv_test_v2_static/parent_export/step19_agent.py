"""Independent wrappers around the fixed step17 policy."""
import json
from pathlib import Path

from step4_agent import identity, passed
from step4_edits import Schema
from step17_agent import create_agent as step17
from step18_ordering import pool as ordering_pool
from step19_grounding import failed_pool, observed_schema, parse, public_pool, requests


ARMS = ("observed", "schema", "mapping", "existing", "joint", "row", "event", "combined")


def public_submission(history, sql):
    return any(e["tool"] == "check" and passed(e["observation"])
               and identity(e["sql"]) == identity(sql) for e in history)


class Step19Agent:
    def __init__(self, baseline, arm):
        self.baseline = baseline
        self.arm = arm
        self.fallback = None
        self.audit = {"pool": [], "observed_candidates": 0, "trigger": False}

    def act(self, obs):
        history = obs["history"]
        last = history[-1] if history else None
        if self.audit.get("selected"):
            if passed(last["observation"]):
                return {"tool": "submit", "sql": self.audit["selected"]["sql"]}
            return self.fallback
        if self.fallback is None:
            action = self.baseline.act(obs)
            if action["tool"] != "submit":
                return action
            self.fallback = action
        action = self.fallback
        sql = action["sql"]
        schema_event = next((e for e in history if e["tool"] == "schema"), None)
        schema = Schema(schema_event["observation"]) if schema_event else None
        remaining = obs["remaining"]
        public = public_submission(history, sql)
        self.audit.update(public=public)
        if public and self.arm in {"observed", "schema", "mapping", "combined"}:
            tree = parse(sql)
            evidence = requests(obs, tree)
            self.audit["requests"] = sorted(evidence)
            if not evidence:
                self.audit["reason"] = "no explicit unresolved constraint"
                return action
            if not self.audit.get("schema_requested"):
                options = public_pool(obs, sql, tree, schema or observed_schema(obs, tree), evidence)
                self.audit["observed_candidates"] = len(options)
            else:
                options = public_pool(obs, sql, tree, schema, evidence, self.arm == "mapping")
            if (not options and schema is None and self.arm != "observed" and remaining >= 2):
                self.audit.update(trigger=True, schema_requested=True)
                return {"tool": "schema"}
            if not options and self.arm == "mapping" and schema is not None:
                options = public_pool(obs, sql, tree, schema, evidence, "relation" in evidence)
        elif public and self.arm in {"row", "event"}:
            options = [c for c in ordering_pool(obs, sql, schema, True)
                       if c["family"] == "event-count" or "frequency" in str(c.get("conflict", ""))]
            if self.arm == "event":
                options.sort(key=lambda c: c["family"] != "event-count")
        elif not public and self.arm in {"existing", "joint", "combined"}:
            # The baseline exhausts its own policy before either independent arm can act.
            joint = self.arm in {"joint", "combined"}
            if joint and schema is None and remaining >= 2:
                self.audit.update(trigger=True, schema_requested=True)
                return {"tool": "schema"}
            options = failed_pool(obs, sql, schema, joint) if not joint or schema else []
        else:
            self.audit["reason"] = "other branch"
            return action
        self.audit["pool"] = options
        if not options or not remaining:
            self.audit["reason"] = "no bound candidate" if not options else "budget exhausted"
            return action
        self.audit.update(trigger=True, selected=options[0])
        return {"tool": "check", "sql": options[0]["sql"]}


def create_agent(checkpoint):
    path = Path(checkpoint).resolve()
    arm = json.loads(path.read_text(encoding="utf-8"))["arm"]
    if arm not in ARMS:
        raise ValueError("Unknown step19 arm")
    return Step19Agent(step17(path.with_name("step17-predicate.json")), arm)
