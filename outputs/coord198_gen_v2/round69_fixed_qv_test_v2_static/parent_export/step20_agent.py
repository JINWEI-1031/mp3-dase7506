"""Late rescue of steps that have no passing public submission."""
import json
from pathlib import Path

from sqlglot import exp

from repair_rules import parse
from step10_bindings import field_words, tokens
from step17_agent import create_agent as step17
from step19_agent import public_submission
from step19_grounding import failed_pool
from step4_agent import identity, passed
from step4_edits import Schema
from step20_paths import path_pool

ARMS = {
    "joint": (0, 1),
    "paths": (2, 1),
    "depth3": (3, 1),
    "paths3": (2, 3),
    "depth3-check3": (3, 3),
    "grounded": (2, 3),
}


def projection_used_as_filter(item, question):
    """A projected label should not absorb an independent filter role."""
    tree = parse(item["sql"])
    if not isinstance(tree, exp.Select) or tree.args.get("where") is None:
        return False
    columns = {p.unalias().name.lower() for p in tree.selects if isinstance(p.unalias(), exp.Column)}
    for atom in tree.args["where"].find_all(exp.EQ):
        if (isinstance(atom.this, exp.Column) and atom.this.name.lower() in columns
                and isinstance(atom.expression, exp.Literal) and atom.expression.is_string):
            words = tokens(question)
            value = tokens(atom.expression.this)
            for i in range(len(words)):
                if value and words[i:i + len(value)] == value:
                    nearby = set(words[max(0, i - 2):i] + words[i + len(value):i + len(value) + 2])
                    if not set(field_words(atom.this.name)) & nearby:
                        return True
    return False


class Step20Agent:
    def __init__(self, baseline, arm):
        self.baseline = baseline
        self.arm = arm
        self.depth, self.max_checks = ARMS[arm]
        self.fallback = None
        self.audit = {"pool": [], "checked": [], "trigger": False}

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
        if not self.audit.get("generated"):
            tables = next((e["observation"] for e in history if e["tool"] == "schema"), None)
            if tables is None and remaining >= 2:
                self.audit.update(trigger=True, schema_requested=True)
                return {"tool": "schema"}
            if tables is None:
                self.audit["reason"] = "schema_unavailable"
                return action
            schema = Schema(tables)
            previous = failed_pool(obs, action["sql"], schema, True)
            added = path_pool(obs, action["sql"], schema, self.depth) if self.depth else []
            candidates = {}
            # Keep the strongest prior joint candidate ahead of speculative path alternatives.
            for item in [*previous[:1], *added, *previous[1:]]:
                candidates.setdefault(identity(item["sql"]), item)
            ordered = list(candidates.values())
            if self.arm == "grounded":
                question = obs["task"]["question"]
                ordered.sort(key=lambda item: projection_used_as_filter(item, question))
            self.audit.update(generated=True, pool=ordered, new_pool=added)
        pool = self.audit["pool"]
        if not pool:
            self.audit["reason"] = "no_candidate"
        elif remaining and len(checked) < min(len(pool), self.max_checks):
            sql = pool[len(checked)]["sql"]
            checked.append(sql)
            self.audit["trigger"] = True
            return {"tool": "check", "sql": sql}
        else:
            self.audit["reason"] = "budget" if not remaining else "checks_completed"
        return action


def create_agent(checkpoint):
    path = Path(checkpoint).resolve()
    settings = json.loads(path.read_text(encoding="utf-8"))
    return Step20Agent(step17(path.with_name("step17-predicate.json")), settings["arm"])
