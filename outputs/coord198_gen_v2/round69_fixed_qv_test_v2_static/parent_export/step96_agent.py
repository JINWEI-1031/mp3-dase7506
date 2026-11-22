"""Reorder named output columns while preserving named ORDER BY expressions."""
from sqlglot import exp

from repair_rules import parse, sql_text
from step4_agent import identity, passed, usable
from step93_agent import create_agent as create_baseline


def ordered_columns(observation, sql):
    tree = parse(sql)
    heads = [head.casefold() for head in observation["example"]["columns"]]
    if (not isinstance(tree, exp.Select)
            or any(tree.args.get(key) for key in ("group", "having", "with"))
            or any(not isinstance(item.unalias(), exp.Column) or item.unalias().is_star
                   for item in tree.expressions)):
        return None
    order = tree.args.get("order")
    if order is None or any(not isinstance(item.this, exp.Column) for item in order.expressions):
        return None
    fields = {item.alias_or_name.casefold(): item for item in tree.expressions}
    if (len(fields) != len(tree.expressions) or len(set(heads)) != len(heads)
            or fields.keys() != set(heads) or list(fields) == heads):
        return None
    query = tree.select(*(fields[head].copy() for head in heads), append=False)
    return {"sql": sql_text(query), "kind": "ordered_public_projection", "columns": heads}


class ProjectionSuffix:
    def __init__(self, fallback):
        self.fallback = fallback
        self.candidate = None
        self.evidence = None

    def act(self, observation):
        if self.candidate:
            return {"tool": "submit", "sql": self.candidate} if passed(
                observation["history"][-1]["observation"]) else self.fallback
        if not observation["remaining"]:
            return self.fallback
        candidate = ordered_columns(observation, self.fallback["sql"])
        if candidate is None or not usable(candidate["sql"]):
            return self.fallback
        self.evidence = candidate
        previous = next((e["observation"] for e in observation["history"] if e["tool"] == "check"
                         and identity(e["sql"]) == identity(candidate["sql"])), None)
        if previous is not None:
            return {"tool": "submit", "sql": candidate["sql"]} if passed(previous) else self.fallback
        self.candidate = candidate["sql"]
        return {"tool": "check", "sql": self.candidate}


class Agent:
    def __init__(self, baseline):
        self.baseline = baseline
        self.suffix = None

    def act(self, observation):
        if self.suffix is None:
            action = self.baseline.act(observation)
            if action["tool"] != "submit":
                return action
            self.suffix = ProjectionSuffix(action)
        return self.suffix.act(observation)


def create_agent(checkpoint=None):
    return Agent(create_baseline(checkpoint))
