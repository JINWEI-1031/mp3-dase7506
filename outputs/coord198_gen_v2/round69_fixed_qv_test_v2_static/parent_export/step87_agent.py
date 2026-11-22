"""Recover the explicitly requested entity ID for a most-recent lookup."""
import re

from sqlglot import exp

from repair_rules import parse, sql_text
from step4_agent import identity, passed, usable
from step4_edits import Schema
from step5x_edits import physical_table
from step86_agent import create_agent as create_previous


LATEST = re.compile(r"\b(?:id|identifier) of (?:the )?([a-z]+) who most recently\b", re.I)


def lookup_shape(observation, sql):
    cue = LATEST.search(observation["task"]["question"])
    heads = observation["example"]["columns"]
    tree = parse(sql) if cue else None
    if (not isinstance(tree, exp.Select) or len(heads) != 1 or len(tree.expressions) != 1
            or heads[0].casefold() != cue[1].casefold() + "_id"
            or not isinstance(tree.expressions[0].unalias(), exp.Column)
            or tree.expressions[0].unalias().name.casefold() == heads[0].casefold()
            or tree.args.get("with") or tree.args.get("group") or tree.args.get("having")
            or not tree.args.get("limit") or tree.args["limit"].expression != exp.Literal.number(1)
            or re.search(r"['\"]|\d", observation["task"]["question"])):
        return None
    return heads[0].casefold()


def compile_lookup(field, tables):
    schema = Schema(tables)
    choices = [(table, date) for table, columns in schema.columns.items() if field in columns
               for date, meta in columns.items() if re.search(r"date|time", meta["type"], re.I)]
    if len(choices) != 1:
        return None
    root, metric = choices[0]
    query = exp.select(schema.column(root, field, "recent")).from_(physical_table(schema, root, "recent"))
    query.order_by(exp.Ordered(this=schema.column(root, metric, "recent"), desc=True, nulls_first=False),
                   copy=False).limit(1, copy=False)
    return {"sql": sql_text(query), "kind": "requested_entity_latest", "entity_id": field,
            "table": root, "date": metric}


class LatestSuffix:
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
        field = lookup_shape(observation, self.fallback["sql"])
        if field is None:
            return self.fallback
        tables = next((e["observation"] for e in observation["history"] if e["tool"] == "schema"), None)
        if tables is None:
            return {"tool": "schema"} if observation["remaining"] >= 2 else self.fallback
        candidate = compile_lookup(field, tables)
        if candidate is None or not usable(candidate["sql"]):
            return self.fallback
        self.evidence = candidate
        if any(e["tool"] == "check" and identity(e["sql"]) == identity(candidate["sql"])
               for e in observation["history"]):
            return self.fallback
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
            self.suffix = LatestSuffix(action)
        return self.suffix.act(observation)


def create_agent(checkpoint=None):
    return Agent(create_previous(checkpoint))
