"""Repair a scalar SUM's table and measure while preserving its LIMIT placement."""
import re

from sqlglot import exp

from repair_rules import parse, sql_text
from step4_agent import identity, passed, usable
from step4_edits import Schema
from step5x_edits import physical_table
from step38.event_grain import terms
from step99_agent import create_agent as create_baseline


SUM_HEADER = re.compile(r"sum\(\s*(\w+)\s*\)", re.I)


def scalar_measure(observation, sql):
    heads = observation["example"]["columns"]
    heading = SUM_HEADER.fullmatch(heads[0]) if len(heads) == 1 else None
    tree = parse(sql) if heading else None
    if (not isinstance(tree, exp.Select) or len(tree.expressions) != 1
            or any(tree.args.get(key) for key in ("distinct", "group", "having", "offset", "with"))
            or len(list(tree.find_all(exp.Select))) != 1 or not tree.args.get("joins")
            or not tree.args.get("order") or not tree.args.get("limit")):
        return None
    aggregate = tree.expressions[0].unalias()
    if (not isinstance(aggregate, exp.Sum) or not isinstance(aggregate.this, exp.Column)
            or aggregate.this.name.casefold() == heading[1].casefold()):
        return None
    order = tree.args["order"].expressions
    if (len(order) != 1 or not isinstance(order[0].this, exp.Column)
            or order[0].this.name.casefold() != aggregate.this.name.casefold()
            or not isinstance(tree.args["limit"].expression, exp.Literal)):
        return None
    where = tree.args.get("where")
    if where:
        atom = where.this
        if (not isinstance(atom, exp.Not) or not isinstance(atom.this, exp.Is)
                or not isinstance(atom.this.this, exp.Column)
                or atom.this.this.name.casefold() != aggregate.this.name.casefold()
                or not isinstance(atom.this.expression, exp.Null)):
            return None
    tables = next((e["observation"] for e in observation["history"] if e["tool"] == "schema"), None)
    if tables is None:
        return None
    schema = Schema(tables)
    sources = schema.sources(tree)
    field = heading[1].casefold()
    candidates = [name for name in sources if field in schema.columns[name]
                  and terms(schema.tables[name]["name"]) <= terms(observation["task"]["question"])
                  and re.search(r"int|real|numeric|decimal|float|double", schema.columns[name][field]["type"], re.I)]
    if len(candidates) != 1:
        return None
    (root,) = candidates
    query = exp.select(exp.Sum(this=schema.column(root, field, "measure"))).from_(
        physical_table(schema, root, "measure"))
    sort = order[0].copy()
    sort.set("this", schema.column(root, field, "measure"))
    query.order_by(sort, copy=False)
    query.set("limit", tree.args["limit"].copy())
    return {"sql": sql_text(query), "kind": "scalar_measure_binding", "table": root, "field": field,
            "preserved_aggregate_limit_scope": True}


class MeasureSuffix:
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
        candidate = scalar_measure(observation, self.fallback["sql"])
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
            self.suffix = MeasureSuffix(action)
        return self.suffix.act(observation)


def create_agent(checkpoint=None):
    return Agent(create_baseline(checkpoint))
