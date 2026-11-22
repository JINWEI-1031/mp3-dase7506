"""Try the last displayed entity's unique numeric measure for an unspecified ordering key."""
import re

from sqlglot import exp

from repair_rules import parse, sql_text
from step4_agent import identity, passed, usable
from step4_edits import Schema
from step38.event_grain import bind_column
from step91_agent import create_agent as create_baseline


DIRECTION = re.compile(r"\bin (ascending|descending) order\s*[?.!]?$", re.I)


def ordering_candidate(observation, sql):
    cue = DIRECTION.search(observation["task"]["question"])
    tree = parse(sql) if cue and observation["ordered"] else None
    if (not isinstance(tree, exp.Select) or len(tree.expressions) < 2
            or any(tree.args.get(key) for key in ("distinct", "group", "having", "limit", "offset", "with", "where"))
            or not tree.args.get("order") or len(tree.args["order"].expressions) != 1):
        return None
    tables = next((e["observation"] for e in observation["history"] if e["tool"] == "schema"), None)
    if tables is None:
        selected = [item.unalias() for item in tree.expressions]
        sort = tree.args["order"].expressions[0].this
        if (all(isinstance(item, exp.Column) and item.table for item in selected)
                and isinstance(sort, exp.Column) and sort.table
                and any(sort == item for item in selected) and sort.table != selected[-1].table):
            return {"request_schema": True}
        return None
    schema = Schema(tables)
    sources = schema.sources(tree)
    projected = [bind_column(item.unalias(), schema, sources) for item in tree.expressions]
    sort = tree.args["order"].expressions[0]
    current = bind_column(sort.this, schema, sources)
    if (not all(projected) or current not in projected or len(set(table for table, _ in projected)) < 2
            or current[0] == projected[-1][0]):
        return None
    table = projected[-1][0]
    keys = {fk["from"].casefold() for fk in schema.tables[table]["foreign_keys"]}
    measures = [name for name, meta in schema.columns[table].items()
                if not meta["pk"] and name not in keys and not name.endswith("_id")
                and re.search(r"int|real|numeric|decimal|float|double", meta["type"], re.I)]
    if len(measures) != 1:
        return None
    (field,) = measures
    descending = cue[1].casefold() == "descending"
    query = tree.copy().order_by(exp.Ordered(
        this=schema.column(table, field, sources[table]), desc=descending, nulls_first=not descending), append=False)
    return {"sql": sql_text(query), "kind": "unspecified_order_measure",
            "table": table, "field": field, "heuristic": "unique measure on the last displayed entity"}


class OrderingSuffix:
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
        candidate = ordering_candidate(observation, self.fallback["sql"])
        if candidate is None:
            return self.fallback
        if candidate.get("request_schema"):
            return {"tool": "schema"} if observation["remaining"] >= 2 else self.fallback
        if not usable(candidate["sql"]):
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
            self.suffix = OrderingSuffix(action)
        return self.suffix.act(observation)


def create_agent(checkpoint=None):
    return Agent(create_baseline(checkpoint))
