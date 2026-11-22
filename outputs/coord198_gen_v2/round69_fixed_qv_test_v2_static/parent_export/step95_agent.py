"""Check row count when an entity table's nonunique foreign key was deduplicated."""
import re

from sqlglot import exp

from repair_rules import parse, sql_text
from step4_agent import identity, passed
from step4_edits import Schema
from step38.event_grain import terms
from step94_agent import create_agent as create_baseline


COUNT_ENTITY = re.compile(r"^\s*how many ([a-z]+)\b", re.I)


def entity_row_count(observation, sql):
    cue = COUNT_ENTITY.search(observation["task"]["question"])
    tree = parse(sql) if cue else None
    if (not isinstance(tree, exp.Select) or len(tree.expressions) != 1
            or re.search(r"\b(?:distinct|different|unique|types|kinds)\b", observation["task"]["question"], re.I)
            or any(tree.args.get(key) for key in ("joins", "group", "having", "order", "limit", "offset", "with"))):
        return None
    count = tree.expressions[0].unalias()
    if (not isinstance(count, exp.Count) or not isinstance(count.this, exp.Distinct)
            or len(count.this.expressions) != 1 or not isinstance(count.this.expressions[0], exp.Column)):
        return None
    root = tree.args.get("from")
    column = count.this.expressions[0]
    if root is None or not isinstance(root.this, exp.Table) or root.this.db or root.this.catalog:
        return None
    tables = next((e["observation"] for e in observation["history"] if e["tool"] == "schema"), None)
    if tables is None:
        return None
    schema = Schema(tables)
    name = root.this.name.casefold()
    fields = schema.columns.get(name, {})
    if (name not in schema.tables or terms(schema.tables[name]["name"]) != terms(cue[1])
            or any(meta["pk"] for meta in fields.values()) or column.name.casefold() not in fields
            or column.table.casefold() not in ("", root.this.alias_or_name.casefold())
            or not any(fk["from"].casefold() == column.name.casefold()
                       for fk in schema.tables[name]["foreign_keys"])):
        return None
    where = tree.args.get("where")
    if where is None or not isinstance(where.this, exp.In):
        return None
    query = tree.copy()
    query.expressions[0].unalias().set("this", exp.Star())
    return {"sql": sql_text(query), "kind": "entity_table_row_count", "table": name,
            "nonunique_counted_key": column.name}


class CountSuffix:
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
        candidate = entity_row_count(observation, self.fallback["sql"])
        if candidate is None:
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
            self.suffix = CountSuffix(action)
        return self.suffix.act(observation)


def create_agent(checkpoint=None):
    return Agent(create_baseline(checkpoint))
