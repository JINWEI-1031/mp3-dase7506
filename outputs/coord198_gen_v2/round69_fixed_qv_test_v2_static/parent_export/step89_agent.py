"""Check the named output endpoint in a relationship with two parallel foreign keys."""
import re

from sqlglot import exp

from repair_rules import parse, sql_text
from step4_agent import identity, passed, usable
from step4_edits import Schema
from step38.event_grain import bind_column, join_graph, terms
from step88_agent import create_agent as create_baseline


def endpoint_candidate(observation, sql):
    tree = parse(sql)
    heads = observation["example"]["columns"]
    if (not isinstance(tree, exp.Select) or len(heads) != 1 or len(tree.expressions) != 1
            or any(tree.args.get(key) for key in ("group", "having", "order", "limit", "offset", "with"))
            or len(tree.args.get("joins", [])) != 1 or tree.expressions[0].alias
            or not isinstance(tree.expressions[0], exp.Column)):
        return None
    selected = tree.expressions[0]
    label = heads[0].casefold()
    if selected.name.casefold() == label or not selected.table:
        return None
    tables = next((e["observation"] for e in observation["history"] if e["tool"] == "schema"), None)
    if tables is None:
        return None
    schema = Schema(tables)
    sources = schema.sources(tree)
    bound = bind_column(selected, schema, sources)
    if bound is None or not join_graph(tree, schema, sources):
        return None
    table, field = bound
    columns = schema.columns[table]
    if (label not in columns or not re.search(r"text|char|clob", columns[label]["type"], re.I)
            or not terms(schema.tables[table]["name"]) & terms(observation["task"]["question"])):
        return None
    keys = schema.tables[table]["foreign_keys"]
    matching = [(a, b) for a in keys for b in keys if a["from"].casefold() == field
                and b["from"].casefold() == label and a["table"].casefold() == b["table"].casefold()
                and a["to"].casefold() == b["to"].casefold() and a["table"].casefold() in sources]
    if len(matching) != 1:
        return None
    query = tree.copy().select(schema.column(table, label, sources[table]), append=False)
    if not re.search(r"\b(?:distinct|unique|different)\b", observation["task"]["question"], re.I):
        query.set("distinct", None)
    return {"sql": sql_text(query), "kind": "parallel_foreign_key_output",
            "table": table, "before": field, "after": label}


class EndpointSuffix:
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
        candidate = endpoint_candidate(observation, self.fallback["sql"])
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
            self.suffix = EndpointSuffix(action)
        return self.suffix.act(observation)


def create_agent(checkpoint=None):
    return Agent(create_baseline(checkpoint))
