"""Test named association-row frequency against an inherited distinct-key ranking."""
import re

from sqlglot import exp

from repair_rules import parse, sql_text
from step4_agent import identity, passed, usable
from step4_edits import Schema
from step38.event_grain import bind_column, join_graph, terms
from step97_agent import create_agent as create_baseline


def association_frequency(observation, sql):
    question = observation["task"]["question"]
    named = re.search(r"\bin ([a-z]+)\s*[?.!]?$", question, re.I)
    tree = parse(sql) if named and re.search(r"\bmost\b", question, re.I) else None
    if (not isinstance(tree, exp.Select) or len(tree.expressions) != 1
            or any(tree.args.get(key) for key in ("distinct", "where", "having", "offset", "with"))
            or not tree.args.get("group") or not tree.args.get("order")
            or len(tree.args["order"].expressions) != 1 or len(tree.args.get("joins", [])) != 2
            or not tree.args.get("limit") or tree.args["limit"].expression != exp.Literal.number(1)
            or re.search(r"\b(?:distinct|unique|different)\b", question, re.I)):
        return None
    count = tree.args["order"].expressions[0].this
    if (not isinstance(count, exp.Count) or not isinstance(count.this, exp.Distinct)
            or len(count.this.expressions) != 1):
        return None
    tables = next((e["observation"] for e in observation["history"] if e["tool"] == "schema"), None)
    if tables is None:
        return None
    schema = Schema(tables)
    sources = schema.sources(tree)
    measured = bind_column(count.this.expressions[0], schema, sources)
    projected = bind_column(tree.expressions[0].unalias(), schema, sources)
    if measured is None or projected is None or not join_graph(tree, schema, sources):
        return None
    carrier, key = measured
    if terms(schema.tables[carrier]["name"]) != terms(named[1]) or carrier == projected[0]:
        return None
    links = [fk for fk in schema.tables[carrier]["foreign_keys"]
             if fk["from"].casefold() == key and fk["table"].casefold() in sources
             and fk["table"].casefold() != projected[0]]
    if len(links) != 1:
        return None
    terminal = links[0]["table"].casefold()
    keep = set(sources) - {terminal}
    for item in [*tree.expressions, *tree.args["group"].expressions]:
        if any((b := bind_column(col, schema, sources)) is None or b[0] not in keep
               for col in item.find_all(exp.Column)):
            return None
    removed = [join for join in tree.args["joins"] if isinstance(join.this, exp.Table)
               and join.this.name.casefold() == terminal]
    if len(removed) != 1:
        return None
    query = tree.copy()
    query.set("joins", [join.copy() for join in tree.args["joins"] if join is not removed[0]])
    if len(schema.sources(query)) != 2 or not join_graph(query, schema, schema.sources(query)):
        return None
    query.args["order"].expressions[0].set("this", exp.Count(this=exp.Star()))
    return {"sql": sql_text(query), "kind": "named_association_frequency",
            "carrier": carrier, "removed_terminal": terminal,
            "heuristic": "test row frequency on the explicitly named association"}


class FrequencySuffix:
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
        candidate = association_frequency(observation, self.fallback["sql"])
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
            self.suffix = FrequencySuffix(action)
        return self.suffix.act(observation)


def create_agent(checkpoint=None):
    return Agent(create_baseline(checkpoint))
