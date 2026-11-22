"""Remove an unnamed join from an all-entity list and honor an undirected sort request."""
import re

from sqlglot import exp

from repair_rules import parse, sql_text
from step4_agent import identity, passed, usable
from step4_edits import Schema
from step38.event_grain import bind_column, join_graph, terms
from step89_agent import create_agent as create_baseline


ORDERED = re.compile(r"\bordered by ([a-z ]+)\s*[?.!]?$", re.I)


def all_entity_order(observation, sql):
    question = observation["task"]["question"]
    cue = ORDERED.search(question)
    tree = parse(sql) if cue else None
    if (not isinstance(tree, exp.Select) or not re.search(r"\b(?:every|all)\b", question, re.I)
            or re.search(r"\b(?:ascending|descending|highest|lowest|largest|smallest|most|least)\b", question, re.I)
            or any(tree.args.get(key) for key in ("distinct", "group", "having", "limit", "offset", "with"))
            or len(tree.args.get("joins", [])) != 1 or not tree.args.get("order")):
        return None
    root = tree.args.get("from")
    if root is None or not isinstance(root.this, exp.Table):
        return None
    tables = next((e["observation"] for e in observation["history"] if e["tool"] == "schema"), None)
    if tables is None:
        return None
    schema = Schema(tables)
    sources = schema.sources(tree)
    table = root.this.name.casefold()
    if len(sources) != 2 or not join_graph(tree, schema, sources):
        return None
    child = next(name for name in sources if name != table)
    child_words = terms(schema.tables[child]["name"]) - terms(schema.tables[table]["name"])
    if not child_words or child_words & terms(question):
        return None
    projections = [bind_column(item.unalias(), schema, sources) for item in tree.expressions]
    if (not projections or any(bound is None or bound[0] != table for bound in projections)
            or [item.alias_or_name.casefold() for item in tree.expressions]
            != [head.casefold() for head in observation["example"]["columns"]]):
        return None
    order = tree.args["order"].expressions
    if len(order) < 2:
        return None
    mentioned = terms(cue[1])
    if any(not isinstance(item.this, exp.Column)
           or (bound := bind_column(item.this, schema, sources)) is None or bound[0] != table
           or not terms(bound[1]) & mentioned for item in order):
        return None
    where = tree.args.get("where")
    if where:
        atom = where.this
        if (not isinstance(atom, exp.EQ) or not isinstance(atom.expression, exp.Literal)
                or not atom.expression.is_string or atom.expression.this.casefold() not in question.casefold()
                or (bound := bind_column(atom.this, schema, sources)) is None or bound[0] != table):
            return None
    query = tree.copy()
    query.set("joins", None)
    for item in query.args["order"].expressions:
        item.set("desc", False)
        item.set("nulls_first", True)
    return {"sql": sql_text(query), "kind": "all_entities_default_order",
            "root": table, "removed_join": child, "ordering": "ascending"}


class OrderSuffix:
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
        candidate = all_entity_order(observation, self.fallback["sql"])
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
            self.suffix = OrderSuffix(action)
        return self.suffix.act(observation)


def create_agent(checkpoint=None):
    return Agent(create_baseline(checkpoint))
