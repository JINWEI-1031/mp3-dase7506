"""Test the opposite endpoint of a parallel-key follower-count relationship."""
import re

from sqlglot import exp

from repair_rules import parse, sql_text
from step4_agent import identity, passed, usable
from step4_edits import Schema
from step38.event_grain import bind_column
from step98_agent import create_agent as create_baseline


def follower_shape(observation, sql):
    if not re.search(r"\bfollowers?\b", observation["task"]["question"], re.I):
        return None
    tree = parse(sql)
    if (not isinstance(tree, exp.Select) or len(tree.args.get("joins", [])) != 1
            or any(tree.args.get(key) for key in ("distinct", "where", "order", "limit", "offset", "with"))
            or not tree.args.get("group") or not tree.args.get("having")):
        return None
    join = tree.args["joins"][0]
    threshold = tree.args["having"].this
    if (join.side or join.kind not in ("", "INNER") or not isinstance(join.args.get("on"), exp.EQ)
            or not isinstance(threshold, (exp.GT, exp.GTE))
            or not isinstance(threshold.this, exp.Count) or not isinstance(threshold.this.this, exp.Column)
            or not isinstance(threshold.expression, exp.Literal) or threshold.expression.is_string):
        return None
    return tree, join, threshold


def reverse_endpoint(shape, tables):
    tree, join, threshold = shape
    schema = Schema(tables)
    sources = schema.sources(tree)
    root = tree.args.get("from")
    if root is None or not isinstance(root.this, exp.Table) or not isinstance(join.this, exp.Table) or len(sources) != 2:
        return None
    entity, association = root.this.name.casefold(), join.this.name.casefold()
    keys = [name for name, meta in schema.columns[entity].items() if meta["pk"]]
    if len(keys) != 1:
        return None
    (key,) = keys
    links = {fk["from"].casefold() for fk in schema.tables[association]["foreign_keys"]
             if fk["table"].casefold() == entity and fk["to"].casefold() == key}
    if len(links) != 2 or any(
            (bound := bind_column(item.unalias(), schema, sources)) is None or bound[0] != entity
            for item in tree.expressions):
        return None
    bindings = [bind_column(col, schema, sources) for col in (join.args["on"].this, join.args["on"].expression)]
    if None in bindings or (entity, key) not in bindings:
        return None
    used = next((field for table, field in bindings if table == association and field in links), None)
    counted = bind_column(threshold.this.this, schema, sources)
    other = links - {used}
    if used is None or len(other) != 1 or counted != (association, next(iter(other))):
        return None
    if [bind_column(item, schema, sources) for item in tree.args["group"].expressions] != [(entity, key)]:
        return None
    query = tree.copy()
    query.args["joins"][0].set("on", exp.EQ(
        this=schema.column(entity, key, sources[entity]),
        expression=schema.column(association, next(iter(other)), sources[association])))
    query.args["having"].this.set("this", exp.Count(this=exp.Star()))
    return {"sql": sql_text(query), "kind": "opposite_follower_endpoint",
            "from": used, "to": next(iter(other)), "association": association,
            "heuristic": "test the alternate role when the two declared endpoints share one entity key"}


class FollowerSuffix:
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
        shape = follower_shape(observation, self.fallback["sql"])
        if shape is None:
            return self.fallback
        tables = next((e["observation"] for e in observation["history"] if e["tool"] == "schema"), None)
        if tables is None:
            return {"tool": "schema"} if observation["remaining"] >= 2 else self.fallback
        candidate = reverse_endpoint(shape, tables)
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
            self.suffix = FollowerSuffix(action)
        return self.suffix.act(observation)


def create_agent(checkpoint=None):
    return Agent(create_baseline(checkpoint))
