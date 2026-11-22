"""Repair a declared join without guessing unlabeled relationship directions."""
import re

from sqlglot import exp

from repair_rules import parse, sql_text
from step4_agent import identity, passed
from step4_edits import Schema
from step36.frequency import terms
from step38.event_grain import bind_column
from step48_agent import create_agent as create_baseline


def join_pair(sql, schema):
    tree = parse(sql)
    joins = tree.args.get("joins") if isinstance(tree, exp.Select) else None
    if not joins or len(joins) != 1 or joins[0].side or joins[0].kind not in ("", "INNER"):
        return None
    sources = schema.sources(tree)
    condition = joins[0].args.get("on")
    if len(sources) != 2 or not isinstance(condition, exp.EQ):
        return None
    left, right = bind_column(condition.this, schema, sources), bind_column(condition.expression, schema, sources)
    if not left or not right or left[0] == right[0]:
        return None
    edges = sorted(set(edge for edge in schema.edges[left[0]] if edge[2] == right[0]))
    if not edges:
        return None
    return tree, sources, (*left, *right), edges


def follower_edge(question, tree, schema, sources, current, edges):
    if not re.search(r"\bfollowers?\b", question, re.I) or len(edges) != 2 or tree.args.get("where"):
        return None
    output = [bind_column(p.unalias(), schema, sources) for p in tree.expressions]
    if not output or any(p is None for p in output) or len({p[0] for p in output}) != 1:
        return None
    root = output[0][0]
    normalized = [(a, ak, b, bk) if a == root else (b, bk, a, ak) for a, ak, b, bk in edges]
    if len({edge[1] for edge in normalized}) != 1 or len({edge[2] for edge in normalized}) != 1:
        return None
    group = tree.args.get("group")
    if not group or [bind_column(col, schema, sources) for col in group.expressions] != [normalized[0][:2]]:
        return None
    labels = {edge[3]: set(edge[3].casefold().split("_")) for edge in normalized}
    target = [edge for edge in normalized if labels[edge[3]] & {"followed", "followee"}]
    follower = [edge for edge in normalized if "follower" in labels[edge[3]]]
    if len(target) != 1 or len(follower) != 1 or target[0] == follower[0]:
        return None
    a, ak, b, bk = current
    if (a, ak, b, bk) not in (follower[0], (follower[0][2], follower[0][3], follower[0][0], follower[0][1])):
        return None
    having = tree.args.get("having")
    counts = list(having.find_all(exp.Count)) if having else []
    if len(counts) != 1 or bind_column(counts[0].this, schema, sources) != target[0][2:]:
        return None
    counts[0].set("this", schema.column(follower[0][2], follower[0][3], sources[follower[0][2]]))
    return target[0]


def propose(question, sql, tables):
    schema = Schema(tables)
    evidence = join_pair(sql, schema)
    if not evidence:
        return None
    tree, sources, edge, declared = evidence
    if edge in declared:
        target = follower_edge(question, tree, schema, sources, edge, declared)
        kind = "named_follower_direction"
    else:
        cue = terms(question)
        if len(declared) != 1 or not all(terms(table) & cue for table in sources):
            return None
        target = declared[0]
        kind = "unique_declared_join"
    if not target:
        return None
    a, ak, b, bk = target
    tree.args["joins"][0].set("on", exp.EQ(this=schema.column(a, ak, sources[a]),
                                         expression=schema.column(b, bk, sources[b])))
    return {"sql": sql_text(tree), "kind": kind, "declared_edge": target}


class JoinSuffix:
    def __init__(self, fallback):
        self.fallback = fallback
        self.candidate = None

    def act(self, observation):
        history, remaining = observation["history"], observation["remaining"]
        if self.candidate:
            return {"tool": "submit", "sql": self.candidate["sql"]} if passed(history[-1]["observation"]) else self.fallback
        tables = next((event["observation"] for event in history if event["tool"] == "schema"), None)
        if not remaining or tables is None:
            return self.fallback
        candidate = propose(observation["task"]["question"], self.fallback["sql"], tables)
        seen = {identity(event["sql"]) for event in history if event["tool"] == "check"}
        if candidate and identity(candidate["sql"]) not in seen | {identity(self.fallback["sql"])}:
            self.candidate = candidate
            return {"tool": "check", "sql": candidate["sql"]}
        return self.fallback


class Agent:
    def __init__(self, baseline):
        self.baseline = baseline
        self.suffix = None

    def act(self, observation):
        if self.suffix is None:
            action = self.baseline.act(observation)
            if action["tool"] != "submit":
                return action
            self.suffix = JoinSuffix(action)
        return self.suffix.act(observation)


def create_agent(checkpoint=None):
    return Agent(create_baseline(checkpoint))
