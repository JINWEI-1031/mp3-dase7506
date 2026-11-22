"""Intersect projected labels when the question requires membership in both categories."""
import re

from sqlglot import exp

from repair_rules import parse, sql_text
from step4_agent import identity, passed, usable
from step4_edits import Schema
from step5x_edits import physical_table
from step7_semantics import terms
from step63_agent import create_agent as create_baseline


BOTH = re.compile(r"\bboth in (?:the )?(.+?) and (.+?) ([a-z]+)\s*[?.!]?$", re.I)


def shape(observation, sql):
    cue = BOTH.search(observation["task"]["question"])
    tree = parse(sql) if cue else None
    if (not isinstance(tree, exp.Select) or len(tree.expressions) != 1
            or any(tree.args.get(k) for k in ("group", "having", "order", "limit", "offset", "with"))
            or not isinstance(tree.expressions[0].unalias(), exp.Column)):
        return None
    where = tree.args.get("where")
    atom = where.this if where else None
    if (not isinstance(atom, exp.In) or atom.args.get("query") or len(atom.expressions) != 2
            or not isinstance(atom.this, exp.Column)
            or any(not isinstance(value, exp.Literal) or not value.is_string for value in atom.expressions)
            or {value.this.casefold() for value in atom.expressions} != {cue[1].casefold(), cue[2].casefold()}
            or cue[3].casefold() not in {atom.this.name.casefold(), atom.this.name.casefold() + "s",
                                        atom.this.name.casefold() + "es"}):
        return None
    return tree, atom, cue


def compile_intersection(observation, shape, tables):
    tree, atom, cue = shape
    schema = Schema(tables)
    source = tree.args.get("from")
    if source is None or not isinstance(source.this, exp.Table) or source.this.db or source.this.catalog:
        return None
    root = source.this.name.casefold()
    projected = tree.expressions[0].unalias()
    field, label = atom.this.name.casefold(), projected.name.casefold()
    if (root not in schema.tables or field not in schema.columns[root] or label not in schema.columns[root]
            or projected.table.casefold() not in ("", source.this.alias_or_name.casefold())
            or not terms(root) & terms(observation["task"]["question"][:cue.start()])
            or [projected.name.casefold()] != [head.casefold() for head in observation["example"]["columns"]]
            or not re.search(r"text|char|clob", schema.columns[root][field]["type"], re.I)):
        return None
    if any(join.side or join.kind not in ("", "INNER") or not isinstance(join.this, exp.Table)
           for join in tree.args.get("joins", [])):
        return None
    branches = [exp.select(schema.column(root, label, "member")).from_(physical_table(schema, root, "member"))
                .where(exp.EQ(this=schema.column(root, field, "member"), expression=value.copy()))
                for value in atom.expressions]
    query = exp.Intersect(this=branches[0], expression=branches[1], distinct=True)
    return {"sql": sql_text(query), "kind": "label_intersection",
            "entity": root, "label": label, "category": field}


class IntersectionSuffix:
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
        original = shape(observation, self.fallback["sql"])
        if original is None:
            return self.fallback
        tables = next((e["observation"] for e in observation["history"] if e["tool"] == "schema"), None)
        if tables is None:
            return {"tool": "schema"} if observation["remaining"] >= 2 else self.fallback
        proposal = compile_intersection(observation, original, tables)
        if proposal is None or not usable(proposal["sql"]):
            return self.fallback
        self.evidence = proposal
        if any(e["tool"] == "check" and identity(e["sql"]) == identity(proposal["sql"])
               for e in observation["history"]):
            return self.fallback
        self.candidate = proposal["sql"]
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
            self.suffix = IntersectionSuffix(action)
        return self.suffix.act(observation)


def create_agent(checkpoint=None):
    return Agent(create_baseline(checkpoint))
