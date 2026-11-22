"""Count distinct purchased entities along a unique declared relationship path."""
import re

from sqlglot import exp

from repair_rules import parse, sql_text
from step4_agent import identity, passed, usable
from step4_edits import Schema
from step5x_edits import physical_table
from step7_semantics import join_path
from step36.frequency import terms
from step63_agent import create_agent as create_baseline


PURCHASE = re.compile(r"^\s*how many (?:types|kinds) of ([a-z]+) (?:have|has) (.+?) "
                      r"(?:bought|purchased|ordered) in total\s*[?.!]?$", re.I)


def shape(observation, sql):
    match = PURCHASE.fullmatch(observation["task"]["question"])
    tree = parse(sql) if match else None
    if (not isinstance(tree, exp.Select) or len(tree.expressions) != 1
            or len(observation["example"]["columns"]) != 1
            or any(tree.args.get(k) for k in ("group", "having", "order", "limit", "offset", "with"))
            or not isinstance(tree.expressions[0].unalias(), exp.Count)
            or not isinstance(tree.expressions[0].unalias().this, exp.Distinct)):
        return None
    where = tree.args.get("where")
    atom = where.this if where else None
    if (not isinstance(atom, exp.EQ) or not isinstance(atom.this, exp.Column)
            or not isinstance(atom.expression, exp.Literal) or not atom.expression.is_string
            or atom.expression.this != match[2] or not atom.this.name.casefold().endswith("name")):
        return None
    return tree, atom, match[1]


def compile_count(original, tables):
    tree, atom, entity = original
    schema = Schema(tables)
    sources = schema.sources(tree)
    if not sources:
        return None
    origins = [table for table, alias in sources.items()
               if atom.this.name.casefold() in schema.columns[table]
               and atom.this.table.casefold() in ("", alias.casefold())]
    targets = [table for table in schema.tables if terms(table) == terms(entity)]
    if len(origins) != 1 or len(targets) != 1 or origins == targets:
        return None
    origin, target = origins[0], targets[0]
    keys = [name for name, meta in schema.columns[target].items() if meta["pk"]]
    if len(keys) != 1:
        return None
    measure = tree.expressions[0].unalias().this.expressions
    if len(measure) != 1 or not isinstance(measure[0], exp.Column):
        return None
    if measure[0].name.casefold() == keys[0]:
        return None
    paths = {tuple(path) for path in schema.path_options([origin], target, 3)}
    if len(paths) != 1:
        return None
    (path,) = paths
    query = exp.select().from_(physical_table(schema, origin, "buyer"))
    aliases = {origin: "buyer"}
    join_path(query, path, schema, aliases)
    query.select(exp.Count(this=exp.Distinct(expressions=[
        schema.column(target, keys[0], aliases[target])])), append=False, copy=False)
    query.where(exp.EQ(this=schema.column(origin, atom.this.name.casefold(), "buyer"),
                      expression=atom.expression.copy()), copy=False)
    return {"sql": sql_text(query), "kind": "purchased_entity_count", "path": path,
            "entity": target, "key": keys[0]}


class PurchaseSuffix:
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
        candidate = compile_count(original, tables)
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
            self.suffix = PurchaseSuffix(action)
        return self.suffix.act(observation)


def create_agent(checkpoint=None):
    return Agent(create_baseline(checkpoint))
