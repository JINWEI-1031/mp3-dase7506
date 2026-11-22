"""Restore a named person's date boundary in a scalar membership draft."""
import re

from sqlglot import exp

from repair_rules import parse, sql_text
from step4_agent import identity, passed, usable
from step4_edits import Schema
from step5x_edits import physical_table
from step21_carry import conjuncts
from step31_population import source
from step57_agent import single_pk
from step7_semantics import terms
from step63_agent import create_agent as create_baseline


BOUNDARY = re.compile(r"\b(before|after) the date of any\b", re.I)
DATE = re.compile(r"date|time", re.I)


def shape(observation, sql):
    match = BOUNDARY.search(observation["task"]["question"])
    tree = parse(sql) if match else None
    if (not isinstance(tree, exp.Select) or source(tree) is None
            or any(tree.args.get(k) for k in ("joins", "group", "having", "order", "limit", "offset", "with", "distinct"))):
        return None
    where = tree.args.get("where")
    atom = where.this if where else None
    if (not isinstance(atom, exp.EQ) or not isinstance(atom.this, exp.Column)
            or not isinstance(atom.expression, exp.Subquery)):
        return None
    inner = atom.expression.this
    if (not isinstance(inner, exp.Select) or source(inner) is None or len(inner.expressions) != 1
            or not isinstance(inner.expressions[0], exp.Column) or not inner.args.get("where")
            or any(inner.args.get(k) for k in ("joins", "group", "having", "order", "limit", "offset", "with", "distinct"))):
        return None
    filters = list(conjuncts(inner.args["where"].this))
    if len(filters) != 2 or any(not isinstance(test, exp.EQ) or not isinstance(test.this, exp.Column)
                               or not isinstance(test.expression, exp.Literal) or not test.expression.is_string for test in filters):
        return None
    first = [test for test in filters if test.this.name.casefold().endswith("first_name")]
    last = [test for test in filters if test.this.name.casefold().endswith("last_name")]
    if len(first) != 1 or len(last) != 1:
        return None
    parts = first[0].expression.this.split()
    if (len(parts) != 2 or parts[1] != last[0].expression.this
            or not re.search(rf"\b{re.escape(' '.join(parts))}\b", observation["task"]["question"])):
        return None
    return tree, atom, inner, filters, parts, match[1].casefold()


def compile_boundary(observation, original, tables):
    tree, atom, inner, filters, parts, direction = original
    schema = Schema(tables)
    root, person = source(tree), source(inner)
    a, b = root.name.casefold(), person.name.casefold()
    if (root.db or root.catalog or person.db or person.catalog or a == b
            or len(schema.sources(tree)) != 2 or not single_pk(schema, b, inner.expressions[0].name.casefold())
            or atom.this.table.casefold() not in ("", root.alias_or_name.casefold())):
        return None
    fk = (a, atom.this.name.casefold(), b, inner.expressions[0].name.casefold())
    if fk not in schema.edges[a]:
        return None
    if any(not isinstance(item.unalias(), exp.Column)
           or item.unalias().name.casefold() not in schema.columns[a]
           or item.unalias().table.casefold() not in ("", root.alias_or_name.casefold()) for item in tree.expressions):
        return None
    role = terms(fk[1]) - terms(fk[3]) - {"by"}
    metrics = [name for name, meta in schema.columns[a].items() if DATE.search(meta["type"])
               and DATE.search(name) and role and role <= terms(name)]
    if len(metrics) != 1 or any(test.this.name.casefold() not in schema.columns[b]
                               or test.this.table.casefold() not in ("", person.alias_or_name.casefold()) for test in filters):
        return None
    (metric,) = metrics
    population = exp.select((exp.Min if direction == "before" else exp.Max)(
        this=schema.column(a, metric, "boundary_rows"))).from_(physical_table(schema, a, "boundary_rows"))
    population.join(physical_table(schema, b, "boundary_person"), on=exp.EQ(
        this=schema.column(a, fk[1], "boundary_rows"), expression=schema.column(b, fk[3], "boundary_person")), copy=False)
    for test in filters:
        value = parts[0] if test.this.name.casefold().endswith("first_name") else parts[1]
        population.where(exp.EQ(this=schema.column(b, test.this.name.casefold(), "boundary_person"),
                                expression=exp.Literal.string(value)), copy=False)
    query = tree.copy()
    query.where((exp.LT if direction == "before" else exp.GT)(
        this=schema.column(a, metric, root.alias_or_name), expression=population.subquery()), append=False, copy=False)
    return {"sql": sql_text(query), "kind": "named_date_boundary", "metric": metric,
            "person_key": fk, "reducer": "min" if direction == "before" else "max"}


class BoundarySuffix:
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
        proposal = compile_boundary(observation, original, tables)
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
            self.suffix = BoundarySuffix(action)
        return self.suffix.act(observation)


def create_agent(checkpoint=None):
    return Agent(create_baseline(checkpoint))
