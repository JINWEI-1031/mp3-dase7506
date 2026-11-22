"""Rebuild a scalar comparison over an independently bound population."""
import re

from sqlglot import exp

from repair_rules import parse, sql_text
from step4_agent import identity, passed
from step4_edits import Schema
from step5x_edits import physical_table
from step31_population import source
from step36.frequency import terms
from step48_agent import create_agent as create_baseline


ROLE = re.compile(r"\b(older|younger) than (?:all|every)\s+([a-z]+)\s*[?!.]?$", re.I)
AVERAGE = re.compile(r"\b(more|greater|higher|less|smaller|lower) than (?:the )?average(?:\s+([\w -]+))?\s*[?!.]?$", re.I)
ABSOLUTE = re.compile(r"\b(?:or|except|without|not|each|both)\b", re.I)
NUMBER = re.compile(r"int|real|decimal|number|numeric|float|double", re.I)
TEXT = re.compile(r"char|text|clob", re.I)
OCCUPATION_FIELDS = {"job", "job_title", "occupation", "profession"}


def comparison_intent(question):
    if ABSOLUTE.search(question):
        return None
    role = ROLE.search(question)
    if role:
        older = role[1].casefold() == "older"
        return (exp.GT if older else exp.LT, exp.Max if older else exp.Min, role[2], "age")
    average = AVERAGE.search(question)
    if average:
        greater = average[1].casefold() in {"more", "greater", "higher"}
        return (exp.GT if greater else exp.LT, exp.Avg, None, average[2])
    return None


def anchor(sql, comparator, width):
    parent = parse(sql)
    if (not isinstance(parent, exp.Select) or not source(parent) or len(parent.expressions) != width
            or any(parent.args.get(k) for k in ("joins", "group", "having", "limit", "offset", "with"))):
        return None
    where = parent.args.get("where")
    atom = where.this.unnest() if where else None
    if (not isinstance(atom, comparator) or not isinstance(atom.this, exp.Column)
            or atom.this.table.casefold() not in ("", source(parent).alias_or_name.casefold())
            or not isinstance(atom.expression, (exp.Subquery, exp.All, exp.Any))):
        return None
    return parent, atom


def already_independent(atom, root, metric, reducer, role, role_field):
    subquery = atom.expression
    if not isinstance(subquery, exp.Subquery) or not isinstance(subquery.this, exp.Select):
        return False
    query = subquery.this
    table = source(query)
    if (not table or table.name.casefold() != root or len(query.expressions) != 1
            or any(query.args.get(k) for k in ("joins", "group", "having", "limit", "offset", "with"))):
        return False
    agg = query.expressions[0].unalias()
    if (not isinstance(agg, reducer) or not isinstance(agg.this, exp.Column) or agg.this.name.casefold() != metric
            or agg.this.table.casefold() not in ("", table.alias_or_name.casefold())):
        return False
    where = query.args.get("where")
    if role is None:
        return where is None
    predicate = where.this.unnest() if where else None
    return (isinstance(predicate, exp.EQ) and isinstance(predicate.this, exp.Column)
            and predicate.this.name.casefold() == role_field
            and predicate.this.table.casefold() in ("", table.alias_or_name.casefold())
            and isinstance(predicate.expression, exp.Literal) and predicate.expression.this == role)


def build(observation, fallback, intent, tables):
    comparator, reducer, role, named_field = intent
    schema = Schema(tables)
    width = len(observation["example"]["columns"])
    question = observation["task"]["question"]
    # Prefer the baseline to preserve its projection. Recover a usable
    # outer query from actual checks if its original AST cannot be parsed.
    sources = [fallback, observation["issue_sql"], *(e["sql"] for e in observation["history"] if e["tool"] == "check")]
    for sql in dict.fromkeys(sources):
        bound = anchor(sql, comparator, width)
        if not bound:
            continue
        parent, atom = bound
        outer = source(parent)
        root, metric = outer.name.casefold(), atom.this.name.casefold()
        fields = schema.columns.get(root, {})
        if metric not in fields or not NUMBER.search(fields[metric]["type"]):
            continue
        if role is not None and metric != named_field:
            continue
        if role is None and (not terms(metric) or (terms(metric) != terms(named_field) if named_field else not terms(metric) <= terms(question))):
            continue
        role_field = None
        if role is not None:
            occupations = [field for field in fields.keys() & OCCUPATION_FIELDS if TEXT.search(fields[field]["type"])]
            if len(occupations) != 1:
                continue
            role_field = occupations[0]
        if already_independent(atom, root, metric, reducer, role, role_field):
            return None

        alias = "comparison_population" if outer.alias_or_name.casefold() != "comparison_population" else "inner_population"
        inner = exp.select(reducer(this=schema.column(root, metric, alias))).from_(physical_table(schema, root, alias))
        if role_field:
            inner.where(exp.EQ(this=schema.column(root, role_field, alias), expression=exp.Literal.string(role)), copy=False)
        query = parent.copy().where(comparator(this=atom.this.copy(), expression=inner.subquery()), append=False, copy=False)
        return {"sql": sql_text(query), "kind": reducer.key, "root": root, "metric": metric,
                "population_field": role_field, "population_value": role, "anchor": sql}
    return None


class PopulationSuffix:
    def __init__(self, fallback):
        self.fallback = fallback
        self.candidate = None

    def act(self, observation):
        history, remaining = observation["history"], observation["remaining"]
        if self.candidate:
            return {"tool": "submit", "sql": self.candidate["sql"]} if passed(history[-1]["observation"]) else self.fallback
        intent = comparison_intent(observation["task"]["question"]) if remaining else None
        tables = next((e["observation"] for e in history if e["tool"] == "schema"), None)
        if intent is None or tables is None:
            return self.fallback
        candidate = build(observation, self.fallback["sql"], intent, tables)
        seen = {identity(e["sql"]) for e in history if e["tool"] == "check"}
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
            self.suffix = PopulationSuffix(action)
        return self.suffix.act(observation)


def create_agent(checkpoint=None):
    return Agent(create_baseline(checkpoint))
