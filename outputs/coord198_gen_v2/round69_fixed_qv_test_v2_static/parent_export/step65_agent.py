"""Guard a D scalar comparison whose named population is missing from SQL."""

import time

from sqlglot import exp

from repair_rules import sql_text
from step4_agent import identity, passed
from step4_edits import Schema
from step31_population import source
from step53_agent import NUMBER, OCCUPATION_FIELDS, TEXT, anchor, comparison_intent
from step63_agent import create_agent as create_d


def proposal(observation, sql):
    intent = comparison_intent(observation["task"]["question"])
    tables = next((e["observation"] for e in observation["history"] if e["tool"] == "schema"), None)
    if not intent or intent[2] is None or tables is None:
        return None
    comparator, reducer, role, metric = intent
    bound = anchor(sql, comparator, len(observation["example"]["columns"]))
    if not bound:
        return None
    parent, atom = bound
    if atom.this.name.casefold() != metric or not isinstance(atom.expression, exp.Subquery):
        return None
    inner = atom.expression.this
    if (not isinstance(inner, exp.Select) or len(inner.expressions) != 1
            or any(inner.args.get(key) for key in ("where", "joins", "group", "having", "order",
                                                   "limit", "offset", "with", "qualify"))):
        return None
    table = source(inner)
    outer = source(parent)
    if table is None or table.name.casefold() != outer.name.casefold():
        return None
    agg = inner.expressions[0].unalias()
    if (not isinstance(agg, reducer) or not isinstance(agg.this, exp.Column)
            or agg.this.name.casefold() != metric
            or agg.this.table.casefold() not in ("", table.alias_or_name.casefold())):
        return None
    schema = Schema(tables)
    root = outer.name.casefold()
    fields = schema.columns.get(root, {})
    occupations = [key for key in fields.keys() & OCCUPATION_FIELDS if TEXT.search(fields[key]["type"])]
    if len(occupations) != 1 or metric not in fields or not NUMBER.search(fields[metric]["type"]):
        return None
    field = occupations[0]
    inner.where(exp.EQ(this=schema.column(root, field, table.alias_or_name),
                       expression=exp.Literal.string(role.casefold())), copy=False)
    return {"sql": sql_text(parent), "metric": metric, "reducer": reducer.key,
            "population_field": field, "population_value": role.casefold()}


class PopulationGuard:
    def __init__(self, fallback):
        self.fallback = fallback
        self.candidate = None
        self.wall_seconds = 0.0
        self.cpu_seconds = 0.0

    def act(self, observation):
        history = observation["history"]
        if self.candidate:
            if passed(history[-1]["observation"]):
                return {"tool": "submit", "sql": self.candidate["sql"]}
            return self.fallback
        if not observation["remaining"]:
            return self.fallback
        wall, cpu = time.perf_counter(), time.process_time()
        try:
            self.candidate = proposal(observation, self.fallback["sql"])
            if self.candidate:
                seen = {identity(e["sql"]) for e in history if e["tool"] == "check"}
                if identity(self.candidate["sql"]) in seen:
                    self.candidate = None
            if self.candidate:
                return {"tool": "check", "sql": self.candidate["sql"]}
            return self.fallback
        finally:
            self.wall_seconds += time.perf_counter() - wall
            self.cpu_seconds += time.process_time() - cpu


class Agent:
    def __init__(self, baseline):
        self.baseline = baseline
        self.guard = None

    def act(self, observation):
        if self.guard is not None:
            return self.guard.act(observation)
        action = self.baseline.act(observation)
        primary = self.baseline.suffix
        if action["tool"] == "submit" and primary is not None and primary.reason == "public_match":
            self.guard = PopulationGuard(action)
            return self.guard.act(observation)
        return action


def create_agent(checkpoint=None):
    return Agent(create_d(checkpoint))
