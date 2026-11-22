"""Give the last check to D on unresolved, schema-known tasks."""

from sqlglot import exp
from sqlglot.errors import SqlglotError
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import traverse_scope

from mp3_sql.sqlite_tool import validate_sql
from step4_agent import identity, passed
from step63_agent import active_model, create_agent as create_d, generate_with


def eligible(observation, action):
    history = observation["history"]
    return (observation["remaining"] == 1 and len(history) == 7 and action["tool"] == "check"
            and any(e["tool"] == "schema" for e in history)
            and not any(e["tool"] == "check" and passed(e["observation"]) for e in history))


def candidate_status(sql, observation, scheduled, tables):
    try:
        tree = validate_sql(sql)
    except ValueError:
        return "invalid_sql"
    seen = {identity(e["sql"]) for e in observation["history"] if e["tool"] == "check"}
    if identity(sql) in seen | {identity(scheduled)}:
        return "duplicate"
    schema = {t["name"]: {c["name"]: "UNKNOWN" for c in t["columns"]} for t in tables}
    names = {name.casefold() for name in schema}
    try:
        for scope in traverse_scope(tree):
            for _, relation in scope.selected_sources.values():
                if (isinstance(relation, exp.Table)
                        and (relation.db or relation.catalog or relation.name.casefold() not in names)):
                    return "invalid_binding"
        bound = qualify(tree.copy(), dialect="sqlite", schema=schema, infer_schema=False, identify=False)
    except (SqlglotError, RecursionError):
        return "invalid_binding"
    if len(bound.selects) != len(observation["example"]["columns"]):
        return "projection_width"
    return "replace"


def propose(observation, tables):
    return generate_with(observation, tables, active_model(), purpose="slot8")


class Agent:
    def __init__(self, baseline, proposer=propose):
        self.baseline = baseline
        self.proposer = proposer
        self.replacement = None
        self.status = None

    def choose(self, sql, observation, scheduled, tables):
        return sql, candidate_status(sql, observation, scheduled, tables)

    def act(self, observation):
        history = observation["history"]
        if self.replacement is not None:
            if passed(history[-1]["observation"]):
                return {"tool": "submit", "sql": self.replacement}
            return self.baseline.act(observation)
        action = self.baseline.act(observation)
        if self.baseline.suffix is not None or not eligible(observation, action):
            return action
        tables = next(e["observation"] for e in history if e["tool"] == "schema")
        sql = self.proposer(observation, tables)
        sql, self.status = self.choose(sql, observation, action["sql"], tables)
        if self.status == "replace":
            self.replacement = sql
            return {"tool": "check", "sql": sql}
        return action


def create_agent(checkpoint=None):
    return Agent(create_d(checkpoint))
