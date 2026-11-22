"""Combine an independently declared JOIN correction with a requested status inflection."""
import re

from sqlglot import exp

from repair_rules import parse, sql_text
from step4_agent import identity, passed, usable
from step4_edits import Schema, key_edits
from step10_bindings import field_words
from step38.event_grain import bind_column
from step92_agent import create_agent as create_baseline


STATUS = {"nomination": "nominated", "qualification": "qualified", "disqualification": "disqualified"}


def status_join_candidate(observation, sql):
    tree = parse(sql)
    if (not isinstance(tree, exp.Select) or len(tree.args.get("joins", [])) != 1
            or any(tree.args.get(key) for key in ("distinct", "group", "having", "order", "limit", "offset", "with"))):
        return None
    where = tree.args.get("where")
    atom = where.this if where else None
    if (not isinstance(atom, exp.EQ) or not isinstance(atom.this, exp.Column)
            or not isinstance(atom.expression, exp.Literal) or not atom.expression.is_string
            or not ((set(field_words(atom.this.name)) | {atom.this.name.casefold()})
                    & {"result", "status", "outcome"})):
        return None
    value = atom.expression.this
    if (value.casefold() not in STATUS or not re.search(rf"\b{re.escape(value)}\b",
                                                      observation["task"]["question"], re.I)):
        return None
    tables = next((event["observation"] for event in observation["history"] if event["tool"] == "schema"), None)
    if tables is None:
        return None
    schema = Schema(tables)
    sources = schema.sources(tree)
    bound = bind_column(atom.this, schema, sources)
    if (len(sources) != 2 or bound is None
            or not re.search(r"text|char|clob", schema.columns[bound[0]][bound[1]]["type"], re.I)
            or any(bind_column(item.unalias(), schema, sources) is None for item in tree.expressions)):
        return None
    choices = list(key_edits(tree, schema))
    if len(choices) != 1:
        return None
    _, corrected, evidence = choices[0]
    query = parse(corrected)
    literal = STATUS[value.casefold()]
    if value.istitle():
        literal = literal.title()
    elif value.isupper():
        literal = literal.upper()
    query.args["where"].this.set("expression", exp.Literal.string(literal))
    return {"sql": sql_text(query), "kind": "declared_join_and_status",
            "join": evidence, "before": value, "after": literal}


class StatusSuffix:
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
        candidate = status_join_candidate(observation, self.fallback["sql"])
        if candidate is None or not usable(candidate["sql"]):
            return self.fallback
        self.evidence = candidate
        previous = next((event["observation"] for event in observation["history"] if event["tool"] == "check"
                         and identity(event["sql"]) == identity(candidate["sql"])), None)
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
            self.suffix = StatusSuffix(action)
        return self.suffix.act(observation)


def create_agent(checkpoint=None):
    return Agent(create_baseline(checkpoint))
