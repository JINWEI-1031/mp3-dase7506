"""Expand a spelled-out degree level into standard short degree codes."""
import re

from sqlglot import exp

from repair_rules import parse, sql_text
from step4_agent import identity, passed
from step4_edits import Schema
from step10_bindings import field_words
from step59_agent import bound_literals, eligible_literals, sample_sql
from step93_agent import create_agent as create_baseline


DEGREES = {"masters": ("MA", "MS", "MSc", "MEng", "MEd", "MBA", "MPhil"),
           "bachelors": ("BA", "BS", "BSc", "BEng", "BBA"),
           "doctorate": ("Ph.D.", "PhD", "DPhil")}
WIDTH = re.compile(r"(?:var)?char\(\s*(\d+)\s*\)", re.I)


class DegreeSuffix:
    def __init__(self, fallback):
        self.fallback = fallback
        self.probe = None
        self.candidate = None
        self.evidence = None

    def act(self, observation):
        history, remaining = observation["history"], observation["remaining"]
        if self.candidate:
            return {"tool": "submit", "sql": self.candidate} if passed(history[-1]["observation"]) else self.fallback
        if not remaining:
            return self.fallback
        tree, literals = eligible_literals(self.fallback["sql"], observation["task"]["question"])
        items = [(atom, value, kind) for atom, value, kind in literals
                 if isinstance(atom, exp.In) and value.this.casefold() in DEGREES
                 and "degree" in field_words(atom.this.name)]
        if len(items) != 1 or not isinstance(tree, exp.Select):
            return self.fallback
        atom, literal, _ = items[0]
        tables = next((e["observation"] for e in history if e["tool"] == "schema"), None)
        if tables is None:
            return {"tool": "schema"} if remaining >= 3 else self.fallback
        schema = Schema(tables)
        binding = bound_literals(tree, items, schema)
        if binding is None:
            return self.fallback
        width = WIDTH.fullmatch(schema.columns[binding[0]][binding[1]]["type"])
        if width is None or len(literal.this) <= int(width[1]):
            return self.fallback
        if self.probe is None:
            self.probe = sample_sql(schema, binding)
        sample = next((e["observation"] for e in history if e["tool"] == "execute" and e["sql"] == self.probe), None)
        if sample is None:
            return {"tool": "execute", "sql": self.probe} if remaining >= 2 else self.fallback
        values = {row[0] for row in sample["rows"] if len(row) == 1}
        if (not sample["ok"] or sample["truncated"] or not values or
                any(not isinstance(value, str) or not re.fullmatch(r"[A-Za-z.]{1,5}", value) for value in values)):
            return self.fallback
        codes = [value for value in DEGREES[literal.this.casefold()] if len(value) <= int(width[1])]
        if all(value == value.lower() for value in values):
            codes = [value.lower() for value in codes]
        old = literal.this
        expressions = []
        for value in atom.expressions:
            expressions.extend([exp.Literal.string(code) for code in codes] if value is literal else [value.copy()])
        atom.set("expressions", expressions)
        sql = sql_text(tree)
        self.evidence = {"kind": "standard_degree_codes", "field": binding,
                         "observed": sorted(values), "before": old, "codes": codes, "sql": sql}
        if any(e["tool"] == "check" and identity(e["sql"]) == identity(sql) for e in history):
            return self.fallback
        self.candidate = sql
        return {"tool": "check", "sql": sql}


class Agent:
    def __init__(self, baseline):
        self.baseline = baseline
        self.suffix = None

    def act(self, observation):
        if self.suffix is None:
            action = self.baseline.act(observation)
            if action["tool"] != "submit":
                return action
            self.suffix = DegreeSuffix(action)
        return self.suffix.act(observation)


def create_agent(checkpoint=None):
    return Agent(create_baseline(checkpoint))
