"""Bind an architectural-style adjective to a field whose observed values describe style."""
import re

from sqlglot import exp

from repair_rules import parse, sql_text
from step4_agent import identity, passed, usable
from step4_edits import Schema
from step57_agent import parent_and_atom
from step70_agent import storage_probe
from step87_agent import create_agent as create_baseline


STYLES = {"traditional", "modern", "contemporary", "classic", "rustic", "minimalist", "industrial"}
FIELDS = {"decor", "style", "design"}


def style_shape(observation, sql):
    shape = parent_and_atom(sql)
    if shape is None:
        return None
    tree, root, atom = shape
    value = atom.expression.this
    if (not atom.expression.is_string or value.casefold() not in STYLES
            or atom.this.name.casefold() in FIELDS or root.db or root.catalog
            or not re.search(rf"\b{re.escape(value)}\b", observation["task"]["question"], re.I)
            or any(not isinstance(item.unalias(), exp.Column) for item in tree.expressions)
            or len(tree.expressions) != len(observation["example"]["columns"])):
        return None
    return tree, root, atom


class StyleSuffix:
    def __init__(self, fallback):
        self.fallback = fallback
        self.probe = None
        self.candidate = None
        self.evidence = None

    def act(self, observation):
        history, remaining = observation["history"], observation["remaining"]
        if self.candidate is not None:
            return {"tool": "submit", "sql": self.candidate} if passed(history[-1]["observation"]) else self.fallback
        if not remaining:
            return self.fallback
        shape = style_shape(observation, self.fallback["sql"])
        if shape is None:
            return self.fallback
        tree, root, atom = shape
        tables = next((e["observation"] for e in history if e["tool"] == "schema"), None)
        if tables is None:
            return {"tool": "schema"} if remaining >= 3 else self.fallback
        schema = Schema(tables)
        table = root.name.casefold()
        fields = schema.columns.get(table, {})
        choices = [name for name, meta in fields.items() if name in FIELDS
                   and re.search(r"text|char|clob", meta["type"], re.I)]
        if (len(choices) != 1 or atom.this.name.casefold() not in fields
                or any(item.unalias().name.casefold() not in fields
                       or item.unalias().table.casefold() not in ("", root.alias_or_name.casefold())
                       for item in tree.expressions)):
            return self.fallback
        (field,) = choices
        if self.probe is None:
            if remaining < 2:
                return self.fallback
            self.probe = storage_probe(schema, (table, field))
        sampled = next((e["observation"] for e in history if e["tool"] == "execute" and e["sql"] == self.probe), None)
        if sampled is None:
            return {"tool": "execute", "sql": self.probe}
        if not sampled["ok"] or sampled["truncated"] or not sampled["rows"]:
            return self.fallback
        values = {row[0] for row in sampled["rows"] if len(row) == 1}
        if not values or any(not isinstance(value, str) or value.casefold() not in STYLES for value in values):
            return self.fallback
        if all(value == value.lower() for value in values):
            literal = atom.expression.this.lower()
        elif all(value == value.title() for value in values):
            literal = atom.expression.this.title()
        else:
            return self.fallback
        atom.set("this", schema.column(table, field, root.alias_or_name))
        atom.set("expression", exp.Literal.string(literal))
        sql = sql_text(tree)
        if not usable(sql):
            return self.fallback
        self.evidence = {"kind": "observed_style_field", "sql": sql, "field": field, "values": sorted(values)}
        previous = next((e["observation"] for e in history if e["tool"] == "check"
                         and identity(e["sql"]) == identity(sql)), None)
        if previous is not None:
            return {"tool": "submit", "sql": sql} if passed(previous) else self.fallback
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
            self.suffix = StyleSuffix(action)
        return self.suffix.act(observation)


def create_agent(checkpoint=None):
    return Agent(create_baseline(checkpoint))
