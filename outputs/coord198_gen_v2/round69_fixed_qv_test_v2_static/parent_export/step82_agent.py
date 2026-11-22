"""Replace a key aliased as a requested label with that table's real label field."""
import re

from sqlglot import exp

from repair_rules import parse, sql_text
from step4_agent import identity, passed
from step4_edits import Schema
from step63_agent import create_agent as create_baseline


def label_candidate(observation, sql):
    tree = parse(sql)
    if not isinstance(tree, exp.Select) or tree.args.get("joins") or len(tree.expressions) != 1:
        return None
    root = tree.args.get("from")
    item = tree.expressions[0]
    if (root is None or not isinstance(root.this, exp.Table) or root.this.db or root.this.catalog
            or not isinstance(item, exp.Alias) or not isinstance(item.this, exp.Column)
            or item.this.table.casefold() not in ("", root.this.alias_or_name.casefold())
            or [item.alias.casefold()] != [head.casefold() for head in observation["example"]["columns"]]
            or not re.search(rf"\b{re.escape(item.alias)}(?:s|es)?\b", observation["task"]["question"], re.I)):
        return None
    tables = next((e["observation"] for e in observation["history"] if e["tool"] == "schema"), None)
    if tables is None:
        return None
    schema = Schema(tables)
    name, field, label = root.this.name.casefold(), item.this.name.casefold(), item.alias.casefold()
    fields = schema.columns.get(name, {})
    if (field == label or label not in fields or field not in fields or not fields[field]["pk"]
            or not re.search(r"int|decimal|numeric", fields[field]["type"], re.I)
            or not re.search(r"text|char|clob", fields[label]["type"], re.I)):
        return None
    item.set("this", schema.column(name, label, root.this.alias_or_name))
    return {"sql": sql_text(tree), "kind": "aliased_key_label", "table": name, "before": field, "after": label}


class LabelSuffix:
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
        candidate = label_candidate(observation, self.fallback["sql"])
        if candidate is None:
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
            self.suffix = LabelSuffix(action)
        return self.suffix.act(observation)


def create_agent(checkpoint=None):
    return Agent(create_baseline(checkpoint))
