"""Generate case-normalized category predicates from observed storage conventions."""
import re

from sqlglot import exp

from repair_rules import parse, sql_text
from step4_agent import identity, passed, usable
from step4_edits import Schema
from step10_bindings import field_words
from step59_agent import bound_literals, eligible_literals, sample_sql
from step63_agent import create_agent as create_baseline


CATEGORY_FIELDS = {"gender", "sex", "pos", "position", "result", "status"}
TITLE_CASE = re.compile(r"[A-Z][a-z]+")
LOWER_CASE = re.compile(r"[a-z]+(?:[ -][a-z]+)*")


def category_literals(sql, question):
    tree, items = eligible_literals(sql, question)
    return tree, [
        item for item in items
        if isinstance(item[0], (exp.EQ, exp.In))
        and set(field_words(item[0].this.name)) & CATEGORY_FIELDS
        and TITLE_CASE.fullmatch(item[1].this)
        and re.search(rf"""(?<![\w'"]){item[1].this.lower()}s?(?![\w'"])""", question)
    ]


def storage_probe(schema, binding):
    # Fetch one extra row so the tool can signal that the domain was truncated.
    tree = parse(sample_sql(schema, binding))
    column = tree.expressions[0]
    value = exp.Collate(this=column.copy(), expression=exp.Var(this="BINARY"))
    tree.select(value, append=False, copy=False).order_by(value.copy(), append=False, copy=False)
    return sql_text(tree.limit(9, copy=False))


def stored_values(result):
    if not result["ok"] or result["truncated"] or not result["rows"]:
        return None
    values = {row[0] for row in result["rows"]}
    if all(isinstance(value, str) and LOWER_CASE.fullmatch(value) for value in values):
        return sorted(values)
    return None


class CategorySuffix:
    def __init__(self, fallback):
        self.fallback = fallback
        self.probe = None
        self.candidate = None
        self.evidence = None

    def act(self, observation):
        history, remaining = observation["history"], observation["remaining"]
        if self.candidate is not None:
            if passed(history[-1]["observation"]):
                return {"tool": "submit", "sql": self.candidate}
            return self.fallback
        if not remaining:
            return self.fallback
        question = observation["task"]["question"]
        tree, items = category_literals(self.fallback["sql"], question)
        if not items:
            return self.fallback
        tables = next((event["observation"] for event in history if event["tool"] == "schema"), None)
        if tables is None:
            return {"tool": "schema"} if remaining >= 3 else self.fallback
        schema = Schema(tables)
        binding = bound_literals(tree, items, schema)
        if binding is None:
            return self.fallback
        if self.probe is None:
            if remaining < 2:
                return self.fallback
            self.probe = storage_probe(schema, binding)
            prior = next((event for event in history
                          if event["tool"] == "execute" and event["sql"] == self.probe), None)
            if prior is None:
                return {"tool": "execute", "sql": self.probe}
        else:
            prior = history[-1]
        values = stored_values(prior["observation"])
        if values is None:
            return self.fallback
        edits = sorted({(literal.this, literal.this.lower()) for _, literal, _ in items})
        for _, literal, _ in items:
            literal.replace(exp.Literal.string(literal.this.lower()))
        candidate = sql_text(tree)
        if not usable(candidate):
            return self.fallback
        self.evidence = {"field": binding, "observed_values": values, "edits": edits,
                         "candidate": candidate}
        checked = next((event["observation"] for event in history if event["tool"] == "check"
                        and identity(event["sql"]) == identity(candidate)), None)
        if checked is not None:
            if passed(checked):
                return {"tool": "submit", "sql": candidate}
            return self.fallback
        self.candidate = candidate
        return {"tool": "check", "sql": candidate}


class Agent:
    def __init__(self, baseline):
        self.baseline = baseline
        self.suffix = None

    def act(self, observation):
        if self.suffix is None:
            action = self.baseline.act(observation)
            if action["tool"] != "submit":
                return action
            self.suffix = CategorySuffix(action)
        return self.suffix.act(observation)


def create_agent(checkpoint=None):
    return Agent(create_baseline(checkpoint))
