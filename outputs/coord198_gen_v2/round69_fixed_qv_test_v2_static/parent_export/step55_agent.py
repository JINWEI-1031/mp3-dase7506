"""Bind a grouped average to its fact table and preserve the stated boundary."""
import re

from sqlglot import exp

from repair_rules import parse, sql_text
from step4_agent import identity, passed
from step4_edits import Schema
from step5x_edits import physical_table
from step36.frequency import terms
from step38.event_grain import bind_column, join_graph
from step48_agent import create_agent as create_baseline


OPERATORS = {"over": exp.GT, "above": exp.GT, "greater than": exp.GT, "more than": exp.GT,
             "under": exp.LT, "below": exp.LT, "less than": exp.LT, "at least": exp.GTE, "at most": exp.LTE}
AVERAGE = re.compile(r"\baverage\s+([\w -]+?)(?:\s+of)?\s+(" + "|".join(OPERATORS)
                     + r")\s+(-?\d+(?:\.\d+)?)\s*[?.!]?$", re.I)
NUMERIC = re.compile(r"int|real|numeric|decimal|float|double|number", re.I)


def intent_for(question):
    match = AVERAGE.search(question)
    if not match or re.search(r"\b(?:or|except|without|not|both|between|per)\b", question, re.I):
        return None
    return match[1], match[2].casefold(), match[3]


def bind_plan(observation, intent, tables):
    measure_phrase, operator, value = intent
    parent = parse(observation["issue_sql"])
    if (not isinstance(parent, exp.Select) or len(observation["example"]["columns"]) != 1
            or any(parent.args.get(slot) for slot in ("where", "order", "limit", "offset", "with", "distinct"))):
        return None
    group, having = parent.args.get("group"), parent.args.get("having")
    if not group or len(group.expressions) != 1 or not having:
        return None
    predicate = having.this.unnest()
    if (not isinstance(predicate, OPERATORS[operator]) or not isinstance(predicate.this, exp.Avg)
            or not isinstance(predicate.this.this, exp.Column) or predicate.expression != exp.Literal.number(value)):
        return None

    schema = Schema(tables)
    sources = schema.sources(parent)
    label = bind_column(group.expressions[0], schema, sources)
    if not label or len(sources) != 2 or label[1] != observation["example"]["columns"][0].casefold():
        return None
    root, grouping = label
    if not terms(grouping) <= terms(observation["task"]["question"]):
        return None
    if sum(bind_column(p.unalias(), schema, sources) == label for p in parent.expressions) != 1:
        return None
    leaf = next(table for table in sources if table != root)
    question = terms(observation["task"]["question"])
    if (terms(leaf) - terms(root)) & question or not join_graph(parent, schema, sources):
        return None
    root_keys = [name for name, entry in schema.columns[root].items() if entry["pk"]]
    links = [fk for fk in schema.tables[leaf]["foreign_keys"] if fk["table"].casefold() == root]
    if (len(root_keys) != 1 or len(links) != 1 or links[0]["to"].casefold() != root_keys[0]
            or any(col["pk"] and col["name"].casefold() == links[0]["from"].casefold()
                   for col in schema.tables[leaf]["columns"])):
        return None
    metric = predicate.this.this.name.casefold()
    words = terms(measure_phrase)
    fields = [name for name, attr in schema.columns[root].items() if words and words <= terms(name)
              and NUMERIC.search(attr["type"])]
    if fields != [metric] or metric in schema.columns[leaf]:
        return None
    return {"schema": schema, "parent": parent, "root": root, "leaf": leaf, "label": grouping,
            "measure": metric, "operator": operator, "threshold": value}


def compile_average(plan, joined=False, inclusive=False):
    schema, root, label = plan["schema"], plan["root"], plan["label"]
    if joined:
        query = plan["parent"].copy()
        alias = schema.sources(query)[root]
        query.select(schema.column(root, label, alias), append=False, copy=False)
    else:
        alias = "fact"
        query = exp.select(schema.column(root, label, alias)).from_(physical_table(schema, root, alias))
        query.group_by(schema.column(root, label, alias), copy=False)
    comparator = OPERATORS[plan["operator"]]
    if inclusive:
        comparator = {exp.GT: exp.GTE, exp.LT: exp.LTE}.get(comparator, comparator)
    query.having(comparator(this=exp.Avg(this=schema.column(root, plan["measure"], alias)),
                            expression=exp.Literal.number(plan["threshold"])), append=False, copy=False)
    return sql_text(query)


class Agent:
    def __init__(self, baseline):
        self.baseline = baseline
        self.candidate = None
        self.mode = None
        self.fallback = None

    def act(self, observation):
        history, remaining = observation["history"], observation["remaining"]
        if self.candidate:
            if passed(history[-1]["observation"]):
                return {"tool": "submit", "sql": self.candidate["sql"]}
            if self.fallback:
                return self.fallback
            action = self.baseline.act(observation)
            return action if action["tool"] == "submit" else {"tool": "submit", "sql": observation["issue_sql"]}

        action = self.baseline.act(observation)
        if not remaining or (action["tool"] != "submit" and (remaining != 1 or action["tool"] != "check")):
            return action
        intent = intent_for(observation["task"]["question"])
        tables = next((e["observation"] for e in history if e["tool"] == "schema"), None)
        if not intent or tables is None:
            return action
        plan = bind_plan(observation, intent, tables)
        if plan is None:
            return action
        sql = compile_average(plan)
        if identity(sql) in {identity(e["sql"]) for e in history if e["tool"] == "check"} or identity(sql) == identity(action["sql"]):
            return action
        if action["tool"] == "check":
            scheduled = parse(action["sql"])
            having = scheduled.args.get("having") if isinstance(scheduled, exp.Select) else None
            if not having or not having.find(exp.Count) or having.find(exp.Avg) or any(
                    e["tool"] == "check" and passed(e["observation"]) for e in history):
                return action
            self.mode = "replace"
        else:
            self.fallback = action
            self.mode = "append"
        self.candidate = {"sql": sql, "fact": plan["root"], "group": plan["label"],
                          "measure": plan["measure"], "boundary": plan["operator"]}
        return {"tool": "check", "sql": sql}


def create_agent(checkpoint=None):
    return Agent(create_baseline(checkpoint))
