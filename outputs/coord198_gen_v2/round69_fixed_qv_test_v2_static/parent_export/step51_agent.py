"""Compile a named event count before using the single available check."""
from dataclasses import dataclass
import re

from sqlglot import exp

from repair_rules import parse, sql_text
from step4_agent import identity, passed
from step4_edits import Schema
from step5x_edits import physical_table
from step36.frequency import terms
from step48_agent import create_agent as create_baseline


NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
LIMITS = {"at least": exp.GTE, "at most": exp.LTE, "more than": exp.GT,
          "less than": exp.LT, "fewer than": exp.LT, "exactly": exp.EQ}
THRESHOLD = re.compile(r"\b(at least|at most|more than|less than|fewer than|exactly)\s+"
                       r"(\d{1,6}|one|two|three|four|five)\s+([a-z][\w-]*)\b", re.I)
ROLE = {"started": "start", "start": "start", "departed": "start", "ended": "end", "end": "end", "arrived": "end"}
COMPLEX = re.compile(r"\b(?:either|both|without|not|distinct|different|except|each|per|before|after|top|most|fewest|average)\b", re.I)


@dataclass(frozen=True)
class Intent:
    operator: str
    number: int
    event: str
    direction: str | None


def intent_for(question):
    matches = list(THRESHOLD.finditer(question))
    if len(matches) != 1 or COMPLEX.search(question):
        return None
    match = matches[0]
    following = re.match(r"\s+([a-z]+)\b", question[match.end():], re.I)
    direction = ROLE.get(following[1].casefold()) if following else None
    raw = match[2].casefold()
    number = int(raw) if raw.isdecimal() else NUMBERS[raw]
    operator = match[1].casefold()
    if number == 0 or operator not in {"at least", "more than", "exactly"}:
        return None
    return Intent(operator, number, match[3], direction)


def stated_order(question, before, query, schema, root, output):
    """Carry a simple order over an output name when explicitly requested."""
    order = before.args.get("order") if isinstance(before, exp.Select) else None
    ordered = re.search(r"\border\b", question, re.I)
    if not ordered:
        return not order
    if (not order or len(order.expressions) != 1 or not isinstance(order.expressions[0].this, exp.Column)
            or order.expressions[0].this.name.casefold() != output):
        return False
    sort = order.expressions[0].copy()
    sort.set("this", schema.column(root, output, "subject"))
    query.order_by(sort, copy=False)
    return True


def group_count(query, group_column, intent):
    query.group_by(group_column, copy=False)
    predicate = LIMITS[intent.operator](this=exp.Count(this=exp.Star()), expression=exp.Literal.number(intent.number))
    return query.having(predicate, copy=False)


def event_side(observation, intent, schema, original, heads):
    if not intent.direction or not isinstance(original, exp.Select) or len(heads) != 2:
        return None
    if any(not head.startswith(intent.direction + "_") for head in heads):
        return None
    if any(original.args.get(k) for k in ("limit", "offset", "with", "distinct", "having")):
        return None
    where = original.args.get("where")
    if where:
        predicate = where.this.unnest()
        if (not isinstance(predicate, LIMITS[intent.operator]) or not isinstance(predicate.this, exp.Column)
                or predicate.expression != exp.Literal.number(intent.number)):
            return None
    labels = [head for head in heads if head.endswith("_name")]
    keys = [head for head in heads if head.endswith("_id")]
    if len(labels) != 1 or len(keys) != 1 or labels[0][:-5] != keys[0][:-3]:
        return None
    question = observation["task"]["question"]
    if not terms(labels[0][len(intent.direction) + 1:-5]) & terms(question) or re.search(r"\border\b", question, re.I):
        return None
    visible = schema.sources(original)
    candidates = [table for table in visible if terms(table) == terms(intent.event) and set(heads) <= schema.columns[table].keys()]
    if len(candidates) != 1:
        return None
    event = candidates[0]
    query = exp.select(*(schema.column(event, field, "event") for field in heads)).from_(physical_table(schema, event, "event"))
    return {"sql": sql_text(group_count(query, schema.column(event, labels[0], "event"), intent)),
            "kind": "event_side_threshold", "event": event, "group": labels[0]}


def association_side(observation, intent, schema, original, fallback, heads):
    if intent.direction or not isinstance(original, exp.Select) or len(heads) != 1:
        return None
    if any(original.args.get(k) for k in ("where", "limit", "offset", "with", "distinct")):
        return None
    visible = schema.sources(original)
    heading = heads[0]
    if len(visible) != 2 or not heading.endswith("name"):
        return None
    question = observation["task"]["question"]
    roots = [table for table in visible if heading in schema.columns[table] and terms(table) & terms(question)]
    events = [table for table in schema.tables if terms(table) == terms(intent.event)]
    if len(roots) != 1 or len(events) != 1 or roots[0] == events[0]:
        return None
    root, event = roots[0], events[0]
    carrier = next(table for table in visible if table != root)
    if carrier == event:
        return None
    keys = [fk for fk in schema.tables[carrier]["foreign_keys"] if fk["table"].casefold() == root]
    if (len(keys) != 1 or sum(fk["table"].casefold() == event for fk in schema.tables[carrier]["foreign_keys"]) != 1
            or not schema.columns[root][keys[0]["to"].casefold()]["pk"]):
        return None
    fk = keys[0]
    label = schema.column(root, heading, "subject")
    query = exp.select(label.copy()).from_(physical_table(schema, root, "subject"))
    query.join(physical_table(schema, carrier, "event"), on=exp.EQ(
        this=schema.column(root, fk["to"].casefold(), "subject"),
        expression=schema.column(carrier, fk["from"].casefold(), "event")), copy=False)
    if not stated_order(question, fallback, query, schema, root, heading):
        return None
    return {"sql": sql_text(group_count(query, label, intent)), "kind": "association_threshold", "event": carrier, "group": heading}


def propose(observation, intent, fallback):
    tables = next((e["observation"] for e in observation["history"] if e["tool"] == "schema"), None)
    if tables is None:
        return None
    schema = Schema(tables)
    original = parse(observation["issue_sql"])
    heads = [item.casefold() for item in observation["example"]["columns"]]
    return (event_side(observation, intent, schema, original, heads) or
            association_side(observation, intent, schema, original, parse(fallback), heads))


class Agent:
    def __init__(self, baseline):
        self.baseline = baseline
        self.candidate = None
        self.mode = None
        self.displaced = None
        self.fallback = None

    def act(self, observation):
        history, remaining = observation["history"], observation["remaining"]
        if self.candidate is not None:
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
        if intent is None:
            return action
        fallback = action["sql"] if action["tool"] == "submit" else observation["issue_sql"]
        candidate = propose(observation, intent, fallback)
        previous = {identity(e["sql"]) for e in history if e["tool"] == "check"}
        if not candidate or identity(candidate["sql"]) in previous | {identity(fallback)}:
            return action
        if action["tool"] == "check":
            if candidate["kind"] != "event_side_threshold" or any(e["tool"] == "check" and passed(e["observation"]) for e in history):
                return action
            scheduled = parse(action["sql"])
            if scheduled is None or any(t.name.casefold() == candidate["event"] for t in scheduled.find_all(exp.Table)):
                return action
            self.displaced = action["sql"]
            self.mode = "replace"
        else:
            self.fallback = action
            self.mode = "append"
        self.candidate = candidate
        return {"tool": "check", "sql": candidate["sql"]}


def create_agent(checkpoint=None):
    return Agent(create_baseline(checkpoint))
