"""Separate the named ranked population from the final result shape."""
import re

from sqlglot import exp

from repair_rules import parse, sql_text
from step4_agent import identity, passed
from step4_edits import Schema
from step5x_edits import physical_table
from step36.frequency import terms
from step38.event_grain import bind_column


DETAIL = re.compile(r"\bby the ([\w-]+) who (?:has|had)\s+\w+\s+the (fewest|most|least) ([\w -]+)\s*[?.!]?$", re.I)
TOP_SUM = re.compile(r"\btotal\b.+?\bof the ([\w-]+) that (?:have|has) the (top|bottom) (\d{1,3}) ([\w -]+)\s*[?.!]?$", re.I)
SUM_HEAD = re.compile(r"sum\s*\(\s*([a-z_][\w]*)\s*\)", re.I)
NUMERIC = re.compile(r"int|real|decimal|numeric|number|float|double", re.I)


def rank_intent(question):
    match = DETAIL.search(question)
    if match:
        return "detail", match.groups()
    match = TOP_SUM.search(question)
    return ("sum", match.groups()) if match and int(match[3]) else None


def table_named(schema, phrase):
    label = terms(phrase)
    matches = [table for table in schema.tables if label and terms(table) == label]
    return matches[0] if len(matches) == 1 else None


def declared_key(schema, event, root):
    keys = [field for field, attr in schema.columns[root].items() if attr["pk"]]
    if len(keys) != 1:
        return None
    candidates = [fk["from"].casefold() for fk in schema.tables[event]["foreign_keys"]
                  if fk["table"].casefold() == root and fk["to"].casefold() == keys[0]]
    return (candidates[0], keys[0]) if len(candidates) == 1 else None


def detail_plan(observation, parent, schema, role, direction, fact):
    if any(parent.args.get(k) for k in ("where", "with", "limit", "offset")):
        return None
    # The rank subquery may mention the event table again; only resolve the
    # outer projection against the top-level relations.
    outer = parent.copy()
    for slot in ("having", "where", "order"):
        outer.set(slot, None)
    sources = schema.sources(outer)
    ranked_root, event = table_named(schema, role), table_named(schema, fact)
    headings = [h.casefold() for h in observation["example"]["columns"]]
    outputs = [bind_column(col.unalias(), schema, sources) for col in parent.expressions]
    if (not ranked_root or not event or ranked_root == event or not outputs or any(o is None for o in outputs)
            or [o[1] for o in outputs] != headings or len({o[0] for o in outputs}) != 1):
        return None
    target = outputs[0][0]
    if target in (ranked_root, event) or not terms(target) & terms(observation["task"]["question"]):
        return None
    rank_key, display_key = declared_key(schema, event, ranked_root), declared_key(schema, event, target)
    if not rank_key or not display_key or not {event, target} <= sources.keys():
        return None
    key = schema.column(event, rank_key[0], "rank_event")
    rank = exp.select(key.copy()).from_(physical_table(schema, event, "rank_event"))
    rank.join(physical_table(schema, ranked_root, "rank_subject"), on=exp.EQ(
        this=key.copy(), expression=schema.column(ranked_root, rank_key[1], "rank_subject")), copy=False)
    rank.group_by(key.copy(), copy=False)
    rank.order_by(exp.Ordered(this=exp.Count(this=exp.Star()), desc=direction.casefold() == "most",
                              nulls_first=False), key.copy(), copy=False).limit(1, copy=False)

    witness = exp.select("1").from_(physical_table(schema, event, "display_event"))
    witness.where(exp.EQ(this=schema.column(event, display_key[0], "display_event"),
                         expression=schema.column(target, display_key[1], "detail_subject")),
                  exp.In(this=schema.column(event, rank_key[0], "display_event"), query=rank.subquery()), copy=False)
    query = exp.select(*(schema.column(target, field, "detail_subject") for _, field in outputs)).from_(
        physical_table(schema, target, "detail_subject"))
    query.where(exp.Exists(this=witness), copy=False)
    order = parent.args.get("order")
    if order:
        for item in order.expressions:
            if not isinstance(item.this, exp.Column) or item.this.name.casefold() not in headings:
                return None
            ordering = item.copy()
            ordering.set("this", schema.column(target, item.this.name.casefold(), "detail_subject"))
            query.order_by(ordering, copy=False)
    return {"sql": sql_text(query), "kind": "rank_then_detail", "rank_root": ranked_root, "fact": event}


def sum_plan(observation, parent, schema, role, direction, size, measure_words):
    heads = observation["example"]["columns"]
    heading = SUM_HEAD.fullmatch(heads[0]) if len(heads) == 1 else None
    root = table_named(schema, role)
    if not heading or not root or len(parent.expressions) != 1 or not isinstance(parent.expressions[0].unalias(), exp.Sum):
        return None
    if any(parent.args.get(k) for k in ("with", "group", "having", "offset")) or not {
            root} <= schema.sources(parent).keys():
        return None
    if not parent.args.get("limit") or parent.args["limit"].expression != exp.Literal.number(int(size)):
        return None
    numeric = {field for field, attr in schema.columns[root].items() if NUMERIC.search(attr["type"])}
    output = heading[1].casefold()
    keys = [field for field, attr in schema.columns[root].items() if attr["pk"]]
    if output not in numeric or len(keys) != 1:
        return None
    words = terms(measure_words)
    measures = [field for field in numeric if terms(field) == words]
    if not measures and "share" in words and "rate" in terms(output) and words - {"share"}:
        if words - {"share"} == terms(output) - {"rate"}:
            measures = [output]
    if len(measures) != 1:
        return None
    metric = schema.column(root, measures[0], "rank_subject")
    rank = exp.select(schema.column(root, output, "rank_subject").as_("measure_value")).from_(
        physical_table(schema, root, "rank_subject"))
    rank.where(exp.Not(this=exp.Is(this=metric.copy(), expression=exp.Null())), copy=False)
    rank.order_by(exp.Ordered(this=metric, desc=direction.casefold() == "top", nulls_first=False),
                  schema.column(root, keys[0], "rank_subject"), copy=False).limit(int(size), copy=False)
    query = exp.select(exp.Sum(this=exp.column("measure_value", table="ranked"))).from_(rank.subquery("ranked"))
    return {"sql": sql_text(query), "kind": "rank_then_sum", "rank_root": root, "measure": measures[0], "size": int(size)}


class RankedSuffix:
    def __init__(self, fallback):
        self.fallback = fallback
        self.candidate = None
        self.requested = False

    def act(self, observation):
        history, remaining = observation["history"], observation["remaining"]
        if self.candidate:
            return {"tool": "submit", "sql": self.candidate["sql"]} if passed(history[-1]["observation"]) else self.fallback
        intent = rank_intent(observation["task"]["question"]) if remaining else None
        parent = parse(self.fallback["sql"])
        if not intent or not isinstance(parent, exp.Select):
            return self.fallback
        mode, words = intent
        tables = next((e["observation"] for e in history if e["tool"] == "schema"), None)
        if tables is None:
            if remaining >= 2 and not self.requested and parent.args.get("joins"):
                self.requested = True
                return {"tool": "schema"}
            return self.fallback
        compiler = detail_plan if mode == "detail" else sum_plan
        candidate = compiler(observation, parent, Schema(tables), *words)
        seen = {identity(e["sql"]) for e in history if e["tool"] == "check"}
        if candidate and identity(candidate["sql"]) not in seen | {identity(self.fallback["sql"])}:
            self.candidate = candidate
            return {"tool": "check", "sql": candidate["sql"]}
        return self.fallback
