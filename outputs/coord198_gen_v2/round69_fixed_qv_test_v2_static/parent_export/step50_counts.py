"""Propose counts with a named population or an explicit event role."""
import re

from sqlglot import exp

from repair_rules import parse, sql_text
from step4_agent import identity, passed
from step4_edits import Schema
from step5x_edits import physical_table
from step36.frequency import COUNT_HEADER, terms
from step38.event_grain import bind_column, join_graph


COMMON = re.compile(r"\bmost common\b.*?\bacross all\s+(?:the\s+)?([\w -]+)\s*[?.!]?\s*$", re.I)
WINS = re.compile(r"\b(?:number of times|how many times)\b.+?\b(?:won|wins?)\s+(?:the\s+)?([\w -]+)\s*[?.!]?\s*$", re.I)
WIN_KEY = re.compile(r"(?:^|_)(?:winning|winner|won)(?:_|$)", re.I)


def intent_for(observation, tree):
    if (not isinstance(tree, exp.Select) or not tree.args.get("group") or not tree.find(exp.Count)
            or any(tree.args.get(k) for k in ("distinct", "where", "having", "with", "offset"))):
        return None
    question = observation["task"]["question"]
    heads = observation["example"]["columns"]
    common = COMMON.search(question)
    if (common and len(heads) == len(tree.expressions) == 1 and tree.args.get("order")
            and tree.args.get("limit") and tree.args["limit"].expression == exp.Literal.number(1)):
        return "named_population", common[1]
    wins = WINS.search(question)
    if (wins and sum(bool(COUNT_HEADER.match(h)) for h in heads) == 1 and len(heads) >= 2
            and (not tree.args.get("limit")) and not tree.args.get("order")):
        return "named_winning_role", wins[1]
    return None


def output_binding(tree, schema, sources):
    projected = [bind_column(e.unalias(), schema, sources) for e in tree.expressions if not e.find(exp.Count)]
    if not projected or any(p is None for p in projected) or len({t for t, _ in projected}) != 1:
        return None
    return projected[0][0], {field for _, field in projected}


def source_plan(observation, tree, intent, tables):
    schema = Schema(tables)
    sources = schema.sources(tree)
    output = output_binding(tree, schema, sources)
    if output is None:
        return None
    root, fields = output
    mode, named_event = intent
    label = terms(named_event)
    events = [t for t in schema.tables if t != root and label and terms(t) == label]
    if len(events) != 1 or not terms(root) & terms(observation["task"]["question"]):
        return None
    event = events[0]
    heads = [h.casefold() for h in observation["example"]["columns"]]
    if mode == "named_population":
        if (len(sources) != 2 or event in sources or heads[0] not in fields or not join_graph(tree, schema, sources)
                or not terms(heads[0]) & terms(observation["task"]["question"])):
            return None
        carrier = next(t for t in sources if t != root)
        candidates = [fk for fk in schema.tables[carrier]["foreign_keys"] if fk["table"].casefold() == event]
        if len(candidates) != 1:
            return None
        fk = candidates[0]
        query = tree.copy()
        query.join(physical_table(schema, event, "population"), on=exp.EQ(
            this=schema.column(carrier, fk["from"].casefold(), sources[carrier]),
            expression=schema.column(event, fk["to"].casefold(), "population")), copy=False)
        query.group_by(schema.column(root, heads[0], sources[root]), append=False, copy=False)
        query.order_by(exp.Ordered(this=exp.Count(this=exp.Star()), desc=True, nulls_first=False), append=False, copy=False)
    else:
        candidates = [fk for fk in schema.tables[event]["foreign_keys"]
                      if fk["table"].casefold() == root and WIN_KEY.search(fk["from"])
                      and schema.columns[root][fk["to"].casefold()]["pk"]]
        if len(candidates) != 1 or any(h not in fields and not COUNT_HEADER.match(h) for h in heads):
            return None
        if len([c for c in schema.columns[root].values() if c["pk"]]) != 1:
            return None
        fk = candidates[0]
        columns = [exp.Count(this=exp.Star()) if COUNT_HEADER.match(h) else schema.column(root, h, "subject") for h in heads]
        key = schema.column(event, fk["from"].casefold(), "event")
        query = exp.select(*columns).from_(physical_table(schema, root, "subject"))
        query.join(physical_table(schema, event, "event"), on=exp.EQ(
            this=schema.column(root, fk["to"].casefold(), "subject"), expression=key.copy()), copy=False)
        query.group_by(key, copy=False)
    return {"sql": sql_text(query), "kind": mode, "root": root, "event": event}


class CountSuffix:
    def __init__(self, fallback):
        self.fallback = fallback
        self.candidate = None
        self.requested = False

    def act(self, observation):
        history, remaining = observation["history"], observation["remaining"]
        if self.candidate is not None:
            return {"tool": "submit", "sql": self.candidate["sql"]} if passed(history[-1]["observation"]) else self.fallback
        tree = parse(self.fallback["sql"])
        intent = intent_for(observation, tree) if remaining else None
        if intent is None:
            return self.fallback
        tables = next((e["observation"] for e in history if e["tool"] == "schema"), None)
        if tables is None:
            if remaining >= 2 and not self.requested:
                self.requested = True
                return {"tool": "schema"}
            return self.fallback
        candidate = source_plan(observation, tree, intent, tables)
        previous = {identity(e["sql"]) for e in history if e["tool"] == "check"}
        if candidate and identity(candidate["sql"]) not in previous | {identity(self.fallback["sql"])}:
            self.candidate = candidate
            return {"tool": "check", "sql": candidate["sql"]}
        return self.fallback
