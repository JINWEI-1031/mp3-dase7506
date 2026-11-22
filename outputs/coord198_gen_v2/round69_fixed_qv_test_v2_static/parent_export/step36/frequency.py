"""Build frequency plans from question roles and the observed key graph."""
from dataclasses import dataclass
import re

from sqlglot import exp

from step4_edits import Schema
from step5x_edits import physical_table
from step7_semantics import join_path, shortest
from step26_relations import edge_is_forward
from step32_plan import tree_plan

RANK = re.compile(
    r"\b(?:most|least)\s+(?:common|frequent(?:ly)?)\b|"
    r"\b(?:greatest|highest|largest|lowest|most|fewest|least)\s+(?:(?:number|count)\s+of|\w+\s+count)\b|"
    r"\b(?:most|fewest)\s+(?:kinds?|types?)\s+of\b|"
    r"\b(?:most|fewest)\s+(?!(?:recently|recent|expensive|popular|likely|valuable)\b)[a-z]+s\b",
    re.I,
)
PER = re.compile(r"\b(?:for each|per)\b", re.I)
HOW_MANY = re.compile(r"\bhow many\b", re.I)
THRESHOLD = re.compile(r"\b(?:at least|at most|more than|less than|over|under)\b", re.I)
AMBIGUOUS = re.compile(r"\b(?:not|never|without|both|either|top|after|before|with status)\b", re.I)
DISTINCT = re.compile(r"\b(?:different|distinct|kinds?|types?)\s+of\b|\bdistinct\b", re.I)
COUNT_HEADER = re.compile(r"^count\s*\(", re.I)
GENERIC = {"the", "of", "a", "an", "and", "in", "for", "who", "that", "which", "with", "what",
           "how", "name", "id", "code", "number", "count", "show", "list", "each", "per",
           "tell", "give", "return", "most", "fewest", "least", "common", "frequently",
           "time", "kind", "type", "greatest", "highest", "largest", "lowest", "all", "total"}


@dataclass(frozen=True)
class Intent:
    mode: str
    subject: frozenset
    counted: frozenset
    distinct: bool
    descending: bool


def terms(text):
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    result = set()
    for word in re.findall(r"[a-z]+", text.lower()):
        if word.endswith("ies") and len(word) > 4:
            word = word[:-3] + "y"
        elif word.endswith(("sses", "ches", "shes", "xes")):
            word = word[:-2]
        elif word.endswith("s") and not word.endswith(("ss", "us")) and len(word) > 3:
            word = word[:-1]
        elif word.endswith("ed") and len(word) > 5:
            word = word[:-2]
        result.add(word)
    return result - GENERIC


def infer(question, headings):
    if AMBIGUOUS.search(question) or THRESHOLD.search(question) or not 1 <= len(headings) <= 4:
        return None
    counts = sum(bool(COUNT_HEADER.match(h)) for h in headings)
    grouped = PER.search(question)
    counting = HOW_MANY.search(question)
    if grouped and counting and counting.end() < grouped.start() and counts == 1 and len(headings) == 2:
        return Intent("count per label", frozenset(terms(question[grouped.end():])),
                      frozenset(terms(question[counting.end():grouped.start()])), False, True)
    rank = RANK.search(question)
    if not rank or counts > 1 or counts == len(headings):
        return None
    # Preserve a noun inside the ranking phrase ("most settlements"), plus
    # an event verb or relation named elsewhere in the request.
    subject = frozenset(terms(question[:rank.start()]))
    counted = frozenset(terms(question[rank.start():]) | (terms(question) - subject))
    if not counted or re.search(r"\b(?:common|frequently|ordered)\b", question, re.I):
        counted |= frozenset(terms(question))
    return Intent("rank by frequency", subject, counted, bool(DISTINCT.search(question)),
                  not bool(re.search(r"\b(?:least|fewest|lowest)\b", question, re.I)))


def heading_fields(headings, schema, intent):
    labels = [h.casefold() for h in headings if not COUNT_HEADER.match(h)]
    for table, fields in schema.columns.items():
        if not all(h in fields for h in labels):
            continue
        cue = terms(table) & intent.subject
        heading_cue = set().union(*(terms(h) for h in labels)) & intent.subject
        if cue or heading_cue:
            yield (3 * len(cue) + len(heading_cue), table, labels)


def event_paths(schema, root, intent):
    options = []
    wanted = intent.counted - terms(root)
    for event in schema.tables:
        if event == root:
            continue
        names = terms(event) - terms(root)
        # An event and its type dimension can share the same role word.
        if not names and re.search(r"(?:^|_)(?:types?|categories|classes|status)(?:_|$)", root):
            names = terms(event)
        matched = names & intent.counted if not (terms(event) - terms(root)) else names & wanted
        if not matched:
            continue
        for path in shortest(schema, root, event):
            if not path or len(path) > 3 or all(edge_is_forward(schema, edge) for edge in path):
                continue
            # Favor roles named in the question, and prefer the closest
            # relation carrying those events over a more distant dimension.
            rank = (len(matched), not bool(names - wanted), -len(path))
            options.append((rank, event, path, sorted(matched)))
    return sorted(options, key=lambda item: (tuple(-int(v) for v in item[0]), item[1], item[2]))


def measure_for(schema, event, path, aliases, distinct):
    if not distinct:
        return exp.Count(this=exp.Star())
    if edge_is_forward(schema, path[-1]):
        left, key, _, _ = path[-1]
        value = schema.column(left, key, aliases[left])
    else:
        keys = [key for key, meta in schema.columns[event].items() if meta["pk"]]
        if len(keys) != 1:
            return None
        value = schema.column(event, keys[0], aliases[event])
    return exp.Count(this=exp.Distinct(expressions=[value]))


def question_plans(question, headings, tables):
    intent = infer(question, headings)
    if not intent:
        return
    schema = Schema(tables)
    roots = sorted(heading_fields(headings, schema, intent), key=lambda item: (-item[0], item[1]))
    for root_score, root, fields in roots[:2]:
        for path_score, event, path, matched in event_paths(schema, root, intent)[:3]:
            aliases = {root: "r0"}
            query = exp.select().from_(physical_table(schema, root, "r0"))
            join_path(query, path, schema, aliases)
            measure = measure_for(schema, event, path, aliases, intent.distinct)
            if measure is None:
                continue
            labels = iter(fields)
            output = [measure.copy() if COUNT_HEADER.match(h) else schema.column(root, next(labels), "r0")
                      for h in headings]
            query.select(*output, copy=False)
            if intent.mode == "count per label":
                groups = fields
            else:
                groups = [name for name, meta in schema.columns[root].items() if meta["pk"]]
                if not groups:
                    continue
            query.group_by(*(schema.column(root, field, "r0") for field in groups), copy=False)
            if intent.mode == "rank by frequency":
                query.order_by(exp.Ordered(this=measure, desc=intent.descending,
                                           nulls_first=not intent.descending), copy=False).limit(1, copy=False)
            yield tree_plan(query, {"kind": "question_schema_frequency", "mode": intent.mode,
                                   "entity": root, "event": event, "event_terms": matched,
                                   "grain": groups, "measure": measure.sql(dialect="sqlite"),
                                   "path": path, "root_score": root_score, "path_score": path_score})


def order_count(parent):
    order = parent.args.get("order")
    if not order:
        return None
    count = order.find(exp.Count)
    if count:
        return count
    first = order.expressions[0].this
    return next((item.find(exp.Count) for item in parent.expressions
                 if item.alias and isinstance(first, exp.Column)
                 and first.name.casefold() == item.alias.casefold()), None)


def needs_observation(intent, parent, public):
    if not intent:
        return False
    if not public:
        return True
    if not isinstance(parent, exp.Select) or any(parent.args.get(k) for k in ("where", "having", "distinct")):
        return False
    if intent.mode == "count per label":
        return True
    count = order_count(parent)
    if count is None:
        return True
    joins = parent.args.get("joins", [])
    if not joins:
        return False
    group = parent.args.get("group")
    key_grain = group and any(re.search(r"(?:^|_)(?:id|code|key|num)$", col.name, re.I)
                              for col in group.find_all(exp.Column))
    return bool(len(joins) > 1 or not key_grain or intent.distinct and not count.find(exp.Distinct))


def selection_evidence(parent, c, candidate, public):
    if not public:
        return "explicit frequency and bound schema roles"
    if not isinstance(parent, exp.Select) or any(parent.args.get(k) for k in ("where", "having", "distinct")):
        return None
    output = [col.unalias() for col in parent.expressions if not col.find(exp.AggFunc)]
    if not output or any(not isinstance(col, exp.Column) for col in output):
        return None
    score = candidate["basis"]
    if score["mode"] == "count per label":
        before = [col.alias_or_name.casefold() for col in parent.expressions]
        if before[0] != c["columns"][0].casefold() and COUNT_HEADER.match(c["columns"][0]):
            return "requested count-first projection"
        return None
    limit = parent.args.get("limit")
    if limit and not (isinstance(limit.expression, exp.Literal) and limit.expression.this == "1"):
        return None
    old_order = parent.args.get("order")
    count = order_count(parent)
    if not count and (not old_order or not old_order.find(exp.AggFunc)):
        return "requested event frequency rather than existing sort expression"
    old_group = parent.args.get("group")
    names = {col.name.casefold() for col in old_group.find_all(exp.Column)} if old_group else set()
    if names and not set(score["grain"]) <= names:
        return "entity key instead of displayed grouping field"
    if "DISTINCT" in score["measure"] and (count is None or not count.find(exp.Distinct)):
        return "explicit distinct event kinds"
    old_tables = {t.name.casefold() for t in parent.find_all(exp.Table)}
    new_tables = {score["entity"], *(edge[2] for edge in score["path"])}
    if count is not None and new_tables < old_tables:
        return "count only the named event path"
    return None


