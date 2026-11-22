"""Direct relationship plans for explicit association qualification."""
from dataclasses import dataclass
import re

from sqlglot import exp

from step4_edits import Schema
from step5x_edits import physical_table
from step7_semantics import join_path, shortest
from step26_relations import edge_is_forward
from step32_plan import tree_plan
from step36.frequency import terms

RELATION = re.compile(r"\b(?:who|whose|that|which)\s+(?:have|has|had|made|made an?|teach|teaches|are trained in|are in)\b"
                      r"|\b(?:who|that)\s+(?:made|teach(?:es)?)\b", re.I)
DIRECT = re.compile(r"^\s*(?:which|what)\s+([\w -]+?)\s+(?:have|has|had|made|are trained in|are in)\b", re.I)
NUMBER = re.compile(r"\b(\d+)\s*[?.!]?$")
UNBOUND_QUANTIFIER = re.compile(r"\b(?:more than|fewer than|less than|at least|at most|exactly)\b", re.I)
PROJECTION = re.compile(r"\b(?:names?|ids?|addresses|email|phone)\b", re.I)
EXCLUDE = re.compile(r"\b(?:both|either|or|without|don't|do not|not|no|most|least|fewest|highest|lowest|"
                     r"at least|at most|how many|count|each|every|ordered|ascending|descending)\b", re.I)
QUOTE = re.compile(r'''"([^"]+)"|'([^']+)' ''', re.X)
MONTH = re.compile(r"\bin (January|February|March|April|May|June|July|August|September|October|November|December)\s*[?.!]?$", re.I)
COMPARISON = re.compile(r"\b(more expensive than|greater than|more than|less than|above|below)\s+(\d+(?:\.\d+)?)\b", re.I)


@dataclass(frozen=True)
class Membership:
    subject: frozenset
    event: frozenset
    phrase: str
    value: str | None
    operator: str | None


def infer(question, headings):
    explicit_outputs = PROJECTION.search(question) or (
        re.match(r"^\s*(?:which|what)\b", question, re.I)
        and all(re.fullmatch(r"[\w ]+", heading) for heading in headings))
    if not 1 <= len(headings) <= 3 or not explicit_outputs or EXCLUDE.search(question):
        return None
    cue = RELATION.search(question)
    direct = DIRECT.match(question) if cue is None else None
    if cue:
        head, tail = question[:cue.start()], question[cue.end():]
    elif direct:
        head, tail = direct[1], question[direct.end():]
    else:
        return None
    tail = tail.split("?", 1)[0]
    if re.search(r"\bbetween\b", tail, re.I) or len(re.findall(r"\b\d+(?:\.\d+)?\b", tail)) > 1:
        return None
    subject = terms(head)
    if not subject:
        return None
    quotes = list(QUOTE.finditer(tail))
    match = COMPARISON.search(tail)
    if len(quotes) > 1 or (quotes and match):
        return None
    if quotes:
        token = quotes[0]
        value = token[1] or token[2]
        event = terms(tail[:token.start()])
        return Membership(frozenset(subject), frozenset(event), tail[:token.start()], value, "eq")
    if match:
        event = terms(tail[:match.start()])
        return Membership(frozenset(subject), frozenset(event), match[1].lower(), match[2],
                          "lt" if match[1].lower() in ("less than", "below") else "gt")
    if UNBOUND_QUANTIFIER.search(tail):
        return None
    number = NUMBER.search(tail)
    if number:
        phrase = tail[:number.start()]
        fields = terms(phrase)
        if fields:
            return Membership(frozenset(subject), frozenset(fields), phrase, number[1], "number_eq")
    month = MONTH.search(tail)
    if month:
        return Membership(frozenset(subject), frozenset(terms(tail[:month.start()])), "month", month[1], "eq")
    if re.search(r"\d|\b(?:with|in|by|from)\b", tail, re.I):
        # A fragment with additional qualification needs a bound field.
        return None
    event = terms(tail)
    return Membership(frozenset(subject), frozenset(event), tail, None, None) if event else None


def filters(schema, target, intent):
    if intent.operator is None:
        return [(None, None)]
    words = terms(intent.phrase)
    if "expensive" in words:
        words |= {"cost", "price"}
    fields = []
    for col, meta in schema.columns[target].items():
        matched = terms(col) & words
        if not matched:
            continue
        numeric = bool(re.search(r"int|real|float|numeric|decimal|double", meta["type"], re.I))
        if intent.operator != "number_eq" and (intent.operator in ("lt", "gt")) != numeric:
            continue
        literal = exp.Literal.number(intent.value) if numeric else exp.Literal.string(intent.value)
        fields.append((-len(matched), col, literal))
    return [(col, literal) for _, col, literal in sorted(fields)[:2]]


def plans(question, headings, tables):
    intent = infer(question, headings)
    if not intent:
        return
    schema = Schema(tables)
    projected = [h.casefold() for h in headings]
    bindings = []
    for root, fields in schema.columns.items():
        affinity = terms(root) & intent.subject
        if not affinity or any(col not in fields for col in projected):
            continue
        for event in schema.tables:
            if event == root:
                continue
            words = terms(event) - terms(root)
            match = words & intent.event
            predicates = filters(schema, event, intent)
            if not match and not (intent.operator == "number_eq" and predicates):
                continue
            for path in shortest(schema, root, event):
                if not path or all(edge_is_forward(schema, edge) for edge in path):
                    continue
                for field, literal in predicates:
                    score = (-len(affinity), -len(match), bool(words - intent.event), len(path))
                    bindings.append((score, root, event, path, field, literal))
    for _, root, event, path, field, literal in sorted(bindings, key=lambda b: (b[0], b[1], b[2], b[3], b[4] or ""))[:3]:
        for mode in ("entity exists", "joined rows"):
            query = exp.select(*(schema.column(root, col, "r0") for col in projected)).from_(
                physical_table(schema, root, "r0"))
            aliases = {root: "r0"}
            scoped = query
            if mode == "entity exists":
                left, root_key, child, child_key = path[0]
                aliases[child] = "r1"
                scoped = exp.select("1").from_(physical_table(schema, child, "r1"))
                scoped.where(exp.EQ(this=schema.column(left, root_key, "r0"),
                                    expression=schema.column(child, child_key, "r1")), copy=False)
                join_path(scoped, path[1:], schema, aliases)
            else:
                join_path(scoped, path, schema, aliases)
            if field:
                op = {"eq": exp.EQ, "number_eq": exp.EQ, "lt": exp.LT, "gt": exp.GT}[intent.operator]
                scoped.where(op(this=schema.column(event, field, aliases[event]), expression=literal.copy()), copy=False)
            if mode == "entity exists":
                query.where(exp.Exists(this=scoped), copy=False)
            yield tree_plan(query, {"kind": "question_schema_membership", "mode": mode, "root": root,
                                   "event": event, "path": path, "field": field, "value": intent.value})




