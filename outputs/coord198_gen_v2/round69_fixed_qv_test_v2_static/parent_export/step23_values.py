"""Observe only the ambiguous predicate field, then compile a bounded type repair."""
from dataclasses import dataclass
from decimal import Decimal
import re

from sqlglot import exp

from repair_rules import parse, sql_text
from step5x_edits import physical_table
from step7_semantics import top_tables
from step10_bindings import existing, field_words
from step11_absence import parent_sqls
from step19_grounding import package
from step22_semantics import (
    NUMBER, TEXT, TIME_FIELDS, alternatives, asked_unit, clock_forms,
    clock_seconds, numeric_literal, storage_unit, unit,
)

COMPARISONS = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)
CLOCK = re.compile(r"\d+:\d{2}(?::\d{2})?(?:\.\d+)?$")
SUFFIX = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*([A-Za-z]+)$")
PREFIX = re.compile(r"([A-Za-z][A-Za-z _-]*?)(\s+)([+-]?\d+(?:\.\d+)?)$")


@dataclass
class Probe:
    tree: exp.Select
    index: int
    source: tuple
    amount: Decimal
    requested: tuple | None
    sample_sql: str


def spec(tree, schema, question, headings):
    if not isinstance(tree, exp.Select) or len(tree.selects) != len(headings):
        return
    for i, atom in enumerate(tree.walk()):
        if not isinstance(atom, COMPARISONS) or not isinstance(atom.this, exp.Column):
            continue
        if atom.find_ancestor(exp.Where) is None or not isinstance(atom.expression, exp.Literal):
            continue
        value = atom.expression.this
        if not NUMBER.fullmatch(value):
            continue
        scope = atom.find_ancestor(exp.Select)
        binding = existing(atom.this, top_tables(scope, schema), schema) if scope else None
        if not binding:
            continue
        _, table, field = binding
        affinity = schema.columns[table][field]["type"]
        roles = set(field_words(field))
        request = asked_unit(question, value)
        ambiguous_time = (request and request[0] == "time" and roles & TIME_FIELDS
                          and not storage_unit(field, request))
        explicit_code = (not request and TEXT.match(affinity) and roles and re.search(
            rf"\b(?:{'|'.join(re.escape(w) for w in sorted(roles))})"
            rf"\b(?:\s+[a-z]+){{0,3}}\s+{re.escape(value)}(?![\w.])", question, re.I))
        if not ambiguous_time and not explicit_code:
            continue
        col = schema.column(table, field, "v")
        sample = exp.select(col.copy().as_("observed_value")).from_(physical_table(schema, table, "v"))
        sample.distinct(copy=False).where(exp.Not(this=exp.Is(this=col, expression=exp.Null())),
                                         copy=False).limit(6, copy=False)
        yield (3 if ambiguous_time else 2), Probe(tree, i, binding, Decimal(value), request, sql_text(sample))


def choose(obs, pending, schema, candidates=()):
    question, headings = obs["task"]["question"], obs["example"]["columns"]
    sqls = [pending] if pending else []
    sqls.extend(c["sql"] for c in candidates[:10])
    sqls.extend(parent_sqls(obs, obs["issue_sql"]))
    options = [p for sql in dict.fromkeys(sqls) for p in spec(parse(sql), schema, question, headings)]
    return max(options, key=lambda p: p[0])[1] if options else None


def layout(feedback):
    values = {"clock": [], "numeric": [], "prefix": [], "suffix": [], "other": []}
    if not feedback or not feedback["ok"]:
        return values
    for row in feedback["rows"]:
        if len(row) != 1 or row[0] is None:
            continue
        text = str(row[0]).strip()
        if CLOCK.fullmatch(text):
            values["clock"].append(text)
        elif NUMBER.fullmatch(text):
            values["numeric"].append(text)
        elif PREFIX.fullmatch(text):
            values["prefix"].append(text)
        elif SUFFIX.fullmatch(text):
            values["suffix"].append(text)
        else:
            values["other"].append(text)
    return values


def compare(atom, column, value):
    result = atom.copy()
    result.set("this", column)
    result.set("expression", numeric_literal(value))
    return result


def text_numeric(column):
    # Guard the cast so a categorical string cannot collapse to zero.
    source = column.sql(dialect="sqlite")
    trimmed = f"trim({source})"
    tail = f"CASE WHEN substr({trimmed},1,1) IN ('+','-') THEN substr({trimmed},2) ELSE {trimmed} END"
    return parse(f"""SELECT CASE WHEN ({tail}) GLOB '[0-9]*' AND ({tail}) NOT GLOB '*[^0-9.]*'
                    AND length({tail}) - length(replace({tail}, '.', '')) <= 1
                    THEN CAST({trimmed} AS REAL) END""").selects[0]


def observed_edits(atom, probe, shapes):
    column = atom.this
    value = probe.amount
    if probe.requested:
        seconds = value * Decimal(probe.requested[1])
        if shapes["clock"]:
            yield compare(atom, clock_seconds(column.sql(dialect="sqlite")), seconds), 160, "clock_seconds"
        numeric = [Decimal(v) for v in shapes["numeric"]]
        if numeric:
            scales = (Decimal(probe.requested[1]), Decimal(1), Decimal(".001"), Decimal(60), Decimal(3600))
            def proximity(scale):
                target = seconds / scale
                return (0 if target in numeric else 1,
                        min(abs(v-target) / max(abs(v), abs(target), Decimal(1)) for v in numeric),
                        scales.index(scale))
            for scale in sorted(set(scales), key=proximity)[:3]:
                yield compare(atom, text_numeric(column), seconds / scale), 155, "numeric_units"
        suffixes = {SUFFIX.fullmatch(v)[2] for v in shapes["suffix"]}
        units = {unit(s) for s in suffixes}
        if len(units) == 1 and None not in units and next(iter(units))[0] == "time":
            stored = next(iter(units))
            # Each observed suffix is bounded and alphabetic, and SQL literals are AST-escaped.
            for suffix in sorted(suffixes)[:2]:
                cleaned = exp.func("TRIM", exp.func("SUBSTR", column.copy(), 1,
                                   exp.func("LENGTH", column.copy()) - exp.Literal.number(len(suffix))))
                predicate = compare(atom, text_numeric(cleaned), seconds / Decimal(stored[1]))
                predicate = exp.and_(exp.func("LOWER", column.copy()).like(
                    exp.Literal.string("%" + suffix.lower())), predicate)
                yield predicate, 152, "explicit_text_units"
    elif shapes["numeric"]:
        projected = text_numeric(column)
        if isinstance(atom, exp.NEQ):
            # Preserve categorical values for inequalities against a numeric code.
            projected = exp.Coalesce(this=projected, expressions=[column.copy()])
        yield compare(atom, projected, value), 152, "numeric_text"


def prefixed_label(atom, probe, shapes):
    values = shapes["prefix"]
    if probe.requested or len(values) < 2 or any(shapes[k] for k in ("clock", "numeric", "suffix", "other")):
        return
    parsed = [(value, PREFIX.fullmatch(value)) for value in values]
    styles = {(match[1].lower(), match[2]) for _, match in parsed}
    if len(styles) != 1:
        return
    prefix, gap = next(iter(styles))
    if not set(field_words(prefix)) & set(field_words(probe.source[2])):
        return
    observed = [value for value, match in parsed if Decimal(match[3]) == probe.amount]
    if len(observed) == 1 and isinstance(atom, (exp.EQ, exp.NEQ)):
        predicate = atom.copy()
        predicate.set("expression", exp.Literal.string(observed[0]))
        yield predicate, 165, "observed_prefixed_label"
    if not isinstance(atom, (exp.EQ, exp.NEQ)):
        tail = exp.func("SUBSTR", atom.this.copy(), exp.Literal.number(len(prefix + gap) + 1))
        prefix_matches = exp.func("LOWER", atom.this.copy()).like(exp.Literal.string(prefix + gap + "%"))
        comparison = compare(atom, text_numeric(tail), probe.amount)
        yield exp.and_(prefix_matches, comparison), 151, "prefixed_numeric_comparison"


def generate(obs, schema, probe, observation=None):
    shapes = layout(observation)
    question = obs["task"]["question"]
    atom = list(probe.tree.walk())[probe.index]
    raw = []
    changes = [*observed_edits(atom, probe, shapes), *prefixed_label(atom, probe, shapes)]
    if not changes:
        changes = [(node, score, evidence["kind"]) for node, score, evidence in
                   alternatives(atom, schema, probe.source, question)]
    for corrected, rank, kind in changes:
        query = probe.tree.copy()
        list(query.walk())[probe.index].replace(corrected)
        raw.append((query, rank, "value_format", {"kind": kind, "binding": probe.source[1:],
                                                   "formats": {k: len(v) for k, v in shapes.items()}}))
    return package(obs, obs["issue_sql"], raw)
