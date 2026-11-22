"""Type, unit, and relationship-scope edits supported by the visible question and schema."""
from decimal import Decimal
import re

from sqlglot import exp

from repair_rules import parse
from step4_edits import Schema
from step5x_edits import physical_table
from step7_semantics import join_path, top_tables
from step8_groups import MEASURE
from step10_bindings import ATOMS, existing, field_words, tokens
from step11_absence import parent_sqls
from step19_grounding import package
from step20_paths import output_options, projection, relation_plans
from step21_carry import bind_clause, extract

UNITS = {
    "second": ("time", "1"), "sec": ("time", "1"),
    "millisecond": ("time", ".001"), "ms": ("time", ".001"),
    "minute": ("time", "60"), "min": ("time", "60"),
    "hour": ("time", "3600"), "hr": ("time", "3600"),
    "day": ("time", "86400"),
    "kilometer": ("distance", "1000"), "km": ("distance", "1000"),
    "meter": ("distance", "1"), "centimeter": ("distance", ".01"),
    "foot": ("distance", ".3048"), "mile": ("distance", "1609.344"),
    "kilogram": ("mass", "1000"), "kg": ("mass", "1000"),
    "gram": ("mass", "1"), "pound": ("mass", "453.59237"),
    "thousand": ("scale", "1000"), "million": ("scale", "1000000"),
    "billion": ("scale", "1000000000"),
}
TIME_FIELDS = {"duration", "runtime", "elapsed", "length"}
NUMBER = re.compile(r"[+-]?\d+(?:\.\d+)?$")
TEXT = re.compile(r"(?:n?varchar|n?char|text|clob|time)", re.I)
NUMERIC = re.compile(r"(?:int|real|dec|num|float|double)", re.I)
ABSENCE = re.compile(r"\b(?:without|never|no associated|(?:do|does|did) not|don['’]t|doesn['’]t|didn['’]t)\b", re.I)
EXISTENCE = re.compile(r"\b(?:who|with (?:an?|any|at least one)|that (?:have|has|had))\b", re.I)


def unit(word):
    return UNITS.get(word.lower()) or UNITS.get(word.lower().removesuffix("s"))


def asked_unit(question, literal):
    value = rf"(?<![\w.]){re.escape(literal)}(?![\w.])"
    matches = {u for m in re.finditer(value + r"\s*[- ]?\s*([a-z]+)", question, re.I)
               if (u := unit(m[1]))}
    return next(iter(matches)) if len(matches) == 1 else None


def storage_unit(field, requested):
    units = {u for word in field_words(field) if (u := unit(word)) and u[0] == requested[0]}
    return next(iter(units)) if len(units) == 1 else None


def numeric_literal(number):
    return exp.Literal.number(format(number.normalize(), "f"))


def clock_forms(seconds):
    if seconds != int(seconds) or not 0 <= seconds < 86400:
        return []
    value = int(seconds)
    hours, remainder = divmod(value, 3600)
    minutes, seconds = divmod(remainder, 60)
    return list(dict.fromkeys([f"{hours:02d}:{minutes:02d}:{seconds:02d}",
                               f"{hours}:{minutes:02d}:{seconds:02d}",
                               f"{value // 60}:{seconds:02d}", f"{value // 60:02d}:{seconds:02d}"]))


def clock_seconds(column):
    # Parse M:SS and H:MM:SS without SQLite time-of-day coercion.
    first = f"instr({column}, ':')"
    tail = f"substr({column}, {first} + 1)"
    second = f"instr({tail}, ':')"
    left = f"CAST(substr({column}, 1, {first} - 1) AS INTEGER)"
    return parse(f"""SELECT CASE WHEN {first} > 0 AND {second} > 0
                  THEN {left} * 3600 + CAST(substr({tail}, 1, {second} - 1) AS INTEGER) * 60
                  + CAST(substr({tail}, {second} + 1) AS REAL)
                  WHEN {first} > 0 THEN {left} * 60 + CAST({tail} AS REAL) END""").selects[0]


def alternatives(atom, schema, source, question):
    value = atom.expression
    if not isinstance(value, exp.Literal) or not NUMBER.fullmatch(value.this):
        return
    if not re.search(rf"(?<![\w.]){re.escape(value.this)}(?![\w.])", question):
        return
    _, table, field = source
    affinity = schema.columns[table][field]["type"]
    number = Decimal(value.this)
    requested = asked_unit(question, value.this)
    if requested:
        stored = storage_unit(field, requested)
        if stored and requested != stored:
            integer = atom.copy()
            integer.set("expression", numeric_literal(number * Decimal(requested[1]) / Decimal(stored[1])))
            yield integer, 150, {"kind": "declared_units", "question_unit": requested, "field_unit": stored}
        time_field = set(field_words(field)) & TIME_FIELDS
        if time_field and requested[0] == "time" and stored is None:
            seconds = number * Decimal(requested[1])
            if TEXT.match(affinity):
                forms = clock_forms(seconds)
                if isinstance(atom, (exp.EQ, exp.NEQ)) and forms:
                    predicate = exp.In(this=atom.this.copy(), expressions=[exp.Literal.string(v) for v in forms])
                    if isinstance(atom, exp.NEQ):
                        predicate = exp.Not(this=predicate)
                    yield predicate, 139, {"kind": "text_clock_forms", "seconds": str(seconds)}
                elif isinstance(atom, (exp.GT, exp.GTE, exp.LT, exp.LTE)):
                    predicate = atom.copy()
                    predicate.set("this", clock_seconds(atom.this.sql(dialect="sqlite")))
                    predicate.set("expression", numeric_literal(seconds))
                    yield predicate, 137, {"kind": "text_clock_comparison", "seconds": str(seconds)}
            if NUMERIC.match(affinity) and seconds != number:
                predicate = atom.copy()
                predicate.set("expression", numeric_literal(seconds))
                yield predicate, 108, {"kind": "numeric_seconds_probe", "seconds": str(seconds)}
        return
    if NUMERIC.match(affinity) and value.is_string:
        result = atom.copy()
        result.set("expression", numeric_literal(number))
        yield result, 105, {"kind": "numeric_literal", "type": affinity}
    elif TEXT.match(affinity) and not value.is_string and isinstance(atom, (exp.EQ, exp.NEQ)):
        result = atom.copy()
        result.set("expression", exp.Literal.string(value.this))
        yield result, 101, {"kind": "text_literal", "type": affinity}
    if TEXT.match(affinity) and isinstance(atom, (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)):
        column = atom.this.sql(dialect="sqlite")
        trimmed = f"trim({column})"
        unsigned = (f"CASE WHEN substr({trimmed}, 1, 1) IN ('+', '-') "
                    f"THEN substr({trimmed}, 2) ELSE {trimmed} END")
        otherwise = column if isinstance(atom, (exp.EQ, exp.NEQ)) else "NULL"
        guarded = parse(f"""SELECT CASE WHEN ({unsigned}) GLOB '[0-9]*'
                         AND ({unsigned}) NOT GLOB '*[^0-9.]*'
                         AND length({unsigned}) -
                             length(replace({unsigned}, '.', '')) <= 1
                         THEN CAST({column} AS REAL)
                         ELSE {otherwise} END""").selects[0]
        result = atom.copy()
        result.set("this", guarded)
        result.set("expression", numeric_literal(number))
        yield result, 100, {"kind": "numeric_text_comparison", "type": affinity}


def repair_projection(tree, schema, headings, question, required):
    options = output_options(tree, headings, schema, set(tokens(question)))
    sources = top_tables(tree, schema)
    aliases = {t: a for a, t in sources.items()}
    for variant in options:
        wanted = {table for _, table, _ in variant if table}
        if wanted - set(aliases) or (required and not wanted & required):
            continue
        output = projection(tree, headings, variant, schema, aliases)
        if output:
            yield tree.copy().select(*output, append=False), sum(s for s, _, _ in variant)


def typed_edits(tree, schema, headings, question):
    nodes = list(tree.walk())
    for i, atom in enumerate(nodes):
        if (not isinstance(atom, ATOMS) or not isinstance(atom.this, exp.Column)
                or atom.find_ancestor(exp.Where) is None):
            continue
        select = atom.find_ancestor(exp.Select)
        if select is None:
            continue
        source = existing(atom.this, top_tables(select, schema), schema)
        if not source:
            continue
        for corrected, rank, evidence in alternatives(atom, schema, source, question):
            result = tree.copy()
            list(result.walk())[i].replace(corrected)
            if len(result.selects) == len(headings):
                yield result, rank, "typed", {**evidence, "binding": source[1:]}
            for adjusted, merit in repair_projection(result, schema, headings, question, set()):
                yield adjusted, rank + min(merit, 30) // 3, "typed_projection", {
                    **evidence, "binding": source[1:]}


def scopes(tree, schema, headings, question):
    requested_absence = bool(ABSENCE.search(question))
    if not requested_absence and not EXISTENCE.search(question):
        return
    if any(tree.args.get(k) for k in ("group", "having", "limit", "offset", "order")):
        return
    words = tokens(question)
    claims = list(extract(tree, schema, question, words, 0))
    if not claims or len(claims) > 3 or any(c.expression.find(exp.Or) for c in claims):
        return
    targets = {t for c in claims for _, t, _ in c.bindings}
    outputs = output_options(tree, headings, schema, set(words))
    for variant in outputs:
        roots = {t for _, t, _ in variant if t}
        if len(roots) != 1 or roots & targets:
            continue
        root = next(iter(roots))
        for _, path, merit in relation_plans(root, targets, schema, set(words), 2):
            origin, left, start, right = path[0]
            aliases = {root: "r0", start: "r1"}
            sub = exp.select("1").from_(physical_table(schema, start, "r1"))
            join_path(sub, path[1:], schema, aliases)
            conditions = [exp.EQ(this=schema.column(origin, left, aliases[origin]),
                                 expression=schema.column(start, right, aliases[start]))]
            for claim in claims:
                clause = bind_clause(claim, schema, aliases)
                if requested_absence and isinstance(clause, exp.NEQ):
                    clause = exp.EQ(this=clause.this.copy(), expression=clause.expression.copy())
                conditions.append(clause)
            sub.where(exp.and_(*conditions), copy=False)
            query = exp.select().from_(physical_table(schema, root, "r0"))
            columns = projection(query, headings, variant, schema, aliases)
            if not columns:
                continue
            query.select(*columns, append=False, copy=False)
            exists = exp.Exists(this=sub)
            query.where(exp.Not(this=exists) if requested_absence else exists, copy=False)
            if tree.args.get("distinct"):
                query.distinct(copy=False)
            yield query, 125 + merit + sum(c.score for c in claims), "relationship_scope", {
                "foreign_keys": path, "negated": requested_absence,
                "bindings": [b[1:] for c in claims for b in c.bindings]}


def observed_relationships(obs, schema):
    """Extend scope plans with working, same-field equalities already probed."""
    expanded = Schema(list(schema.tables.values()))
    for event in obs["history"]:
        if event["tool"] != "check" or not event["observation"]["ok"]:
            continue
        tree = parse(event["sql"])
        if tree is None:
            continue
        for select in tree.find_all(exp.Select):
            sources = top_tables(select, schema)
            for join in select.args.get("joins") or []:
                on = join.args.get("on")
                if not isinstance(on, exp.EQ) or not all(isinstance(c, exp.Column) for c in (on.this, on.expression)):
                    continue
                left, right = (existing(c, sources, schema) for c in (on.this, on.expression))
                if not left or not right or left[1] == right[1] or left[2] != right[2]:
                    continue
                a, field = left[1:]
                b = right[1]
                if schema.columns[a][field]["type"].lower() != schema.columns[b][field]["type"].lower():
                    continue
                if (a, field, b, field) not in expanded.edges[a]:
                    expanded.edges[a].append((a, field, b, field))
                    expanded.edges[b].append((b, field, a, field))
    return expanded


def candidate_pool(obs, fallback, schema, modes, previous):
    question = obs["task"]["question"]
    headings = obs["example"]["columns"]
    raw = []
    scope_schema = observed_relationships(obs, schema) if "observed" in modes else schema
    parents = list(dict.fromkeys([*parent_sqls(obs, fallback), *(c["sql"] for c in previous[:10])]))
    for sql in parents[:20]:
        tree = parse(sql)
        if not isinstance(tree, exp.Select) or tree.args.get("with"):
            continue
        if "typed" in modes:
            raw.extend(typed_edits(tree, schema, headings, question))
        if "scope" in modes:
            scoped = list(scopes(tree, scope_schema, headings, question))
            raw.extend(scoped)
            if "typed" in modes:
                for variant, _, _, evidence in scoped:
                    for adjusted, rank, _, detail in typed_edits(variant, schema, headings, question):
                        raw.append((adjusted, rank + 5, "typed_scope", {**evidence, **detail}))
    return package(obs, fallback, raw)[:16]


def unit_replacements(obs, pending, schema):
    """Let a grounded conversion compete for the same pending probe."""
    tree = parse(pending)
    headings = obs["example"]["columns"]
    if not isinstance(tree, exp.Select) or len(tree.selects) != len(headings):
        return []
    explicit = {"declared_units", "text_clock_forms", "text_clock_comparison"}
    return package(obs, pending, (
        edit for edit in typed_edits(tree, schema, headings, obs["task"]["question"])
        if edit[2] == "typed" and edit[3]["kind"] in explicit
    ))
