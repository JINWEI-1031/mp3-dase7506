"""Generate only edits grounded in a concrete question/SQL conflict."""
import re

from sqlglot import exp

from mp3_sql.sqlite_tool import validate_sql
from repair_rules import parse, sql_text
from step4_agent import identity
from step4_edits import THRESHOLD
from step7_semantics import terms
from step8_groups import alias_free
from step10_bindings import public_role_conflict

FREQUENCY = re.compile(r"\bmost (?:common|popular|frequent)\b", re.I)
QUALIFIED = re.compile(r"""\b(?:whose|who|where|with|during|between|after|before|including|excluding)\b|["']""", re.I)
TOTAL = re.compile(r"\btotal\b", re.I)
MEASURE = {"amount", "stock", "quantity", "cost", "price", "payment", "sale", "value", "balance"}
SHAPE = ("with", "qualify", "offset")


def simple_root(tree):
    source = tree.args.get("from")
    return bool(source and isinstance(source.this, exp.Table) and not any(tree.args.get(k) for k in SHAPE))


def frequency_edit(tree, question):
    if (not FREQUENCY.search(question) or QUALIFIED.search(question) or not simple_root(tree)
            or len(tree.selects) != 1 or tree.args.get("joins") or tree.args.get("having")):
        return
    label = tree.selects[0].unalias()
    order = tree.args.get("order")
    if not isinstance(label, exp.Column) or (order is not None and order.find(exp.Count)):
        return
    query = exp.select(label.copy()).from_(tree.args["from"].this.copy()).group_by(label.copy())
    query.order_by(exp.Ordered(this=exp.Count(this=exp.Star()), desc=True, nulls_first=False), copy=False)
    query.limit(1, copy=False)
    yield query, 90, "frequency requested; ordered query lacks a frequency count"


def measure_field(column, question):
    words = terms(column.name)
    asked = terms(question)
    payment = "paid" in asked and bool(words & {"paid", "amount", "payment"})
    return bool(words & MEASURE & asked or ("stock" in words and "stock" in asked) or payment)


def threshold_measure(tree, question):
    where = tree.args.get("where")
    if not where:
        return None
    expression = where.this
    if (not isinstance(expression, (exp.GT, exp.GTE, exp.LT, exp.LTE, exp.EQ))
            or not isinstance(expression.this, exp.Column)
            or not isinstance(expression.expression, exp.Literal)
            or not measure_field(expression.this, question)):
        return None
    values = {value for _, value in THRESHOLD.findall(question)}
    values.update(re.findall(r"\b(?:bigger|greater|more|less|smaller) than (\d+(?:\.\d+)?)\b",
                             question, flags=re.I))
    return expression if expression.expression.this in values else None


def total_edit(tree, question):
    if (not TOTAL.search(question) or not simple_root(tree)
            or not all(isinstance(p.unalias(), exp.Column) for p in tree.selects)
            or len(list(tree.find_all(exp.Select))) != 1 or tree.args.get("having")):
        return
    new = alias_free(tree)
    leaves = [p.unalias() for p in new.selects]
    measures = [p for p in leaves if measure_field(p, question)]
    order = new.args.get("order")
    ranks = [o.this for o in order.expressions if isinstance(o.this, exp.Column)
             and measure_field(o.this, question)] if order else []
    threshold = threshold_measure(new, question)
    metric = threshold.this if threshold is not None else measures[0] if len(measures) == 1 else None
    entity = [p for p in leaves if p != metric]
    has_each = bool(re.search(r"\b(?:each|per)\b", question, re.I))
    max_total = bool(re.search(r"\b(?:max(?:imum)?|highest|largest|greatest)\s+total\b", question, re.I))
    group = new.args.get("group")
    if (metric is None or not entity
            or not (threshold is not None or has_each and group or max_total and ranks)
            or any(p != metric for p in [*measures, *ranks])):
        return
    summed = exp.Sum(this=metric.copy())
    projected = [summed.copy() if p.unalias() == metric else p.copy() for p in new.selects]
    new.select(*projected, append=False, copy=False)
    new.group_by(*(p.copy() for p in entity), append=False, copy=False)
    if threshold is not None:
        clause = threshold.copy()
        clause.set("this", summed.copy())
        new.set("where", None)
        new.having(clause, copy=False)
    if order:
        for ordered in order.expressions:
            if ordered.this == metric:
                ordered.set("this", summed.copy())
    if max_total:
        new.limit(1, copy=False)
    yield new, 100, "requested per-entity total; raw measure is used as projection, filter or rank"
    # "More than" is the preferred strict reading, but a frozen-pool control
    # can include the inclusive boundary to measure ambiguity at the same slot.
    if threshold is not None and isinstance(threshold, (exp.GT, exp.LT)):
        inclusive = exp.GTE if isinstance(threshold, exp.GT) else exp.LTE
        variant = new.having(inclusive(this=summed.copy(), expression=threshold.expression.copy()), append=False)
        yield variant, 75, "aggregate total with alternate boundary"


def quantifier_edits(tree, question):
    temporal = re.search(r"\b(after|before)\s+(?:the (?:date|time) of )?any\b", question, re.I)
    universal = re.search(r"\b(?:greater|higher|less|lower) than\b.{0,60}\bof all\b", question, re.I)
    if not (temporal or universal):
        return
    for index, predicate in enumerate(tree.walk()):
        if not isinstance(predicate, (exp.GT, exp.LT)) or not isinstance(predicate.this, exp.Column):
            continue
        subquery = predicate.expression
        if not isinstance(subquery, exp.Subquery) or not isinstance(subquery.this, exp.Select):
            continue
        inner = subquery.this
        if (len(inner.selects) != 1 or not simple_root(inner) or inner.args.get("group")
                or inner.args.get("limit") or inner.args.get("order")):
            continue
        outer_name = predicate.this.name.lower()
        argument = inner.selects[0].unalias()
        if temporal:
            if (not terms(outer_name) & {"date", "time", "year"} or
                    isinstance(predicate, exp.GT) != (temporal[1].lower() == "after")):
                continue
            desired = exp.Min if isinstance(predicate, exp.GT) else exp.Max
        else:
            desired = exp.Max if isinstance(predicate, exp.GT) else exp.Min
        if isinstance(argument, desired):
            continue
        field = argument.this if isinstance(argument, (exp.Min, exp.Max)) else argument
        if not isinstance(field, exp.Column) or field.name.lower() != outer_name:
            continue
        copy = tree.copy()
        selected = list(copy.walk())[index].expression.this
        selected.select(desired(this=field.copy()), append=False, copy=False)
        yield copy, 95, "comparison quantifier requires the opposite extreme or explicit aggregation"


def distinct_edit(tree, question):
    if (re.search(r"\bdistinct\b", question, re.I) and not tree.args.get("distinct")
            and simple_root(tree) and not any(tree.args.get(k) for k in ("limit", "group", "having"))
            and all(isinstance(p.unalias(), exp.Column) for p in tree.selects)):
        yield tree.distinct(), 80, "distinct output requested; row projection preserves duplicates"


LOCAL = {"frequency": frequency_edit, "total": total_edit, "quantifier": quantifier_edits,
         "distinct": distinct_edit}


def conflicts(obs, fallback, families, schema=None):
    question = obs["task"]["question"]
    tree = parse(fallback)
    seen = {identity(e["sql"]) for e in obs["history"] if "sql" in e} | {identity(fallback)}
    result = []
    if isinstance(tree, exp.Select):
        for family in families:
            if family not in LOCAL:
                continue
            for query, rank, reason in LOCAL[family](tree, question):
                result.append({"sql": sql_text(query), "family": family, "priority": rank, "conflict": reason})
    if "role" in families and schema is not None:
        item = public_role_conflict(obs, schema, fallback)
        if item:
            result.append(item)
    pool = []
    for item in result:
        key = identity(item["sql"])
        if key in seen:
            continue
        try:
            validate_sql(item["sql"])
        except ValueError:
            continue
        seen.add(key)
        pool.append(item)
    return pool
