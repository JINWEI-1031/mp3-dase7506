"""A matched comparison of exact time and the named whole-minute bucket."""
import re
from decimal import Decimal

from sqlglot import exp

from step19_grounding import package
from step22_semantics import numeric_literal
from step23_values import generate as exact_pool, layout, observed_edits

# Explicit precision, a different granularity, and a stated comparison take precedence.
PRECISE = re.compile(
    r"\b(?:exact(?:ly)?|precis(?:e|ely)|equal(?:s|led| to)?|seconds?|secs?|"
    r"at least|at most|no (?:more|less) than|(?:less|more|longer|shorter|greater|fewer) than|"
    r"between|under|over|up to|approximately|roughly|about|around|nearest|rounded)\b"
    r"|(?<!\d)\d+:\d{2}(?!\d)",
    re.I,
)


def minute_bucket(question, probe, atom):
    return (probe.requested == ("time", "60") and isinstance(atom, exp.EQ)
            and probe.amount >= 0 and bool(re.fullmatch(r"\+?\d+", atom.expression.this))
            and not PRECISE.search(question))


def widen(predicate, lower, upper):
    result = predicate.copy()
    eq = result if isinstance(result, exp.EQ) else next(result.find_all(exp.EQ), None)
    if eq is None or not isinstance(eq.expression, exp.Literal):
        return None
    column = eq.this
    interval = exp.and_(exp.GTE(this=column.copy(), expression=numeric_literal(lower)),
                        exp.LT(this=column.copy(), expression=numeric_literal(upper)))
    if eq is result:
        return interval
    eq.replace(interval)
    return result


def generate(obs, schema, probe, sample, arm):
    exact = exact_pool(obs, schema, probe, sample)
    question = obs["task"]["question"]
    atom = list(probe.tree.walk())[probe.index]
    if not exact or arm == "exact" or not minute_bucket(question, probe, atom):
        return exact
    shapes = layout(sample)
    raw = []
    for predicate, priority, kind in observed_edits(atom, probe, shapes):
        eq = predicate if isinstance(predicate, exp.EQ) else next(predicate.find_all(exp.EQ), None)
        if eq is None or not isinstance(eq.expression, exp.Literal) or not eq.expression.is_number:
            continue
        lower = Decimal(eq.expression.this)
        # The parsed clock is in seconds. For the scalar candidates, the
        # exact conversion already carries the inferred storage scale.
        width = Decimal(60) if kind == "clock_seconds" else lower / probe.amount if probe.amount else None
        if width is None or width <= 0:
            continue
        bucket = widen(predicate, lower, lower + width)
        if bucket is None:
            continue
        query = probe.tree.copy()
        list(query.walk())[probe.index].replace(exp.Paren(this=bucket))
        raw.append((query, priority, "minute_bucket", {
            "representation": kind, "binding": probe.source[1:],
            "lower": str(lower), "upper_exclusive": str(lower + width)}))
    return package(obs, obs["issue_sql"], raw) or exact
