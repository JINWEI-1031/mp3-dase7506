"""Replay the grounded ordering rule and compare its counting grain."""
import re

from sqlglot import exp

from mp3_sql.sqlite_tool import validate_sql
from repair_rules import parse, sql_text
from step4_agent import identity
from step7_semantics import terms
from step12_absence import matching_words
from step17_conflicts import FREQUENCY, alias_tables, conflicts


def event_key(tree, question):
    """Find an occurrence key already equated by the query, without querying schema."""
    aliases = alias_tables(tree)
    candidates = {}
    question_words = terms(question)
    for join in tree.args.get("joins") or []:
        on = join.args.get("on")
        if not isinstance(on, exp.EQ) or not all(isinstance(c, exp.Column) for c in (on.this, on.expression)):
            continue
        a, b = on.this, on.expression
        if a.name.lower() != b.name.lower() or not re.search(r"(?:_id|id)$", a.name, re.I):
            continue
        words = terms(re.sub(r"(?:_id|id)$", "", a.name, flags=re.I))
        matches = matching_words(question_words, words)
        if not words or not matches:
            continue
        for c in (a, b):
            table = aliases.get(c.table.lower())
            if table and words <= terms(table):
                candidates.setdefault(a.name.lower(), (c, bool(matches - words)))
    if len(candidates) == 1:
        return next(iter(candidates.values()))[0]
    inflected = [column for column, verb in candidates.values() if verb]
    return inflected[0] if len(inflected) == 1 else None


def pool(obs, fallback, schema, extended):
    result = conflicts(obs, fallback, ["ordering"], schema)
    if not extended or not FREQUENCY.search(obs["task"]["question"]):
        return result
    tree = parse(fallback)
    key = event_key(tree, obs["task"]["question"]) if isinstance(tree, exp.Select) else None
    if key is None:
        return result
    seen = {identity(e["sql"]) for e in obs["history"] if "sql" in e} | {identity(fallback)}
    seen.update(identity(c["sql"]) for c in result)
    for candidate in result.copy():
        compiled = parse(candidate["sql"])
        if not isinstance(compiled, exp.Select):
            continue
        order = compiled.args.get("order")
        if not order or len(order.expressions) != 1:
            continue
        count = order.expressions[0].this
        if not isinstance(count, exp.Count) or not isinstance(count.this, exp.Star):
            continue
        count.set("this", exp.Distinct(expressions=[key.copy()]))
        sql = sql_text(compiled)
        if identity(sql) in seen:
            continue
        try:
            validate_sql(sql)
        except ValueError:
            continue
        result.append({"sql": sql, "family": "event-count", "priority": candidate["priority"] - 1,
                       "conflict": "an observed event key offers an alternative counting grain",
                       "event_key": key.sql()})
        seen.add(identity(sql))
    return result
