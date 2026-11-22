"""Rank existing fact candidates and compile relation-backed grouped repairs."""
import re

from sqlglot import exp

from repair_rules import parse, sql_text
from step5x_edits import physical_table
from step7_semantics import THRESHOLD, terms, top_tables
from step8_groups import alias_free, bind, output_bindings, plain_group


def fact_priority(candidate, question):
    if candidate["rule"] not in {"fact_relation_grain", "fact_label_grain"}:
        return 0
    tree = parse(candidate["sql"])
    having = tree.args.get("having")
    if not having:
        return 0
    if re.search(r"\b(?:average|mean)\b", question) and having.find(exp.Avg):
        return 2
    if THRESHOLD.search(question) and having.find(exp.Count):
        heading_words = set().union(*(terms(c.name) for c in tree.selects if isinstance(c, exp.Column)))
        role = ({"start", "origin", "source"} if re.search(r"\bstart\w*\b", question) else
                {"end", "destination", "arrival"} if re.search(r"\b(?:end|arriv)\w*\b", question) else set())
        if role & heading_words and len(tree.selects) > 1:
            return 1
    return 0


def quoted_role_filter(tree, schema, headings, question):
    """Keep a grounded role and the string type explicitly supplied in the question."""
    if not plain_group(tree) or not tree.args.get("where") or not tree.args.get("order"):
        return
    quoted = re.findall(r"""(?:"([^"]+)"|'([^']+)')""", question)
    values = {a or b for a, b in quoted}
    if not values:
        return
    out = output_bindings(tree, schema, headings)
    if not out:
        return
    for pred in tree.args["where"].find_all(exp.EQ):
        col, literal = pred.this, pred.expression
        if not isinstance(col, exp.Column) or not isinstance(literal, exp.Literal):
            continue
        if literal.is_string or literal.this not in values or not any(
                len(w) >= 4 and re.search(r"\b" + re.escape(w) + r"\w*\b", question.lower())
                for w in terms(col.name)):
            continue
        proposed = alias_free(tree).select(*out, append=False)
        for node in proposed.args["where"].find_all(exp.Literal):
            if node == literal:
                node.replace(exp.Literal.string(literal.this))
        yield "typed_role_filter", proposed, 100
        labels = [p for p in out if isinstance(p, exp.Column)]
        if labels:
            yield "typed_role_with_label_grain", proposed.group_by(*labels, append=False), 99


def count_fact_join(tree, schema, headings, question):
    if not plain_group(tree) or not re.search(r"\b(?:maximum|minimum|most|fewest) (?:number|count)\b", question):
        return
    order = tree.args.get("order")
    if tree.args.get("where") or tree.args.get("having") or not order:
        return
    group = tree.args["group"].expressions
    if len(group) != 1 or not isinstance(group[0], exp.Column):
        return
    sources = top_tables(tree, schema)
    bound = bind(group[0], sources, schema)
    if bound is None:
        return
    entity = sources[bound.table]
    key = bound.name.lower()
    anchors = terms(entity)
    qwords = terms(question)
    for _, a, fact, b in sorted(set(schema.edges[entity])):
        if a != key or not anchors & terms(fact) or len(schema.edges[fact]) < 2:
            continue
        extras = set().union(*(terms(n) for n in schema.columns[fact])) - anchors
        relevance = len(extras & qwords)
        ekey = schema.column(entity, a, "e")
        fkey = schema.column(fact, b, "f")
        select = exp.select().from_(physical_table(schema, entity, "e"))
        select.join(physical_table(schema, fact, "f"), on=exp.EQ(this=ekey, expression=fkey), copy=False)
        out = output_bindings(select, schema, headings)
        if out is None:
            continue
        desc = not bool(re.search(r"\b(?:minimum|fewest)\b", question))
        query = select.select(*out, append=False).group_by(ekey.copy())
        query.order_by(exp.Ordered(this=exp.Count(this=exp.Star()), desc=desc, nulls_first=not desc), copy=False)
        query.limit(1, copy=False)
        yield "count_bound_relationship", query, 80 + relevance


def compact_detail(tree, schema, headings, question):
    if not re.search(r"\bof the \w+ (?:who|which|that)\b", question):
        return
    where = tree.args.get("where")
    if not where or not isinstance(where.this, (exp.In, exp.EQ)):
        return
    key = where.this.this
    sub = where.this.args.get("query") if isinstance(where.this, exp.In) else where.this.expression
    if not isinstance(key, exp.Column) or not isinstance(sub, exp.Subquery):
        return
    inner = sub.this
    if not isinstance(inner, exp.Select) or not plain_group(inner) or not (
            inner.args.get("having") or inner.args.get("order") and inner.args.get("limit")):
        return
    outer = top_tables(tree, schema)
    inner_sources = top_tables(inner, schema)
    bound = bind(key, outer, schema)
    if not bound or len(outer) != 1 or len(inner_sources) != 1:
        return
    output = output_bindings(tree, schema, headings)
    if output:
        grouped = inner.copy()
        result = tree.select(*output, append=False)
        result.where(exp.In(this=bound, query=exp.Subquery(this=grouped)), append=False, copy=False)
        yield "ranked_relation_then_details", result, 60


def candidates(obs, schema, seen):
    headings = obs["example"]["columns"]
    question = obs["task"]["question"]
    ranked = []
    for event in obs["history"]:
        if event["tool"] != "check" or not event["observation"]["ok"]:
            continue
        tree = parse(event["sql"])
        if not isinstance(tree, exp.Select):
            continue
        for generate in (quoted_role_filter, count_fact_join, compact_detail):
            for name, query, priority in generate(tree, schema, headings, question):
                ranked.append((priority, name, sql_text(query), event["sql"]))
    result = []
    for priority, rule, sql, parent in sorted(ranked, key=lambda x: -x[0]):
        if sql not in seen:
            seen.add(sql)
            result.append({"rule": rule, "sql": sql, "parent": parent, "priority": priority})
    return result[:6]


def preaggregate(sql, schema, headings):
    """Count fact rows by their FK before joining or grouping dimension labels."""
    tree = parse(sql)
    if not isinstance(tree, exp.Select) or not plain_group(tree):
        return None
    tree = alias_free(tree)
    joins = tree.args.get("joins") or []
    if len(joins) != 1 or joins[0].side or joins[0].kind not in ("", "INNER"):
        return None
    if tree.args.get("where") or tree.args.get("having"):
        return None
    equality = joins[0].args.get("on")
    if not isinstance(equality, exp.EQ) or not all(isinstance(c, exp.Column) for c in (
            equality.this, equality.expression)):
        return None
    sources = top_tables(tree, schema)
    keys = [bind(c, sources, schema) for c in (equality.this, equality.expression)]
    if len(sources) != 2 or not all(keys) or keys[0].table == keys[1].table:
        return None
    for dim, fact in (keys, keys[::-1]):
        table = sources[dim.table]
        primary = [c for c, meta in schema.columns[table].items() if meta["pk"]]
        if primary == [dim.name.lower()]:
            break
    else:
        return None
    dim_table, fact_table = sources[dim.table], sources[fact.table]
    edge = (fact_table, fact.name.lower(), dim_table, dim.name.lower())
    if edge not in schema.edges[fact_table]:
        return None
    projections = output_bindings(tree, schema, headings)
    if not projections:
        return None
    group = tree.args["group"].expressions
    if len(group) != 1 or not isinstance(group[0], exp.Column):
        return None
    grain = bind(group[0], sources, schema)
    if grain is None or (grain.table != dim.table and grain != fact):
        return None
    merge_labels = grain not in (dim, fact)
    order = tree.args.get("order")
    if order and (len(order.expressions) != 1 or not is_count(order.expressions[0].this)):
        return None
    if not any(is_count(p) for p in projections) and order is None:
        return None
    fact_key = schema.column(fact_table, fact.name.lower(), "f")
    dimension_key = schema.column(dim_table, dim.name.lower(), "d")
    counted = exp.select(fact_key.as_("k"), exp.Count(this=exp.Star()).as_("n"))
    counted.from_(physical_table(schema, fact_table, "f"), copy=False)
    counted.group_by(fact_key.copy(), copy=False)
    count = exp.column("n", table="counts")
    aggregate = exp.Sum(this=count) if merge_labels else count
    new = []
    for projection in projections:
        if is_count(projection):
            new.append(aggregate.copy())
        elif isinstance(projection, exp.Column) and projection in (dim, fact):
            new.append(exp.column("k", table="counts"))
        elif isinstance(projection, exp.Column) and projection.table == dim.table:
            new.append(schema.column(dim_table, projection.name.lower(), "d"))
        else:
            return None
    result = exp.select(*new).from_(counted.subquery("counts"))
    result.join(physical_table(schema, dim_table, "d"),
                on=exp.EQ(this=dimension_key, expression=exp.column("k", table="counts")), copy=False)
    if merge_labels:
        if any(isinstance(c, exp.Column) and c.name.lower() != grain.name.lower() for c in new):
            return None
        result.group_by(schema.column(dim_table, grain.name.lower(), "d"), copy=False)
    if order:
        key = order.expressions[0].copy()
        key.set("this", aggregate.copy())
        result.order_by(key, copy=False)
    if tree.args.get("limit"):
        result.set("limit", tree.args["limit"].copy())
    return {"rule": "count_fact_before_dimension_join", "sql": sql_text(result),
            "parent": sql, "priority": 100}


def is_count(expression):
    return isinstance(expression, exp.Count) and isinstance(expression.this, exp.Star)
