"""Compile grouped repairs using checked SQL and official output headings."""
import re

from sqlglot import exp

from repair_rules import parse, sql_text
from step5x_edits import physical_table
from step7_semantics import bindings, terms, top_tables

HEADING = re.compile(r"[A-Za-z_][A-Za-z0-9_ ]*")
MEASURE = re.compile(r"(count|sum|avg|min|max)\(\s*(distinct\s+)?(\*|[\w.]+)\s*\)", re.I)
FUNCTIONS = {"count": exp.Count, "sum": exp.Sum, "avg": exp.Avg, "min": exp.Min, "max": exp.Max}
DETAIL = re.compile(r"\b(?:for (?:all|each) (?:those|these)|of the \w+ (?:who|which|that)"
                    r"|for each \w+ (?:who|which|that))\b", re.I)


def alias_free(tree):
    """Keep group, HAVING and ORDER valid when a projection alias is removed."""
    aliases = {p.alias.lower(): p.unalias() for p in tree.selects if p.alias}
    clean = tree.copy()
    for name in ("group", "having", "order"):
        clause = clean.args.get(name)
        if clause:
            for col in list(clause.find_all(exp.Column)):
                if not col.table and col.name.lower() in aliases:
                    col.replace(aliases[col.name.lower()].copy())
    return clean


def bind(col, sources, schema, preferred=()):
    options = [(alias, table, name) for table, name in bindings(col, sources, schema)
               for alias, target in sources.items() if target == table]
    if not options:
        return None
    candidates = sorted(options, key=lambda item: (
        item[0] != col.table.lower(), item[0] not in preferred))
    alias, table, name = candidates[0]
    return schema.column(table, name, alias)


def measures(tree):
    return list(tree.find_all(exp.AggFunc))


def output_bindings(tree, schema, headings):
    sources = top_tables(tree, schema)
    if not sources:
        return None
    previous = {p.alias_or_name.lower(): p.unalias() for p in tree.selects if p.alias_or_name}
    preferred = {c.table.lower() for p in tree.selects for c in p.find_all(exp.Column) if c.table}
    output = []
    for heading in headings:
        heading = heading.strip().lower()
        match = MEASURE.fullmatch(heading)
        if match:
            function, distinct, field = match.groups()
            token = field.split(".")
            argument = exp.Star() if field == "*" else bind(
                exp.column(token[-1], table=token[0] if len(token) == 2 else None),
                sources, schema, preferred)
            if argument is None:
                return None
            if distinct:
                argument = exp.Distinct(expressions=[argument])
            output.append(FUNCTIONS[function](this=argument))
        elif heading in previous:
            expr = previous[heading]
            if isinstance(expr, exp.Column):
                expr = bind(expr, sources, schema, preferred)
            if expr is None:
                return None
            output.append(expr.copy())
        elif HEADING.fullmatch(heading):
            value = bind(exp.column(heading.replace(" ", "_")), sources, schema, preferred)
            if value is None:
                return None
            output.append(value)
        else:
            return None
    return output


def role_overlap(query, question, schema):
    """Favor the FK endpoint matching an explicit direction in the question."""
    asked = set()
    if re.search(r"\b(?:start|depart|origin|source)\w*\b", question):
        asked |= {"start", "src", "source", "origin", "departure"}
    if re.search(r"\b(?:arriv|end|destination)\w*\b", question):
        asked |= {"end", "dst", "destination", "arrival"}
    if not asked:
        return 0
    joins = query.args.get("joins") or []
    return max((len(terms(c.name) & asked) for j in joins
                for c in j.find_all(exp.Column)), default=0)


def group_labels(query, output):
    """Replace a coarse or over-specific grouping with the requested labels."""
    originals = query.args.get("group")
    labels = [c.copy() for c in output if isinstance(c, exp.Column)]
    if originals and labels and originals.expressions != labels:
        return query.group_by(*labels, append=False)
    return None


def plain_group(query):
    return query.args.get("group") and not any(query.args.get(k) for k in (
        "with", "distinct", "offset", "qualify")) and len(list(query.find_all(exp.Select))) == 1


def local_candidates(tree, schema, headings, question):
    if not plain_group(tree):
        return
    output = output_bindings(tree, schema, headings)
    if not output:
        return
    base = alias_free(tree)
    sql = base.select(*output, append=False)
    yield "ordered_projection", sql, 8
    grain = group_labels(sql, output)
    if grain is not None:
        yield "output_grain", grain, 4
    if DETAIL.search(question) and all(isinstance(c, exp.Column) for c in output):
        keys = base.args["group"].expressions
        if len(keys) == 1 and isinstance(keys[0], exp.Column) and (
                base.args.get("having") or base.args.get("limit") and base.args.get("order")):
            key = keys[0]
            group = base.select(key.copy(), append=False)
            outer = sql.copy()
            for clause in ("group", "having", "order", "limit", "where"):
                outer.set(clause, None)
            outer.where(exp.In(this=key.copy(), query=exp.Subquery(this=group)), copy=False)
            yield "group_filter_then_details", outer, 13


def one_relation(tree, schema, headings):
    """Move all output and grouping roles onto a single fact relation."""
    if not plain_group(tree) or not tree.args.get("joins") or tree.args.get("where"):
        return
    sources = top_tables(tree, schema)
    for table in dict.fromkeys(sources.values()):
        if not any(h.lower() in schema.columns[table] for h in headings):
            continue
        if any(isinstance(m, exp.Count) and isinstance(m.this, exp.Distinct) for m in measures(tree)):
            continue
        candidate = alias_free(tree)
        candidate.set("from", exp.From(this=physical_table(schema, table, "g")))
        candidate.set("joins", None)
        for aggregate in measures(candidate):
            if isinstance(aggregate, exp.Count):
                aggregate.set("this", exp.Star())
        output = output_bindings(candidate, schema, headings)
        if not output:
            continue
        parts = [candidate.args.get(key) for key in ("having", "order")]
        if any(c.name.lower() not in schema.columns[table] for p in parts if p
               for c in p.find_all(exp.Column)):
            continue
        for p in parts:
            if p:
                for c in p.find_all(exp.Column):
                    c.replace(schema.column(table, c.name.lower(), "g"))
        columns = []
        for expr in output:
            for c in list(expr.find_all(exp.Column)):
                if c.name.lower() not in schema.columns[table]:
                    break
                c.set("table", exp.to_identifier("g"))
            else:
                columns.append(expr)
        labels = [c.copy() for c in columns if isinstance(c, exp.Column)]
        if len(columns) != len(headings) or not labels:
            continue
        rebuilt = candidate.select(*columns, append=False).group_by(*labels, append=False)
        yield "fact_relation_grain", rebuilt, 6
        by_name = [c for c in labels if terms(c.name) & {"name", "label", "title"}]
        if len(by_name) == 1 and len(labels) > 1:
            yield "fact_label_grain", rebuilt.group_by(*by_name, append=False), 5


def compile_pool(obs, schema, seen, families):
    question = obs["task"]["question"].lower()
    headings = obs["example"]["columns"]
    target_rows = obs["example"]["row_count"]
    parents = {}
    for event in obs["history"]:
        if event["tool"] == "check":
            parents[event["sql"]] = event["observation"]
    ranked = []
    for parent, feedback in parents.items():
        tree = parse(parent)
        if not isinstance(tree, exp.Select):
            continue
        generators = [local_candidates(tree, schema, headings, question)]
        if "fact" in families:
            generators.append(one_relation(tree, schema, headings))
        for candidates in generators:
            for rule, query, prior in candidates:
                if rule == "group_filter_then_details" and "detail" not in families:
                    continue
                score = prior + 4 * role_overlap(query, question, schema)
                if feedback["ok"]:
                    score += 4
                    if feedback["row_count"] == target_rows:
                        score += 3
                if rule == "ordered_projection" and len(headings) != len(tree.selects):
                    score += 2
                sql = sql_text(query)
                ranked.append((score, {"sql": sql, "rule": rule, "parent": parent, "priority": score}))
    pool = []
    for _, candidate in sorted(ranked, key=lambda item: -item[0]):
        sql = candidate["sql"]
        if sql in seen:
            continue
        seen.add(sql)
        pool.append(candidate)
        if len(pool) == 8:
            break
    return pool
