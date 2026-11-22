"""Role-aware evidence for the historical candidate pool."""
from collections import Counter
import re

from sqlglot import exp

from repair_rules import parse, sql_text
from step38.event_grain import bind_column
from step39.agent import COMPARISONS, conjuncts, plain_select
from step40.witness import words

EVENT = re.compile(r"\b(?:trained in|made|attended|performed|completed|placed)\b", re.I)


def indicated_role(literal, field, schema, question):
    mention = re.search(rf"(?<!\w){re.escape(literal)}(?!\w)", question, re.I)
    if mention is None:
        return None
    before = question[:mention.start()]
    cues = [(m.end(), t) for t in schema.tables if field in schema.columns[t]
            for word in words(t) for m in re.finditer(rf"\b{re.escape(word)}s?\b", before, re.I)]
    return max(cues)[1] if cues else None


def evidence(sql, schema, question):
    tree = parse(sql)
    scopes = list(tree.find_all(exp.Select)) if isinstance(tree, exp.Select) else []
    if not scopes:
        return None
    sources = schema.sources(tree)
    if not sources:
        return None
    predicates = []
    for index, scope in enumerate(scopes):
        where = scope.args.get("where")
        having = scope.args.get("having")
        clauses = [("where", where.this if where else None), ("having", having.this if having else None)]
        clauses.extend(("join", j.args.get("on")) for j in scope.args.get("joins", []))
        for clause, expression in clauses:
            for node in conjuncts(expression):
                left, right = node.this, node.expression
                owner = bind_column(left, schema, sources)
                other = bind_column(right, schema, sources)
                rhs = right if isinstance(right, exp.Literal) else None
                value = rhs.this if rhs is not None else None
                physical, field = owner if owner else (None, None)
                expected = indicated_role(value, field, schema, question) if value is not None and owner else None
                predicates.append({"table": physical, "role": expected or physical,
                                   "scope": "outer" if index == 0 else f"inner:{index}",
                                   "clause": clause, "column": field, "operator": node.key, "literal": value,
                                   "is_string": rhs.is_string if rhs is not None else None,
                                   "right_binding": other, "sql": sql_text(node),
                                   "explicit_role": expected, "role_supported": expected in (None, physical)})
    return predicates


def bound_signature(records):
    keys = ("table", "role", "scope", "clause", "column", "operator", "literal", "is_string")
    return Counter(tuple(p[k] for k in keys) for p in records if p["literal"] is not None)


def role_safe(original, proposed):
    return (original is not None and proposed is not None and
            bound_signature(original) == bound_signature(proposed) and
            all(p["role_supported"] for p in proposed))


def event_signature(sql, schema, headings, records):
    tree = parse(sql)
    if not isinstance(tree, exp.Select) or records is None:
        return None
    flat = tree.copy()
    flat.set("distinct", None)
    if not plain_select(flat):
        return None
    sources = schema.sources(tree)
    if not tree.args.get("joins") or not sources or [p.alias_or_name.casefold() for p in tree.expressions] != headings:
        return None
    outputs = tuple(bind_column(p.unalias(), schema, sources) for p in tree.expressions)
    if not all(outputs) or any(j.args.get("side") or j.args.get("kind") not in (None, "INNER")
                               for j in tree.args["joins"]):
        return None
    joins, filters = Counter(), Counter()
    for record in records:
        left = (record["table"], record["column"])
        right = record["right_binding"]
        op = record["operator"]
        if record["scope"] != "outer" or not left[0]:
            return None
        if op == "eq" and right and (*left, *right) in schema.edges[left[0]]:
            joins[tuple(sorted((left, right)))] += 1
        elif record["literal"] is not None and op in {c.key for c in COMPARISONS} and record["role_supported"]:
            filters[(left, op, record["literal"], record["is_string"])] += 1
        else:
            return None
    if len(joins) != len(sources) - 1:
        return None
    return outputs, joins, filters


def priority(sql, reference, headings, schema, question, records):
    conflicts = sum(not p["role_supported"] for p in records or ())
    tree = parse(sql)
    relations = event_signature(sql, schema, headings, records)
    # A named event permits preserving its observed flat join row source.
    event_rows = (bool(EVENT.search(question)) and relations is not None and relations == reference
                  and not tree.args.get("distinct")
                  and not re.search(r"\b(?:distinct|unique|different|each)\b", question, re.I))
    return -conflicts, bool(event_rows)

