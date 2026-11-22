"""Compile row inequality and entity exclusion from the same grounded relation plan."""
from dataclasses import dataclass
import re

from sqlglot import exp

from repair_rules import parse
from step5x_edits import physical_table
from step7_semantics import join_path, top_tables
from step8_groups import MEASURE
from step10_bindings import existing, local_occurrences, tokens
from step11_absence import ABSENCE, parent_sqls
from step19_grounding import package
from step20_paths import meaningful, output_options, projection
from step21_carry import anchor, conjuncts

NEGATION = re.compile(
    r"\b(?:not|never|without|except|neither)\b|\b\w+n['’]t\b", re.I)
ROW_REQUEST = re.compile(
    r"\b(?:at least one|one or more|some)\b[^?.;]{0,55}\b(?:outside|other than|not in|different from)\b",
    re.I)
COMPLEX = re.compile(r"\b(?:for each|group(?:ed)? by|top|bottom|highest|lowest|most|fewest|order(?:ed)? by)\b", re.I)


@dataclass(frozen=True)
class Plan:
    parent: exp.Select
    root: str
    projection_fields: tuple
    path: tuple
    blocked_table: str
    blocked_field: str
    atom: exp.Expression
    required: tuple
    distinct: bool
    priority: int


def negative_span(question):
    natural = ABSENCE.search(question)
    return natural or NEGATION.search(question)


def forward_edge(schema, a, k, b, ref):
    return any(fk["from"].lower() == k and fk["table"].lower() == b and fk["to"].lower() == ref
               for fk in schema.tables[a]["foreign_keys"])


def repeated_relation(schema, path):
    return any(not forward_edge(schema, a, k, b, ref) for a, k, b, ref in path)


def root_constraints(tree, root, blocked, schema, question, words):
    where = tree.args.get("where")
    if where is None:
        return ()
    sources = top_tables(tree, schema)
    result = []
    for item in conjuncts(where.this):
        if not isinstance(item, (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Like)):
            continue
        if not isinstance(item.this, exp.Column) or not isinstance(item.expression, exp.Literal):
            continue
        source = existing(item.this, sources, schema)
        if not source or item.expression.this.casefold() == blocked.casefold():
            continue
        _, table, name = source
        if anchor(item, question, words, name, table):
            if table != root:
                return None
            result.append((item, name))
    return tuple(result)


def filter_anchors(tree, schema, question, trailing_words):
    for scope in tree.find_all(exp.Select):
        where = scope.args.get("where")
        if where is None:
            continue
        sources = top_tables(scope, schema)
        for atom in where.find_all(exp.EQ, exp.NEQ, exp.Like):
            if atom.find_ancestor(exp.Select) is not scope or atom.find_ancestor(exp.Or) is not None:
                continue
            if not isinstance(atom.this, exp.Column) or not isinstance(atom.expression, exp.Literal):
                continue
            raw = atom.expression.this.strip("%")
            if not raw or not local_occurrences(trailing_words, raw):
                continue
            binding = existing(atom.this, sources, schema)
            if not binding:
                continue
            _, table, field = binding
            # A bare integer with no role may describe a limit or a code elsewhere.
            role = meaningful(field)
            if not atom.expression.is_string and not (role & set(trailing_words)):
                continue
            if isinstance(atom, exp.Like) and not re.search(r"\b(?:contain|contains|start|starts|end|ends)\b", question, re.I):
                continue
            yield table, field, atom, 6 * len(role & set(trailing_words))


def root_options(tree, headings, schema, words):
    sources = top_tables(tree, schema)
    first = next(iter(sources.values()), None)
    for fields in output_options(tree, headings, schema, set(words)):
        roots = {t for _, t, _ in fields if t}
        if not roots and first:
            roots = {first}
        if len(roots) == 1:
            yield next(iter(roots)), fields, first


def plans(obs, schema, fallback, extra=()):
    question = obs["task"]["question"]
    match = negative_span(question)
    if not match or COMPLEX.search(question) or ROW_REQUEST.search(question):
        return []
    prefix = tokens(question[:match.start()])
    tail = tokens(question[match.start():])
    heads = obs["example"]["columns"]
    if any(MEASURE.fullmatch(h.strip().lower()) and "distinct" in h.lower() for h in heads):
        return []
    parents = list(dict.fromkeys([*parent_sqls(obs, fallback), *extra]))
    explicit_distinct = bool(re.search(r"\b(?:distinct|unique)\b", question, re.I))
    results = {}
    for text in parents[:24]:
        tree = parse(text)
        if not isinstance(tree, exp.Select) or any(tree.args.get(k) for k in (
                "group", "having", "order", "limit", "offset", "with")):
            continue
        filters = list(filter_anchors(tree, schema, question, tail))
        if not filters:
            continue
        if len({a.expression.this.casefold() for _, _, a, _ in filters}) > 1:
            continue
        for root, fields, first in root_options(tree, heads, schema, prefix + tail):
            root_evidence = meaningful(root) & set(prefix)
            if not root_evidence and root != first:
                continue
            for t, name, atom, binding_rank in filters:
                if t == root:
                    continue
                required = root_constraints(tree, root, atom.expression.this, schema, question, prefix + tail)
                if required is None:
                    continue
                for path in sorted({tuple(p) for p in schema.path_options([root], t, 3)}):
                    if not repeated_relation(schema, path):
                        continue
                    path_roles = set().union(*(meaningful(dest) for _, _, dest, _ in path))
                    rank = (110 + 8 * len(root_evidence) + 5 * (first == root) + binding_rank
                            + len(path_roles & set(tail)) * 3 - 5 * len(path)
                            + sum(s for s, _, _ in fields) + 5 * len(required))
                    plan = Plan(tree, root, fields, path, t, name, atom, required, explicit_distinct, rank)
                    key = (root, tuple((table, field) for _, table, field in fields), path, name,
                           atom.expression.sql(), isinstance(atom, exp.Like),
                           tuple((f, a.key, a.expression.sql()) for a, f in required), explicit_distinct)
                    if key not in results or rank > results[key].priority:
                        results[key] = plan
    return sorted(results.values(), key=lambda plan: -plan.priority)[:8]


def blocked_filter(plan, schema, aliases):
    node = plan.atom
    op = exp.Like if isinstance(node, exp.Like) else exp.EQ
    return op(this=schema.column(plan.blocked_table, plan.blocked_field, aliases[plan.blocked_table]),
              expression=node.expression.copy())


def compile_plan(plan, schema, headings, arm):
    tree = exp.select().from_(physical_table(schema, plan.root, "r0"))
    root_only = {plan.root: "r0"}
    selected = projection(tree, headings, plan.projection_fields, schema, root_only)
    if not selected:
        return None
    tree.select(*selected, append=False, copy=False)
    if plan.distinct:
        tree.distinct(copy=False)
    for condition, field in plan.required:
        atom = condition.copy()
        atom.set("this", schema.column(plan.root, field, "r0"))
        tree.where(atom, copy=False)
    if arm == "row":
        aliases = dict(root_only)
        join_path(tree, plan.path, schema, aliases)
        match = blocked_filter(plan, schema, aliases)
        predicate = (exp.Not(this=match) if isinstance(match, exp.Like) else
                     exp.NEQ(this=match.this.copy(), expression=match.expression.copy()))
    else:
        a, k, b, ref = plan.path[0]
        aliases = {**root_only, b: "r1"}
        sub = exp.select("1").from_(physical_table(schema, b, "r1"))
        join_path(sub, plan.path[1:], schema, aliases)
        sub.where(exp.EQ(this=schema.column(a, k, "r0"), expression=schema.column(b, ref, "r1")),
                  blocked_filter(plan, schema, aliases), copy=False)
        predicate = exp.Not(this=exp.Exists(this=sub))
    tree.where(predicate, copy=False)
    return tree


def candidate_pool(obs, schema, fallback, arm, extra=()):
    headings = obs["example"]["columns"]
    raw = []
    for plan in plans(obs, schema, fallback, extra):
        tree = compile_plan(plan, schema, headings, arm)
        if tree is None:
            continue
        raw.append((tree, plan.priority, arm, {"root": plan.root, "path": plan.path,
                                               "filter": (plan.blocked_table, plan.blocked_field),
                                               "value": plan.atom.expression.this}))
    # Keep matched prior probes in the analysis pool. The agent will not
    # issue a second check for a SQL already in the history.
    return package({"history": []}, fallback, raw)
