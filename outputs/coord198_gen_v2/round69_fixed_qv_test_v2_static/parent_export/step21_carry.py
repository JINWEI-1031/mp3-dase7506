"""Recover question-anchored predicates from the visible drafts."""
from dataclasses import dataclass
import re

from sqlglot import exp

from repair_rules import parse
from step5x_edits import physical_table
from step7_semantics import join_path, top_tables
from step8_groups import MEASURE
from step10_bindings import ATOMS, existing, field_words, local_occurrences, tokens
from step11_absence import parent_sqls
from step19_grounding import package
from step20_paths import bound_atom, meaningful, output_options, projection, relation_plans

GROUP_REQUEST = re.compile(
    r"\b(?:for each|per|group(?:ed)? by|top|bottom|most|fewest|least|highest|lowest|"
    r"latest|earliest|first|last|rank(?:ed)?|sorted|ordered)\b", re.I)
FIRST_LAST_FIELD = re.compile(r"\b(?:first|last) names?\b", re.I)


@dataclass
class Clause:
    expression: exp.Expression
    bindings: list
    anchors: frozenset
    score: int
    source: int
    nested: bool


def positive_membership(select, outer):
    """Nested filters can be lifted only through positive WHERE membership."""
    current = select
    while current is not outer:
        if any(current.args.get(key) for key in ("group", "having", "limit", "offset", "order")):
            return False
        node = current.parent
        if isinstance(node, exp.Subquery):
            node = node.parent
        if not isinstance(node, (exp.Exists, exp.In)):
            return False
        while node.parent is not None and isinstance(node.parent, (exp.And, exp.Paren)):
            node = node.parent
        if not isinstance(node.parent, exp.Where):
            return False
        current = node.parent.parent
        if not isinstance(current, exp.Select):
            return False
    return True


def conjuncts(expr):
    if isinstance(expr, exp.Paren):
        yield from conjuncts(expr.this)
    elif isinstance(expr, exp.And):
        yield from conjuncts(expr.this)
        yield from conjuncts(expr.expression)
    else:
        yield expr


def local_atoms(expr):
    if isinstance(expr, exp.Paren):
        return local_atoms(expr.this)
    if isinstance(expr, (exp.And, exp.Or)):
        lhs, rhs = local_atoms(expr.this), local_atoms(expr.expression)
        return lhs + rhs if lhs and rhs else []
    if (isinstance(expr, ATOMS) and isinstance(expr.this, exp.Column)
            and not expr.find(exp.Select) and not any(c is not expr.this for c in expr.find_all(exp.Column))):
        return [expr]
    return []


def anchor(atom, question, words, field, table):
    values = list(atom.find_all(exp.Literal))
    if not values:
        return None
    spans, scores = set(), []
    role = meaningful(field)
    entity = meaningful(table)
    for literal in values:
        text = literal.this
        term = tokens(text.strip("%"))
        positions = local_occurrences(words, text.strip("%"))
        if not positions or not term:
            return None
        best = 0
        for i in positions:
            end = i + len(term)
            nearby = set(words[max(0, i - 4):i] + words[end:end + 3])
            match = role & nearby
            quoted = bool(re.search(rf"['\"]{re.escape(text)}['\"]", question, re.I))
            numeric = bool(re.fullmatch(r"[+-]?\d+(?:\.\d+)?", text))
            if quoted or match or numeric and entity & nearby:
                spans.add((i, end))
                best = max(best, 5 + 5 * len(match) + 3 * quoted + 3 * numeric)
        if not best:
            return None
        scores.append(best)
    return frozenset(spans), sum(scores)


def extract(tree, schema, question, words, source):
    for select in tree.find_all(exp.Select):
        where = select.args.get("where")
        if not where or not positive_membership(select, tree):
            continue
        sources = top_tables(select, schema)
        for expr in conjuncts(where.this):
            atoms = local_atoms(expr)
            if not atoms:
                continue
            bindings, slots, rank = [], set(), 0
            for atom in atoms:
                binding = existing(atom.this, sources, schema)
                if not binding:
                    break
                _, table, field = binding
                found = anchor(atom, question, words, field, table)
                if found is None:
                    break
                spans, evidence = found
                slots.update(spans)
                rank += evidence
                bindings.append((evidence, table, field))
            else:
                if expr.find(exp.Or) and "or" not in words:
                    continue
                yield Clause(expr, bindings, frozenset(slots), rank, source, select is not tree)


def plans(clauses, merged, question):
    sources = {c.source for c in clauses}
    lists = [[c for c in clauses if c.source == idx] for idx in sorted(sources)]
    if merged:
        lists.append(clauses)
    emitted = set()
    use_or = bool(re.search(r"\bor\b", question, re.I))
    explicit_and = bool(re.search(r"\band\b", question, re.I))
    for options in lists:
        for seed in sorted(options, key=lambda c: -c.score)[:8]:
            chosen = [seed]
            taken = set(seed.anchors)
            for other in sorted(options, key=lambda c: -c.score):
                if other is seed or other.anchors & taken:
                    continue
                if other.source != seed.source and (use_or or not explicit_and):
                    continue
                if len(chosen) == 3:
                    break
                chosen.append(other)
                taken.update(other.anchors)
            key = tuple(sorted((c.expression.sql(), tuple(c.bindings)) for c in chosen))
            if key not in emitted:
                emitted.add(key)
                yield chosen


def bind_clause(clause, schema, aliases):
    expr = clause.expression.copy()
    for atom, binding in zip(local_atoms(expr), clause.bindings, strict=True):
        rebuilt = bound_atom(atom, binding, schema, aliases)
        if atom is expr:
            return rebuilt
        atom.replace(rebuilt)
    return expr


def project_existing(tree, schema, headings, question, clauses):
    """Keep a scoped predicate and its joins when only the output needs rebuilding."""
    if not clauses or not (tree.args.get("group") or any(c.nested for c in clauses)):
        return
    if (any(tree.args.get(k) for k in ("having", "limit", "offset", "order"))
            or any(MEASURE.fullmatch(h.strip().lower()) for h in headings)):
        return
    sources = top_tables(tree, schema)
    aliases = {t: a for a, t in sources.items()}
    if len(aliases) != len(sources):
        return
    for variant in output_options(tree, headings, schema, set(question)):
        if any(table not in aliases for _, table, _ in variant):
            continue
        query = tree.copy()
        query.set("group", None)
        columns = projection(query, headings, variant, schema, aliases)
        if columns:
            query.select(*columns, append=False, copy=False)
            yield query, 115 + sum(c.score for c in clauses), "scoped_projection", {
                "sources": sorted({c.source for c in clauses}),
                "anchors": sorted(set().union(*(c.anchors for c in clauses)))}


def rebuild(parents, clauses, schema, headings, words, question):
    targets = {t for c in clauses for _, t, _ in c.bindings}
    outputs = {}
    for parent in parents:
        for variant in output_options(parent, headings, schema, set(words)):
            key = tuple((t, f) for _, t, f in variant)
            if key not in outputs or sum(x[0] for x in variant) > sum(x[0] for x in outputs[key]):
                outputs[key] = variant
    options = sorted(outputs.values(), key=lambda v: -sum(s for s, _, _ in v))[:8]
    for variant in options:
        projection_tables = {t for _, t, _ in variant if t}
        for root in sorted(projection_tables or targets)[:2]:
            for _, path, merit in relation_plans(root, targets | projection_tables, schema, set(words), 2):
                aliases = {root: "r0"}
                query = exp.select().from_(physical_table(schema, root, "r0"))
                join_path(query, path, schema, aliases)
                columns = projection(query, headings, variant, schema, aliases)
                if not columns:
                    continue
                query.select(*columns, append=False, copy=False)
                conditions = [bind_clause(c, schema, aliases) for c in clauses]
                query.where(exp.and_(*conditions), copy=False)
                if re.search(r"\b(?:distinct|unique)\b", question, re.I):
                    query.distinct(copy=False)
                yield query, 95 + sum(c.score for c in clauses) + merit, "anchor_paths", {
                    "sources": sorted({c.source for c in clauses}),
                    "anchors": sorted(set().union(*(c.anchors for c in clauses))),
                    "filter_bindings": [b[1:] for c in clauses for b in c.bindings], "foreign_keys": path}


def carry_pool(obs, fallback, schema, mode):
    question = obs["task"]["question"]
    if GROUP_REQUEST.search(FIRST_LAST_FIELD.sub("", question)):
        return []
    words = tokens(question)
    headings = obs["example"]["columns"]
    parents = [tree for sql in parent_sqls(obs, fallback)[:12]
               if isinstance((tree := parse(sql)), exp.Select)
               and not tree.args.get("with")]
    raw, all_clauses = [], []
    for source, tree in enumerate(parents):
        if any(tree.args.get(k) for k in ("limit", "offset", "having", "order")):
            continue
        clauses = list(extract(tree, schema, question, words, source))
        all_clauses.extend(clauses)
        if (not mode == "paths" and clauses and tree.args.get("where")
                and not tree.args["where"].find(exp.Not)):
            raw.extend(project_existing(tree, schema, headings, words, clauses))
    if mode != "scoped" and any(c.nested or parents[c.source].args.get("group") for c in all_clauses):
        for bundle in plans(all_clauses, mode == "merge", question):
            if not any(c.nested or parents[c.source].args.get("group") for c in bundle):
                continue
            raw.extend(rebuild(parents, bundle, schema, headings, words, question))
    # Omit a candidate which loses a strictly stronger, independently anchored alternative.
    richest = [set(entry[3]["anchors"]) for entry in raw]
    return package(obs, fallback, (entry for entry, anchors in zip(raw, richest)
                                   if not any(anchors < other for other in richest)))[:12]
