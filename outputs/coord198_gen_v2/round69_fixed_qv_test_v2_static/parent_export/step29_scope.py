"""Compile one correlated witness or separately correlated witnesses."""
from dataclasses import dataclass
import re

from sqlglot import exp

from repair_rules import parse
from step4_edits import Schema
from step5x_edits import physical_table
from step7_semantics import join_path, top_tables
from step10_bindings import existing, field_words, local_occurrences, tokens
from step19_grounding import package, observed_schema
from step26_relations import Filter, edge_is_forward, membership
from step27_grain import primary
from step28_roles import link_columns

BOTH = re.compile(r"\bboth\b([^?.;]+?)\band\b([^?.;]+)", re.I)
EXCLUSION = re.compile(r"\b(?:but not|and not)\b", re.I)
QUALIFIER = re.compile(r"\b(?:who|whose|which|that|with|have|has|had|made|taking|"
                       r"are trained|are taking|are playing)\b", re.I)
RICH = re.compile(r"\b(?:for each|each|every|highest|lowest|most|fewest|top|"
                  r"at least|at most|exactly|number of times|total number of)\b", re.I)
NEGATION = re.compile(r"\b(?:no|not|without|never)\b", re.I)


@dataclass(frozen=True)
class Scope:
    kind: str
    subject: frozenset
    terms: tuple = ()


def words(name):
    return set(field_words(name))


def subject_words(question):
    front = re.sub(r"^\s*(?:which|what|who|how many)\b", "", question, flags=re.I)
    match = QUALIFIER.search(front)
    return frozenset(words(front[:match.start()] if match else front))


def supported_subject(subject, table):
    return bool(words(table)) and words(table) <= subject


def anchored(value, question):
    clean = value.strip("%")
    if local_occurrences(question, clean):
        return True
    parts = tokens(clean)
    # Preserve an already checked literal when the question uses its
    # adjectival form, such as Australia / Australian.
    return len(parts) == 1 and len(parts[0]) >= 5 and (
        parts[0] + "n" in question or parts[0] + "ian" in question)


def scope_signal(obs, sql):
    question = obs["task"]["question"]
    tree = parse(sql)
    if not isinstance(tree, exp.Select) or RICH.search(question):
        return None
    if any(tree.args.get(k) for k in ("with", "group", "having", "limit", "offset", "windows")):
        return None
    subject = subject_words(question)
    where = tree.args.get("where")
    both = BOTH.search(question)
    exclusion = EXCLUSION.search(question)
    if where and (both or exclusion):
        if re.search(r"\b(?:same|single)\b", question, re.I):
            return None
        spans = both.groups() if both else (question[:exclusion.start()], question[exclusion.end():])
        space = tuple(tokens(s) for s in spans)
        values = [lit.this for lit in where.find_all(exp.Literal)]
        anchors = tuple(sorted({v for v in values if local_occurrences(span, v)}) for span in space)
        if all(len(v) == 1 for v in anchors) and anchors[0] != anchors[1]:
            return Scope("positive_negative" if exclusion else "separate_positive",
                         subject, tuple(v[0] for v in anchors))
        return None
    if (NEGATION.search(question) or not tree.args.get("joins") or not QUALIFIER.search(question)
            or len(list(tree.find_all(exp.Select))) != 1):
        return None
    visible = observed_schema(obs, tree)
    sources = top_tables(tree, visible)
    cols = [p.unalias() for p in tree.selects]
    if not all(isinstance(c, exp.Column) and existing(c, sources, visible) for c in cols):
        return None
    named = {t for t in sources.values() if supported_subject(subject, t)}
    if len(named) != 1:
        return None
    root = next(iter(named))
    fields = [existing(c, sources, visible) for c in cols]
    if tree.args.get("distinct") and re.search(r"\b(?:distinct|unique|different)\b", question, re.I):
        return None
    if where:
        values = list(where.find_all(exp.Literal))
        question_tokens = tokens(question)
        if not values or not all(anchored(v.this, question_tokens) for v in values):
            return None
        filters = [existing(c, sources, visible) for c in where.find_all(exp.Column)]
        if not all(filters):
            return None
        if all(b[1] == root for b in filters) and not any(
                t != root and supported_subject(words(question) - words(root), t) for t in sources.values()):
            return None
    # Entity identity is already preserved when a distinct projection includes
    # an explicitly requested identifier of the root.
    if tree.args.get("distinct") and any(
            b[1] == root and "id" in words(c.name)
            and re.search(r"\bids?\b", question, re.I) for c, b in zip(cols, fields, strict=True)):
        return None
    return Scope("single_witness", subject)


def declared_edges(tree, sources, schema):
    edges = link_columns(tree, sources, schema)
    if edges is None or any(e not in schema.edges[e[0]] for e in edges):
        return ()
    tables = set(sources.values())
    if len(tables) != len(sources) or len(edges) + 1 != len(tables):
        return ()
    return tuple(edges)


def outer_columns(tree, sources, schema, root, edges):
    """Map an output to the root, including an equated foreign key."""
    cols = []
    equivalent = {(b, ref): k for a, k, b, ref in edges if a == root}
    equivalent.update({(a, k): ref for a, k, b, ref in edges if b == root})
    for select in tree.selects:
        node = select.unalias()
        if not isinstance(node, exp.Column):
            return ()
        hit = existing(node, sources, schema)
        if not hit:
            return ()
        _, table, col = hit
        root_col = col if table == root else equivalent.get((table, col))
        if not root_col:
            return ()
        cols.append(root_col)
    return tuple(cols)


def single_witness(tree, schema, scope):
    sources = top_tables(tree, schema)
    edges = declared_edges(tree, sources, schema)
    if not edges:
        return
    roots = [(a, t) for a, t in sources.items() if supported_subject(scope.subject, t)]
    if len(roots) != 1:
        return
    alias, root = roots[0]
    key = primary(schema, root)
    columns = outer_columns(tree, sources, schema, root, edges)
    if not key or not columns or (tree.args.get("distinct") and set(key) <= set(columns)):
        return
    reverse_edge = any((a == root and not edge_is_forward(schema, e)) or
                       (b == root and not edge_is_forward(schema, (b, ref, a, k)))
                       for e in edges for a, k, b, ref in (e,))
    if not reverse_edge:
        return
    root_alias = next(a for i in range(len(sources) + 1) if (a := f"entity_{i}") not in sources)
    outer = exp.select(*(schema.column(root, col, root_alias) for col in columns)).from_(
        physical_table(schema, root, root_alias))
    sub = tree.copy().select(exp.Literal.number(1), append=False)
    sub.set("distinct", None)
    sub.set("order", None)
    for pk in key:
        sub.where(exp.EQ(this=schema.column(root, pk, root_alias),
                         expression=schema.column(root, pk, alias)), copy=False)
    outer.where(exp.Exists(this=sub), copy=False)
    if tree.args.get("order"):
        sort = tree.args["order"].copy()
        for column in list(sort.find_all(exp.Column)):
            bound = existing(column, sources, schema)
            if not bound or bound[1] != root:
                return
            column.replace(schema.column(root, bound[2], root_alias))
        if any(isinstance(key.this, exp.Literal) for key in sort.expressions):
            return
        outer.set("order", sort)
    yield outer, 145, "one_correlated_witness", {
        "subject": root, "correlation_key": key, "output": columns,
        "path_edges": edges, "record_condition": "one shared witness for the original WHERE",
        "previous_label_distinct": bool(tree.args.get("distinct")),
    }


def atoms_of(tree, schema, values):
    answers = {}
    sources = top_tables(tree, schema)
    where = tree.args["where"]
    clauses = [(where.this, sources, False)]
    while clauses:
        node, available, inner = clauses.pop()
        if isinstance(node, (exp.And, exp.Or)):
            clauses.extend((part, available, inner) for part in (node.this, node.expression))
            continue
        if isinstance(node, exp.Paren):
            clauses.append((node.this, available, inner))
            continue
        if isinstance(node, exp.Exists) and not inner and isinstance(node.this, exp.Select):
            nested = node.this
            local = top_tables(nested, schema)
            if (len(local) != 1 or local.keys() & available.keys() or not nested.args.get("where")
                    or any(nested.args.get(k) for k in ("joins", "group", "having", "limit", "offset", "with"))):
                return ()
            clauses.append((nested.args["where"].this, available | local, True))
            continue
        if not isinstance(node, (exp.EQ, exp.NEQ, exp.In)) or not isinstance(node.this, exp.Column):
            return ()
        bind = existing(node.this, available, schema)
        if not bind:
            return ()
        if inner and isinstance(node, exp.EQ) and isinstance(node.expression, exp.Column):
            other = existing(node.expression, available, schema)
            if not other or (bind[1], bind[2], other[1], other[2]) not in schema.edges[bind[1]]:
                return ()
            continue
        literals = node.expressions if isinstance(node, exp.In) and not node.args.get("query") else [node.expression]
        if not literals or not all(isinstance(v, exp.Literal) and v.this in values for v in literals):
            return ()
        for literal in literals:
            binding = (bind[1], bind[2], literal)
            if literal.this in answers and answers[literal.this] != binding:
                return ()
            answers[literal.this] = binding
    if set(answers) != set(values):
        return ()
    return tuple(answers[v] for v in values)


def role_field(atoms, root, schema, subject, question, outer_sources):
    first, second = atoms
    if first[:2] == second[:2]:
        return first[:2]
    if first[0] not in outer_sources or second[0] in outer_sources:
        return None
    # An independently named field on the first event can license both
    # values when the second value was incorrectly pushed to a nested role.
    if not words(first[1]) <= words(question) or not supported_subject(subject, root):
        return None
    if words(second[0]) - words(first[0]) <= words(question):
        return None
    if len(list(schema.path_options([root], first[0], 2))) != 1:
        return None
    return first[:2]


def separate_witnesses(obs, tree, schema, scope):
    sources = top_tables(tree, schema)
    edges = declared_edges(tree, sources, schema)
    if not edges or tree.args.get("order"):
        return
    atoms = atoms_of(tree, schema, scope.terms)
    if not atoms:
        return
    roots = [t for t in sources.values() if supported_subject(scope.subject, t)]
    if len(roots) == 1:
        root = roots[0]
        outputs = outer_columns(tree, sources, schema, root, edges)
        field = role_field(atoms, root, schema, scope.subject, obs["task"]["question"], set(sources.values()))
        if not outputs or not field:
            return
        event, name = field
        paths = sorted({tuple(p) for p in schema.path_options([root], event, 3)})
        if len(paths) != 1 or not paths[0] or all(edge_is_forward(schema, e) for e in paths[0]):
            return
        path = paths[0]
        root_key = primary(schema, root)
        if not root_key:
            return
        aliases = {root: "e0"}
        outer = exp.select(*(schema.column(root, c, "e0") for c in outputs)).from_(physical_table(schema, root, "e0"))
        if tree.args.get("distinct"):
            outer.distinct(copy=False)
        for i, (_, _, value) in enumerate(atoms):
            atom = Filter(event, name, exp.EQ(expression=value.copy()))
            membership(outer, path, schema, aliases, atom, negative=i == 1 and scope.kind == "positive_negative")
        yield outer, 165, "independent_entity_witnesses", {
            "subject": root, "correlation_key": root_key, "condition_field": field,
            "separate_values": scope.terms, "path_edges": path, "second_is_absence": scope.kind == "positive_negative",
        }
        return
    # A named grouping attribute can itself be the requested entity, as in
    # counts of regions across records in multiple child rows.
    if len(tree.selects) != 1:
        return
    measure = tree.selects[0].unalias()
    if not isinstance(measure, exp.Count) or not isinstance(measure.this, exp.Distinct):
        return
    cols = measure.this.expressions
    if len(cols) != 1 or not isinstance(cols[0], exp.Column):
        return
    group = existing(cols[0], sources, schema)
    if not group or not words(group[2]) <= scope.subject or atoms[0][:2] != atoms[1][:2]:
        return
    root, label = group[1:]
    event, name = atoms[0][:2]
    paths = sorted({tuple(p) for p in schema.path_options([root], event, 3)})
    if len(paths) != 1 or not paths[0] or not paths[0][-1][2] == event:
        return
    path = paths[0]
    branches = []
    for _, _, value in atoms:
        aliases = {root: "r0"}
        query = exp.select(schema.column(root, label, "r0").as_("entity_value")).from_(physical_table(schema, root, "r0"))
        join_path(query, path, schema, aliases)
        query.where(Filter(event, name, exp.EQ(expression=value.copy())).bind(schema, aliases), copy=False)
        branches.append(query)
    combined = exp.except_(*branches) if scope.kind == "positive_negative" else exp.intersect(*branches)
    outer = exp.select(exp.Count(this=exp.column("entity_value", table="w"))).from_(combined.subquery("w"))
    yield outer, 160, "independent_group_witnesses", {
        "subject": (root, label), "condition_field": (event, name), "values": scope.terms,
        "path_edges": path, "second_is_absence": scope.kind == "positive_negative",
    }


def candidate_pool(obs, fallback, tables, scope):
    tree = parse(fallback)
    schema = Schema(tables)
    raw = single_witness(tree, schema, scope) if scope.kind == "single_witness" else separate_witnesses(obs, tree, schema, scope)
    return package({"history": []}, fallback, raw)
