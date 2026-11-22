"""Synthesize relation plans from target anchors and the declared key graph."""
from dataclasses import dataclass
from itertools import product
import re

from sqlglot import exp

from repair_rules import parse, sql_text
from step4_edits import Schema
from step5x_edits import physical_table
from step7_semantics import join_path
from step10_bindings import compatible, existing, local_occurrences, tokens
from step28_roles import link_columns, words
from step32_plan import tree_plan

ATOMS = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.In, exp.Like, exp.Between)
BOOLEAN = (exp.And, exp.Or, exp.Paren)
GENERIC = {"id", "name", "code", "type", "date", "title", "description", "number", "record", "detail"}
BOTH = re.compile(r"\bboth\b", re.I)
NEGATION = re.compile(r"\b(?:without|not|no|don['’]t|doesn['’]t)\b", re.I)


@dataclass(frozen=True)
class Binding:
    table: str
    field: str
    score: int


def role(table, question):
    return (words(table) - GENERIC) & question


def lexical(value, question):
    return len((words(value) - GENERIC) & question)


def fields(column, heading, sources, schema, question, atom=None):
    current = existing(column, {a: t for a, t in sources.items() if t in schema.tables}, schema)
    name = column.name.lower()
    heading_words = words(heading)
    result = []
    for table, cols in schema.columns.items():
        for field, meta in cols.items():
            exact, named = field == name, bool(role(table, question))
            matches_heading = words(field) == heading_words and bool(heading_words)
            anchored = bool(current and current[1:] == (table, field))
            descriptive = lexical(field, question)
            if atom is not None:
                if not compatible(atom, meta) or not (exact or anchored or
                        descriptive >= 2 and bool(words(name) & words(field))):
                    continue
            elif not (exact or matches_heading or anchored or descriptive >= 2):
                continue
            if not (named or anchored or matches_heading and field not in GENERIC or exact and field not in GENERIC):
                continue
            score = (6 * exact + 5 * matches_heading + 2 * anchored + 4 * named + 2 * descriptive
                     + lexical(table, question))
            result.append(Binding(table, field, score))
    return sorted(result, key=lambda b: (-b.score, b.table, b.field))[:3]


def atoms(where, question):
    if where is None:
        return []
    if isinstance(where, BOOLEAN):
        left = atoms(where.this, question)
        right = atoms(where.expression, question) if where.expression is not None else []
        return None if left is None or right is None else left + right
    if not isinstance(where, ATOMS) or not isinstance(where.this, exp.Column):
        return None
    literals = list(where.find_all(exp.Literal))
    if not literals or len(list(where.find_all(exp.Column))) != 1 or where.find(exp.Select):
        return None
    if not all(local_occurrences(question, node.this) for node in literals):
        return None
    return [where]


def spec(tree, question, width):
    if not isinstance(tree, exp.Select) or len(tree.expressions) != width:
        return None
    if any(not isinstance(e.unalias(), exp.Column) or e.unalias().is_star for e in tree.expressions):
        return None
    if any(tree.args.get(k) for k in ("group", "having", "with", "offset", "windows", "qualify")):
        return None
    if any(not isinstance(join.this, exp.Table) or join.side or join.kind not in ("", "INNER", "CROSS")
           for join in tree.args.get("joins", [])):
        return None
    where = tree.args.get("where")
    predicate = where.this if where else None
    scope, inner = "rows", None
    if isinstance(predicate, exp.Not) and NEGATION.search(question):
        node = predicate.this
        if isinstance(node, exp.In) and isinstance(node.args.get("query"), exp.Subquery):
            inner = node.args["query"].this
            if (not isinstance(inner, exp.Select) or len(inner.expressions) != 1
                    or not isinstance(inner.expressions[0], exp.Column)
                    or any(inner.args.get(k) for k in ("group", "having", "order", "limit", "with"))):
                return None
            source = inner.args.get("from")
            if not source or not isinstance(source.this, exp.Table):
                return None
            scope = "entity exclusion"
            predicate = inner.args["where"].this if inner.args.get("where") else None
            anchors = atoms(predicate, tokens(question))
            if anchors is None:
                return None
            return scope, predicate, anchors, inner
    anchors = atoms(predicate, tokens(question))
    return (scope, predicate, anchors, inner) if anchors is not None else None


def tables(tree):
    return {t.alias_or_name.lower(): t.name.lower() for t in tree.find_all(exp.Table)}


def graph_paths(schema, root, targets, limit=3):
    """Connect terminals with bounded declared paths. Ambiguous shortest paths stay separate."""
    partial = [((), (root,))]
    for target in sorted(set(targets) - {root}):
        expanded = []
        for edges, visited in partial:
            if target in visited:
                expanded.append((edges, visited))
                continue
            paths = sorted(schema.path_options(sorted(visited), target, limit), key=lambda p: (len(p), p))
            if paths:
                for path in [p for p in paths if len(p) == len(paths[0])][:3]:
                    expanded.append((edges + tuple(path), (*visited, *(edge[2] for edge in path))))
        partial = sorted(set(expanded), key=lambda item: (len(item[0]), item))[:4]
    yield from partial


def scope_sql(tree, schema, output, conditions, predicate, path, mode):
    root = output[0].table
    aliases = {root: "r0"}
    query = exp.select().from_(physical_table(schema, root, aliases[root]))
    if mode == "rows":
        join_path(query, path, schema, aliases)
        scoped = query
    else:
        if not path or any(b.table != root for b in output):
            return None
        left, left_key, first, right_key = path[0]
        if left != root:
            return None
        entity_alias = aliases[root]
        aliases = {root: entity_alias, first: "r1"}
        scoped = exp.select("1").from_(physical_table(schema, first, aliases[first]))
        scoped.where(exp.EQ(this=schema.column(root, left_key, entity_alias),
                            expression=schema.column(first, right_key, aliases[first])), copy=False)
        join_path(scoped, path[1:], schema, aliases)
    queue = iter(conditions)
    if predicate is not None:
        def bind(node):
            if isinstance(node, exp.Column):
                b = next(queue)
                return schema.column(b.table, b.field, aliases[b.table])
            return node
        scoped.where(predicate.transform(bind), copy=False)
    if mode != "rows":
        existence = exp.Exists(this=scoped)
        query.where(exp.Not(this=existence) if mode == "entity exclusion" else existence, copy=False)
    for original, binding in zip(tree.expressions, output, strict=True):
        column = schema.column(binding.table, binding.field, aliases[binding.table])
        query.select(exp.alias_(column, original.alias, quoted=True) if original.alias else column, copy=False)
    if tree.args.get("distinct"):
        query.set("distinct", tree.args["distinct"].copy())
    # Ordering is supported when every sort key names one projected role.
    order = tree.args.get("order")
    if order:
        mapped = []
        for item in order.expressions:
            target = item.this
            if isinstance(target, exp.Literal) and target.is_int and 1 <= int(target.this) <= len(output):
                mapped.append(item.copy())
                continue
            if not isinstance(target, exp.Column):
                return None
            indexes = [i for i, col in enumerate(tree.expressions) if target.name.lower() in
                       (col.alias_or_name.lower(), col.unalias().name.lower())
                       and (not target.table or target.table.lower() == col.unalias().table.lower())]
            if len(indexes) != 1:
                return None
            b = output[indexes[0]]
            mapped.append(item.copy())
            mapped[-1].set("this", schema.column(b.table, b.field, aliases[b.table]))
        query.set("order", exp.Order(expressions=mapped))
    if tree.args.get("limit"):
        query.set("limit", tree.args["limit"].copy())
    return query


def plan_options(c, tree, public, observed_schema):
    question = c["question"]
    shape = spec(tree, question, len(c["columns"]))
    if not shape:
        return
    scope, predicate, anchors, inner = shape
    schema = Schema(observed_schema)
    source = {a: t for a, t in tables(tree).items() if t in schema.tables}
    qwords = set(tokens(question))
    output_options = [fields(col.unalias(), heading, source, schema, qwords)
                      for col, heading in zip(tree.expressions, c["columns"], strict=True)]
    condition_options = [fields(atom.this, "", tables(inner) if inner else source, schema, qwords, atom)
                         for atom in anchors]
    if any(not bindings for bindings in (*output_options, *condition_options)):
        return
    width = len(output_options)
    if width + len(condition_options) > 7:
        return
    bindings = sorted(product(*output_options, *condition_options),
                      key=lambda bs: -sum(b.score for b in bs))[:18]
    old_roles = set(source.values())
    edges = link_columns(tree, source, schema)
    old_graph = {frozenset(((a, b), (d, e))) for a, b, d, e in edges or ()}
    for combined in bindings:
        output, cond = combined[:width], combined[width:]
        root = output[0].table
        terminal = {b.table for b in combined}
        if scope == "entity exclusion":
            inner_table = inner.args["from"].this.name.lower()
            if inner_table not in schema.columns or inner_table == root:
                continue
            terminal.add(inner_table)
        for path, visited in graph_paths(schema, root, terminal):
            if scope == "entity exclusion" and (not path or any(edge[0] == root for edge in path[1:])
                                                 or any(b.table == root for b in cond)):
                continue
            removed = old_roles - set(visited)
            added = set(visited) - old_roles
            root_words = role(root, qwords)
            projected = [existing(e.unalias(), source, schema) for e in tree.expressions]
            rebound = any(old is None or old[1:] != (new.table, new.field)
                          for old, new in zip(projected, output, strict=True))
            graph = {frozenset(((a, b), (d, e))) for a, b, d, e in path}
            moved_predicate = any(old is None or old[1:] != (new.table, new.field)
                                  for old, new in zip((existing(a.this, source, schema) for a in anchors),
                                                      cond, strict=True))
            if (scope == "rows" and not rebound and not moved_predicate and not removed and not added
                    and edges is not None and old_graph == graph):
                continue
            old_root_named = any(old and role(old[1], qwords) for old in projected)
            # Keep the public check gate narrow. A removed role must have no
            # independent question anchor, or the projected role must be clearer.
            independent_removed = set().union(*(role(t, qwords) - root_words for t in removed))
            # A table whose entire role is already represented by the root is redundant
            # only when every predicate and displayed column anchors that root.
            anchored_root = root_words or all(lexical(b.field, qwords) for b in combined)
            prune = bool(removed and anchored_root and len(terminal) == 1 and not independent_removed
                         and anchors and not rebound and not moved_predicate and not BOTH.search(question))
            rebind = bool(rebound and root_words and not old_root_named and
                          all(role(b.table, qwords) for b in output) and anchors)
            evidence = "unrequested relation" if prune else "question-named output role" if rebind else None
            if not public:
                evidence = "schema-bound output and predicate path"
            mode = "entity existence" if (scope == "rows" and tree.args.get("distinct") and path and
                                           all(b.table == root for b in output) and
                                           not any(edge[0] == root for edge in path[1:])) else scope
            options = [scope] if mode == scope else [scope, mode]
            both = (scope == "rows" and mode == "entity existence" and BOTH.search(question)
                    and isinstance(predicate, exp.In) and len(predicate.expressions) == 2)
            if both:
                options.append("independent witnesses")
            for chosen in options:
                if chosen == "independent witnesses":
                    witnesses = [scope_sql(tree, schema, output, cond,
                                           exp.EQ(this=predicate.this.copy(), expression=value.copy()),
                                           path, "entity existence") for value in predicate.expressions]
                    if not all(witnesses):
                        continue
                    sql = witnesses[0]
                    sql.where(witnesses[1].args["where"].this, copy=False)
                else:
                    sql = scope_sql(tree, schema, output, cond, predicate, path, chosen)
                basis = {"kind": "declared relation graph", "scope": chosen, "root": root,
                         "bindings": [(b.table, b.field) for b in combined], "path": path,
                         "removed_roles": sorted(removed), "added_roles": sorted(added),
                         "advantage": evidence if chosen != "independent witnesses" or not public else None,
                         "predicate_atoms": [sql_text(a) for a in anchors]}
                if sql is not None:
                    yield tree_plan(sql, basis)


def request_binding(c, parent, public):
    shape = spec(parent, c["question"], len(c["columns"]))
    if not shape:
        return None
    _, _, anchors, _ = shape
    if public and parent.args.get("joins") and anchors and not BOTH.search(c["question"]):
        q = set(tokens(c["question"]))
        source = tables(parent)
        outputs = [e.unalias() for e in parent.expressions]
        columns = [*outputs, *(atom.this for atom in anchors)]
        bound_aliases = {column.table.lower() for column in columns if column.table}
        if len(bound_aliases) == 1 and all(column.table for column in columns):
            alias = next(iter(bound_aliases))
            root = source.get(alias, "")
            rooted = role(root, q) or all(lexical(column.name, q) for column in columns)
            if rooted and any(not (role(t, q) - role(root, q)) for a, t in source.items() if a != alias):
                return "test whether the named output and predicates use a single relation"
        if any(role(t, q) for t in source.values()) and all(
                col.table and not role(source.get(col.table.lower(), ""), q) for col in outputs):
            return "bind the projection to the named relation"
    if not public and (parent.args.get("joins") or parent.args.get("where")):
        return "resolve the draft relation path for grounded output and predicates"
    return None
