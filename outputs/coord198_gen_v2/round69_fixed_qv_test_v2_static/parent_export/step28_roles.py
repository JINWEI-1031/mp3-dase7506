"""Repair a relation-role conflict supported by a previously executed SQL."""
from collections import Counter
import re

from sqlglot import exp

from repair_rules import parse
from step4_edits import Schema
from step5x_edits import physical_table
from step7_semantics import join_path, top_tables
from step10_bindings import existing, field_words, local_occurrences, tokens
from step19_grounding import observed_schema, package
from step26_relations import membership

COMPLEX = re.compile(r"\b(?:and their|corresponding|for each|both|either|how many|number of|"
                     r"most|fewest|average|top|highest|lowest|greatest|least|without|not)\b", re.I)
ABSENCE = re.compile(r"\b(?:without|do not|does not|don['’]t|doesn['’]t)\b", re.I)
ATOMS = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.In, exp.Between, exp.Like)
LABEL = re.compile(r"[a-z_][a-z0-9_ ]*", re.I)


def words(name):
    result = set()
    for word in field_words(name):
        if word.startswith(("authoriz", "authoris")):
            result.add("authorize")
        elif word.startswith(("destroy", "destruct")):
            result.add("destroy")
        else:
            result.add(word)
    return result


def named(table, question):
    return bool(words(table)) and words(table) <= question


def output_named(output, conditions, question):
    return (any(named(t, question) for _, t, _ in output)
            or len({t for _, t, _ in [*output, *conditions]}) == 1
            and all(words(c) <= question and words(c) - {"name", "id", "code"} for _, _, c in output))


def independent_roles(tables, retained, question):
    retained_words = set().union(*(words(t) for t in retained))
    return {t for t in tables if (words(t) - retained_words) & question}


def selectors(tree):
    if not isinstance(tree, exp.Select) or any(tree.args.get(k) for k in
                                               ("group", "having", "offset", "with", "limit", "windows")):
        return ()
    cols = [p.unalias() for p in tree.selects]
    if cols and all(isinstance(p, exp.Column) and not p.is_star for p in cols):
        return tuple(cols)
    return ()


def link_columns(tree, sources, schema):
    """Read inner equijoins without treating their keys as foreign-key evidence."""
    result = []
    for join in tree.args.get("joins") or []:
        on = join.args.get("on")
        if join.side or join.kind not in ("", "INNER") or not isinstance(on, exp.EQ):
            return None
        if not all(isinstance(c, exp.Column) for c in (on.this, on.expression)):
            return None
        a, b = (existing(c, sources, schema) for c in (on.this, on.expression))
        if not a or not b or a[1] == b[1]:
            return None
        result.append((a[1], a[2], b[1], b[2]))
    return result


def anti_shape(tree, obs):
    negation = ABSENCE.search(obs["task"]["question"])
    where = tree.args.get("where")
    if not negation or not where or not isinstance(where.this, exp.Not) or not isinstance(where.this.this, exp.In):
        return None
    node = where.this.this
    sub = node.args.get("query")
    if not isinstance(node.this, exp.Column) or not isinstance(sub, exp.Subquery):
        return None
    inner = sub.this
    if tree.args.get("joins") or not isinstance(inner, exp.Select) or inner.args.get("joins"):
        return None
    tables = list(inner.find_all(exp.Table))
    if len(tables) != 1 or len(inner.selects) != 1 or not isinstance(inner.selects[0], exp.Column):
        return None
    # A non-key IN projection and an explicit relation-bearing verb are
    # visible before schema. Keep every literal qualifier for other plans.
    tail = words(obs["task"]["question"][negation.end():])
    if node.this.name.lower() == inner.selects[0].name.lower() or inner.find(exp.Literal):
        return None
    if not tail & (words(tables[0].name) - {"id", "name"}):
        return None
    if not re.search(r"\b(?:any|a|an)\b", obs["task"]["question"][negation.end():], re.I):
        return None
    return node.this, inner, tables[0].name.lower(), tail


def shape(obs, fallback):
    tree = parse(fallback)
    question = obs["task"]["question"]
    headings = obs["example"]["columns"]
    cols = selectors(tree)
    if not cols or len(cols) != len(headings) or not all(LABEL.fullmatch(h) for h in headings):
        return None
    anti = anti_shape(tree, obs)
    if anti:
        return tree, ("anti_role",)
    if COMPLEX.search(question) or len(list(tree.find_all(exp.Select))) != 1:
        return None
    joins = tree.args.get("joins") or []
    where = tree.args.get("where")
    if not joins or len(joins) > 4 or not where:
        return None
    if not next(where.find_all(*ATOMS), None) or where.find(exp.Not):
        return None
    qwords = words(question)
    if not any(local_occurrences(tokens(question), n.this) for n in where.find_all(exp.Literal)):
        return None
    observed = observed_schema(obs, tree)
    sources = top_tables(tree, observed)
    edges = link_columns(tree, sources, observed)
    if not edges:
        return None
    output = [existing(c, sources, observed) for c in cols]
    clauses = [where, *([tree.args["order"]] if tree.args.get("order") else [])]
    conditions = [existing(c, sources, observed) for clause in clauses for c in clause.find_all(exp.Column)]
    if not all([*output, *conditions]):
        return None
    required = {b[1] for b in [*output, *conditions]}
    degree = Counter(t for a, _, b, _ in edges for t in (a, b))
    available = set(sources.values())
    cues = []
    spare = available - required
    unrelated = spare - independent_roles(spare, required, qwords)
    if any(degree[t] == 1 and not named(t, qwords) for t in unrelated):
        if output_named(output, conditions, qwords):
            cues.append("unused_leaf")
    if any(a in required and b in required and k != ref and "id" in words(k) | words(ref)
           and (named(a, qwords) or named(b, qwords)) for a, k, b, ref in edges):
        cues.append("asymmetric_key")
    old_fields = {col for _, _, col in output}
    if any(head.lower().replace(" ", "_") not in old_fields and named(table, qwords)
           for head, (_, table, _) in zip(headings, output, strict=True)):
        cues.append("heading_role")
    if any(not named(t, qwords) for _, t, _ in output) and any(
            named(t, qwords) for t in available - {b[1] for b in output}):
        # A named table must share a projection word with the question or
        # already expose an exact requested column in an executed query.
        mentioned = {t for t in available if named(t, qwords)}
        if any(h.lower().replace(" ", "_") in observed.columns[t] for t in mentioned for h in headings) or (
                len({b[1] for b in output}) > 1 and any(words(h) & qwords for h in headings)):
            cues.append("projection_relation")
        elif any(degree[t] == 1 and t not in required for t in mentioned):
            cues.append("named_unprojected_leaf")
    return (tree, tuple(cues)) if cues else None


def entity_anti(obs, tree, schema):
    anti = anti_shape(tree, obs)
    if not anti:
        return
    old, inner, event, tail = anti
    if event not in schema.tables:
        return
    sources = top_tables(tree, schema)
    binding = existing(old, sources, schema)
    if not binding:
        return
    _, root, identity_key = binding
    if not schema.columns[root][identity_key]["pk"]:
        return
    output = [existing(c, sources, schema) for c in selectors(tree)]
    if not all(b and b[1] == root for b in output):
        return
    relations = []
    trivial = words(root) | {"code", "name", "id", "by", "to", "be", "the"}
    for edge in schema.edges[root]:
        a, k, b, ref = edge
        if (a, k, b) != (root, identity_key, event):
            continue
        support = (words(ref) & tail) - trivial
        if support:
            relations.append((len(support), edge, sorted(support)))
    relations.sort(reverse=True)
    if not relations or len(relations) > 1 and relations[0][0] == relations[1][0]:
        return
    _, edge, support = relations[0]
    if edge[3] == inner.selects[0].name.lower():
        return
    inner_where = inner.args.get("where")
    if inner_where:
        # Unrequested date null tests cannot stand in for actor identity.
        # Refuse to discard any other kind of inner condition.
        leaves = [n for n in inner_where.this.walk() if not isinstance(n, (exp.Or, exp.And, exp.Paren, exp.Identifier))]
        if (not all(isinstance(n, (exp.Is, exp.Column, exp.Null)) for n in leaves)
                or not all(isinstance(n.this, exp.Column) and isinstance(n.expression, exp.Null)
                           and not words(n.this.name) & tail for n in inner_where.find_all(exp.Is))):
            return
    query = tree.copy()
    query.set("where", None)
    membership(query, (edge,), schema, {root: output[0][0]}, negative=True)
    yield query, 160, "authorizer_or_actor_role", {"entity": root, "event": event, "role": edge,
                                                    "question_role_terms": support,
                                                    "old_key": inner.selects[0].name}


def output_roles(tree, schema, sources, headings, qwords, cues):
    old = [existing(c, sources, schema) for c in selectors(tree)]
    if not all(old):
        return
    occupants = set(sources.values())
    options = []
    old_fields = {f for _, _, f in old}
    for (alias, table, field), heading in zip(old, headings, strict=True):
        label = heading.lower().replace(" ", "_")
        choices = [(table, field)]
        if label not in old_fields and "heading_role" in cues and label in schema.columns[table] and named(table, qwords):
            choices = [(table, label)]
        if "projection_relation" in cues or "named_unprojected_leaf" in cues:
            if not named(table, qwords):
                stronger = [(t, label) for t in sorted(occupants) if t != table and
                            label in schema.columns[t] and named(t, qwords)]
                if len(stronger) == 1:
                    choices = stronger
        options.append(choices)
    output = tuple(c[0] for c in options)
    if len(set(output)) == len(output):
        yield output, tuple((b[1], b[2]) for b in old)


def graph_path(root, targets, schema, old_edges):
    graph = {frozenset(((a, k), (b, ref))) for a, k, b, ref in old_edges}
    present = {root}
    edges = ()
    for target in sorted(targets - {root}):
        if target in present:
            continue
        variants = sorted({tuple(p) for p in schema.path_options(sorted(present), target, 3)})
        ranked = sorted((len(path), -sum(frozenset(((a, k), (b, ref))) in graph
                                         for a, k, b, ref in path), path) for path in variants)
        if not ranked or len(ranked) > 1 and ranked[0][:2] == ranked[1][:2]:
            return None
        chosen = ranked[0][2]
        edges += chosen
        present.update(edge[2] for edge in chosen)
    return edges, present


def relation_roles(obs, tree, schema, cues):
    qwords = words(obs["task"]["question"])
    heads = obs["example"]["columns"]
    sources = top_tables(tree, schema)
    edges = link_columns(tree, sources, schema)
    if not edges or len(sources) != len(set(sources.values())):
        return
    clauses = {k: tree.args[k] for k in ("where", "order") if tree.args.get(k)}
    bindings = {}
    for clause in clauses.values():
        for column in clause.find_all(exp.Column):
            bound = existing(column, sources, schema)
            if not bound:
                return
            bindings[(column.table.lower(), column.name.lower())] = bound[1:]
    for output, old_output in output_roles(tree, schema, sources, heads, qwords, cues):
        root = output[0][0]
        targets = {t for t, _ in [*output, *bindings.values()]}
        if not output_named([(None, *p) for p in output], [(None, *p) for p in bindings.values()], qwords):
            continue
        matched = graph_path(root, targets, schema, edges)
        if matched is None:
            continue
        path, present = matched
        dropped = set(sources.values()) - present
        if independent_roles(dropped, present, qwords) or any(named(t, qwords) for t in dropped):
            continue
        declared_missing = [(a, k, b, ref) for a, k, b, ref in edges
                            if a in present and b in present and (a, k, b, ref) not in schema.edges[a]]
        if not (dropped or declared_missing or output != old_output):
            continue
        if (not declared_missing and not dropped and
                {frozenset(((a, k), (b, ref))) for a, k, b, ref in edges} !=
                {frozenset(((a, k), (b, ref))) for a, k, b, ref in path}):
            continue
        query = exp.select().from_(physical_table(schema, root, "r0"))
        aliases = {root: "r0"}
        join_path(query, path, schema, aliases)
        query.select(*(schema.column(t, f, aliases[t]) for t, f in output), copy=False)
        if tree.args.get("distinct"):
            query.distinct(copy=False)

        def rebind(node):
            if isinstance(node, exp.Column):
                table, field = bindings[node.table.lower(), node.name.lower()]
                return schema.column(table, field, aliases[table])
            return node

        for name, clause in clauses.items():
            query.set(name, clause.transform(rebind))
        evidence = {"cues": cues, "output_before": old_output, "output_after": output,
                    "predicate_roles": sorted(set(bindings.values())), "declared_path": path,
                    "removed_join_roles": sorted(dropped), "unbacked_keys": declared_missing}
        yield query, 120 + 10 * bool(declared_missing) + 10 * (output != old_output), "grounded_detail_roles", evidence


def candidate_pool(obs, fallback, tables, plan):
    tree, cues = plan
    schema = Schema(tables)
    raw = entity_anti(obs, tree, schema) if "anti_role" in cues else relation_roles(obs, tree, schema, cues)
    return package({"history": []}, fallback, raw)
