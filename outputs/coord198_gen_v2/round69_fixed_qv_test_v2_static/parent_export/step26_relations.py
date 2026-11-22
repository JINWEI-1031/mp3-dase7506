"""Bind projected entities and predicate scopes before compiling relation edits."""
from dataclasses import dataclass
import re

from sqlglot import exp

from repair_rules import parse
from step4_edits import COMPARISONS, OPERATORS, THRESHOLD, Schema
from step5x_edits import physical_table
from step7_semantics import join_path, top_tables
from step10_bindings import existing, field_words, local_occurrences, tokens
from step11_absence import parent_sqls
from step19_grounding import observed_schema, package
from step20_paths import field_options, output_options, relation_plans

ABSENCE = re.compile(r"\b(?:no|without|(?:do|does|did) not|\w+n['’]t)\b", re.I)
PAIR = re.compile(r"\bfrom\s+(.+?)\s+to\s+(.+?)(?=[?.;,]|$)", re.I)
UNSUPPORTED = re.compile(r"\b(?:for each|most|fewest|average|highest|lowest|top|every)\b", re.I)
FREQUENCY = re.compile(r"\b(?:most|greatest|largest|maximum|fewest|least|smallest|minimum)\s+(?:number|count)(?:\s+of\s+(\w+))?", re.I)
FILTER_ATOMS = (exp.In, exp.EQ, exp.NEQ)


@dataclass(frozen=True)
class Filter:
    table: str
    field: str
    value: exp.Expression

    def bind(self, schema, aliases):
        atom = self.value.copy()
        atom.set("this", schema.column(self.table, self.field, aliases[self.table]))
        return atom


def entity_words(table, schema):
    return set(field_words(schema.tables[table]["name"]))


def select_fields(fields, schema, aliases):
    return [schema.column(table, field, aliases[table]) for _, table, field in fields]


def unique_path(schema, root, target, words):
    options = relation_plans(root, {target}, schema, set(words), 3)
    if not options or len(options) > 1 and options[0][2] == options[1][2]:
        return ()
    return options[0][1]


def relation_query(root, fields, schema):
    query = exp.select().from_(physical_table(schema, root, "r0"))
    aliases = {root: "r0"}
    query.select(*select_fields(fields, schema, aliases), append=False, copy=False)
    return query, aliases


def membership(query, path, schema, aliases, atom=None, negative=False):
    a, k, b, ref = path[0]
    inner = {**aliases, b: "m1"}
    sub = exp.select("1").from_(physical_table(schema, b, "m1"))
    join_path(sub, path[1:], schema, inner)
    sub.where(exp.EQ(this=schema.column(a, k, aliases[a]),
                     expression=schema.column(b, ref, inner[b])), copy=False)
    if atom is not None:
        sub.where(atom.bind(schema, inner), copy=False)
    predicate = exp.Exists(this=sub)
    query.where(exp.Not(this=predicate) if negative else predicate, copy=False)


def direct_roles(tree, schema, question, fields):
    match = PAIR.search(question)
    if not match or len({table for _, table, _ in fields}) != 1:
        return
    root = fields[0][1]
    if not (entity_words(root, schema) & set(tokens(question[:match.start()]))):
        return
    roles = ({"origin", "source", "start", "departure"}, {"destination", "end", "arrival"})
    atoms = []
    for role, value in zip(roles, match.groups(), strict=True):
        available = [f for f in schema.columns[root] if set(field_words(f)) & role]
        if len(available) != 1:
            return
        literals = {n.this: n for n in tree.find_all(exp.Literal)
                    if n.is_string and tokens(n.this) == tokens(value)}
        if len(literals) != 1:
            return
        atoms.append(Filter(root, available[0], exp.EQ(expression=next(iter(literals.values())))))
    query, aliases = relation_query(root, fields, schema)
    query.where(*(atom.bind(schema, aliases) for atom in atoms), copy=False)
    yield query, 155, "endpoint_fields", {"output": root, "roles": [a.field for a in atoms]}


def combined_entities(tree, schema, heads, question):
    if len(heads) != 1 or not re.search(r"\b(?:all|both)\b.*\band\b", question, re.I):
        return
    words = set(tokens(question))
    field = heads[0].lower()
    sources = set(top_tables(tree, schema).values())
    targets = sorted(t for t in sources if field in schema.columns[t] and entity_words(t, schema) <= words)
    if len(targets) != 2 or tree.args.get("where"):
        return
    queries = []
    for t in targets:
        query, _ = relation_query(t, [(0, t, field)], schema)
        queries.append(query)
    yield exp.union(*queries), 145, "entity_union", {"entities": targets, "field": field}


def scoped_filters(tree, schema, fields, question):
    words = tokens(question)
    roots = {table for _, table, _ in fields}
    if len(roots) != 1 or UNSUPPORTED.search(question):
        return
    root = next(iter(roots))
    if not entity_words(root, schema) & set(words):
        return
    where = tree.args.get("where")
    if where is None:
        return
    atoms = []
    for atom in where.this.find_all(*FILTER_ATOMS):
        if not isinstance(atom.this, exp.Column) or atom.find(exp.Select):
            continue
        values = list(atom.find_all(exp.Literal))
        if not values or not all(local_occurrences(words, value.this) for value in values):
            continue
        positions = [(i, i + len(tokens(v.this))) for v in values for i in local_occurrences(words, v.this)]
        options = field_options(atom, positions, tree, schema, words)
        if not options:
            continue
        score, table, field = options[0]
        if not score or len(options) > 1 and options[1][0] == score:
            continue
        for literal in values:
            atoms.append(Filter(table, field, exp.EQ(expression=literal.copy())))
    negative = ABSENCE.search(question)
    both = re.search(r"\bboth\b.+\band\b", question, re.I)
    if both and len(atoms) == 2 and (atoms[0].table, atoms[0].field) == (atoms[1].table, atoms[1].field):
        condition_table = atoms[0].table
        path = unique_path(schema, root, condition_table, words)
        if condition_table != root and not path:
            return
        sides = []
        for atom in atoms:
            side, aliases = relation_query(root, fields, schema)
            join_path(side, path, schema, aliases)
            side.where(atom.bind(schema, aliases), copy=False)
            sides.append(side)
        yield exp.intersect(*sides), 150, "both_values", {"field": atoms[0].field, "path": path}
        if path:
            query, aliases = relation_query(root, fields, schema)
            for atom in atoms:
                membership(query, path, schema, aliases, atom)
            yield query, 149, "both_on_entity", {"field": atoms[0].field, "path": path}
    if not negative:
        return
    # A row comparison is meaningful even when the blocked field is on the root.
    if len(atoms) == 1:
        atom = atoms[0]
        path = unique_path(schema, root, atom.table, words)
        if atom.table == root or path:
            query, aliases = relation_query(root, fields, schema)
            join_path(query, path, schema, aliases)
            atom_sql = atom.bind(schema, aliases)
            query.where(exp.NEQ(this=atom_sql.this.copy(), expression=atom_sql.expression.copy()), copy=False)
            yield query, 103, "row_inequality", {"path": path, "filter": [atom.table, atom.field]}
    if atoms:
        return
    tail = set(tokens(question[negative.end():]))
    for target in sorted(schema.tables):
        if target == root or not entity_words(target, schema) <= tail:
            continue
        path = unique_path(schema, root, target, words)
        if not path:
            continue
        # When the question names a relationship role, require an edge saying which one.
        roles = {"advisor", "authoriz", "manager", "destruction", "destroy", "teach"}
        role_words = {word for word in roles if any(w.startswith(word) for w in tail)}
        evidence = " ".join("_".join(edge) for edge in path)
        if role_words and not all(role in evidence for role in role_words):
            continue
        query, aliases = relation_query(root, fields, schema)
        membership(query, path, schema, aliases, negative=True)
        yield query, 113, "absent_relation", {"output": root, "condition": target, "path": path}


def edge_is_forward(schema, edge):
    source, key, target, ref = edge
    return any((fk["from"].lower(), fk["table"].lower(), fk["to"].lower()) == (key, target, ref)
               for fk in schema.tables[source]["foreign_keys"])


def frequency_plan(tree, schema, fields, headings, question):
    """Use a named event relation as the grain of a grouped threshold or ranking."""
    rank = FREQUENCY.search(question)
    threshold = THRESHOLD.search(question)
    if not (rank or threshold):
        return
    where = tree.args.get("where")
    if where and (not threshold or len(list(where.find_all(exp.Literal))) != 1
                  or next(where.find_all(exp.Literal)).this != threshold[2]):
        return
    if tree.args.get("offset") or re.search(r"\b(?:both|or|without)\b", question, re.I):
        return
    roots = {t for _, t, _ in fields if t is not None}
    if len(roots) != 1 or not all(t or h.lower().strip() == "count(*)"
                                  for (_, t, _), h in zip(fields, headings, strict=True)):
        return
    root = next(iter(roots))
    words = set(tokens(question))
    if not entity_words(root, schema) <= words:
        return
    marker = rank or threshold
    after = tokens(question[marker.end():])[:6]
    if rank and rank[1]:
        after = tokens(rank[1]) + after
    names = set(after)
    role = {"start", "source", "origin", "departure"} if names & {"start", "started", "departed"} else (
        {"end", "destination", "arrival"} if names & {"end", "ended", "arrived"} else set())
    paths = []
    for table in sorted(schema.tables):
        if table == root:
            continue
        explicit = entity_words(table, schema) <= names
        for edge in sorted(set(schema.edges[root])):
            if edge[2] != table or edge_is_forward(schema, edge):
                continue
            key_roles = set(field_words(edge[3]))
            if role and not role & key_roles:
                continue
            # Bridge names and their event fields can supply the vocabulary
            # when the question uses the verb form rather than a table name.
            field_words_in_table = set().union(*(set(field_words(f)) for f in schema.columns[table]))
            context = (field_words_in_table - entity_words(root, schema)) & words
            if not explicit and not (entity_words(root, schema) <= entity_words(table, schema)
                                     and len(context) >= 2):
                continue
            paths.append((int(explicit), len(context), edge))
    paths.sort(reverse=True)
    if not paths or len(paths) > 1 and paths[0][:2] == paths[1][:2]:
        return
    edge = paths[0][2]
    query = exp.select().from_(physical_table(schema, root, "r0"))
    aliases = {root: "r0"}
    join_path(query, (edge,), schema, aliases)
    count = exp.Count(this=exp.Star())
    projected = [schema.column(t, f, aliases[t]) if t else count.copy() for _, t, f in fields]
    query.select(*projected, append=False, copy=False)
    query.group_by(schema.column(root, edge[1], "r0"), copy=False)
    if threshold:
        op = COMPARISONS[OPERATORS[threshold[1].lower()]]
        query.having(op(this=count, expression=exp.Literal.number(threshold[2])), copy=False)
    else:
        ascending = bool(re.match(r"(?:fewest|least|smallest|minimum)", rank[0], re.I))
        query.order_by(exp.Ordered(this=count, desc=not ascending, nulls_first=ascending), copy=False)
        query.limit(1, copy=False)
    yield query, 140, "relation_frequency", {"output": root, "event_edge": edge}


def via_membership(tree, schema, fields, question):
    """Compile positive membership when the original join repeats output entities."""
    sources = top_tables(tree, schema)
    targets = {t for _, t, _ in fields}
    if len(targets) != 1 or len(sources) < 2 or UNSUPPORTED.search(question):
        return
    match = re.search(r"\b(?:who|that|which)\s+(?:have|has|had|made|took|received|attended)\b", question, re.I)
    if match is None:
        return
    front, tail = (set(tokens(part)) for part in (question[:match.start()], question[match.end():]))
    root = next(iter(targets))
    if not entity_words(root, schema) <= front:
        return
    related = [t for t in schema.tables if t != root and entity_words(t, schema) <= tail]
    if len(related) != 1:
        return
    target = related[0]
    path = unique_path(schema, root, target, tail)
    if not path or all(edge_is_forward(schema, edge) for edge in path):
        return
    where = tree.args.get("where")
    atom = None
    if where:
        node = where.this
        if not isinstance(node, exp.EQ) or not isinstance(node.this, exp.Column) or not isinstance(node.expression, exp.Literal):
            return
        if not local_occurrences(tokens(question[match.end():]), node.expression.this):
            return
        binding = existing(node.this, sources, schema)
        if not binding or binding[1] != target:
            return
        atom = Filter(target, binding[2], node)
    query, aliases = relation_query(root, fields, schema)
    membership(query, path, schema, aliases, atom)
    if tree.args.get("distinct") or re.search(r"\b(?:unique|distinct)\b", question, re.I):
        query.distinct(copy=False)
    yield query, 129, "positive_membership", {"output": root, "condition": target, "path": path}


def explicit_order(tree, schema, fields, question):
    """Apply a direction explicitly specified for the fields of an existing sort."""
    order = tree.args.get("order")
    direction = re.search(r"\b(ascending|descending)\b", question, re.I)
    if order is None or direction is None or tree.args.get("limit"):
        return
    sources = top_tables(tree, schema)
    if not all(isinstance(key.this, exp.Column) and existing(key.this, sources, schema)
               for key in order.expressions):
        return
    desc = direction[1].lower() == "descending"
    if all(bool(key.args.get("desc")) == desc for key in order.expressions):
        return
    query = tree.copy()
    for key in query.args["order"].expressions:
        key.set("desc", desc)
        key.set("nulls_first", not desc)
    if [p.unalias().name.lower() for p in tree.selects] != [f for _, _, f in fields]:
        return
    yield query, 130, "explicit_order", {"descending": desc}


def compact_layout(tree, schema, fields, question):
    sources = top_tables(tree, schema)
    if not sources or tree.args.get("distinct"):
        return
    wordset = set(tokens(question))
    root = fields[0][1]
    if not entity_words(root, schema) & wordset:
        return
    # Preserve local predicates and group scope. Projection slots can be reordered
    # or trim an extra aggregate when the public headings specify the result.
    clauses = {k: tree.args[k] for k in ("where", "group", "having", "order", "limit", "offset") if tree.args.get(k)}
    bindings = {}
    for clause in clauses.values():
        if clause.find(exp.Select):
            return
        for column in clause.find_all(exp.Column):
            binding = existing(column, sources, schema)
            if binding is None:
                return
            bindings[(column.table.lower(), column.name.lower())] = binding[1:]
    targets = {t for _, t, _ in fields} | {t for t, _ in bindings.values()}
    unneeded = set(sources.values()) - targets
    if any(entity_words(t, schema) <= wordset for t in unneeded):
        return
    slots = [f for _, _, f in fields]
    layout_changed = [p.unalias().name.lower() for p in tree.selects] != slots
    if not unneeded and not layout_changed:
        return
    if tree.find(exp.Count) and unneeded:
        return
    # Rebuilding multi-table layout requires a single role-supported path.
    for present, path, _ in relation_plans(root, targets, schema, wordset, 3)[:1]:
        if unneeded and not (unneeded - present) and not layout_changed:
            continue
        query = exp.select().from_(physical_table(schema, root, "r0"))
        aliases = {root: "r0"}
        join_path(query, path, schema, aliases)
        query.select(*select_fields(fields, schema, aliases), append=False, copy=False)
        for key, clause in clauses.items():
            def rebind(node):
                if not isinstance(node, exp.Column):
                    return node
                table, field = bindings[node.table.lower(), node.name.lower()]
                return schema.column(table, field, aliases[table])
            query.set(key, clause.transform(rebind))
        yield query, 115 + 5 * bool(unneeded), "compact_layout", {"output": slots, "path": path}


def candidate_pool(obs, fallback):
    question = obs["task"]["question"]
    heads = obs["example"]["columns"]
    words = set(tokens(question))
    tables = next((e["observation"] for e in obs["history"] if e["tool"] == "schema"), None)
    raw = []
    seen = set()
    declared = Schema(tables) if tables is not None else None
    for sql in parent_sqls(obs, fallback)[:12]:
        tree = parse(sql)
        if not isinstance(tree, exp.Select) or tree.args.get("with"):
            continue
        key = tree.sql(dialect="sqlite")
        if key in seen:
            continue
        seen.add(key)
        schema = declared or observed_schema(obs, tree)
        simple = not any(tree.args.get(k) for k in ("group", "having", "order", "limit", "offset"))
        if simple:
            raw.extend(combined_entities(tree, schema, heads, question))
        variants = output_options(tree, heads, schema, words)
        for fields in variants[:4]:
            raw.extend(frequency_plan(tree, schema, fields, heads, question))
            if any(table is None for _, table, _ in fields):
                continue
            if simple:
                raw.extend(direct_roles(tree, schema, question, fields))
                raw.extend(scoped_filters(tree, schema, fields, question))
                raw.extend(via_membership(tree, schema, fields, question))
            raw.extend(explicit_order(tree, schema, fields, question))
            raw.extend(compact_layout(tree, schema, fields, question))
    return package({"history": []}, fallback, raw)[:12]
