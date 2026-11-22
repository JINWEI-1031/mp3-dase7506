"""Change row multiplicity along a question-bound, declared witness path."""
import re

from sqlglot import exp

from repair_rules import sql_text
from step32_plan import compiled, tree_plan
from step4_edits import Schema
from step5x_edits import physical_table
from step36.frequency import terms as frequency_terms

RELATIVE = re.compile(r"\b(?:who|that|which)\s+(?:have|has|had|made|are|were|some|teach(?:es)?|resulted)\b", re.I)
DIRECT = re.compile(r"^\s*(?:which|what)\s+(.+?)\s+(?:have|has|had|made|are trained in)\b", re.I)
UNSUPPORTED = re.compile(
    r"\b(?:without|not|never|no|both|either|or|most|fewest|highest|lowest|how many|"
    r"at least|at most|each|every|all|distinct|unique|different|ordered|ascending|descending)\b", re.I)
CLAUSES = ("group", "having", "order", "limit", "offset", "with", "qualify")


def terms(text):
    return frequency_terms(re.sub(r"\b(\w+us)es\b", r"\1", text, flags=re.I))


def intent(question):
    if UNSUPPORTED.search(question):
        return None
    relative = RELATIVE.search(question)
    direct = DIRECT.match(question)
    if relative and (not direct or relative.start() < direct.end(1)):
        return terms(question[:relative.start()]), terms(question[relative.start():].split("?")[0])
    if direct:
        return terms(direct[1]), terms(question[direct.end():].split("?")[0])


def shape(parent, headings):
    if not isinstance(parent, exp.Select) or not parent.args.get("joins"):
        return False
    return (len(list(parent.find_all(exp.Select))) == 1
            and not any(parent.args.get(key) for key in CLAUSES)
            and len(parent.expressions) == len(headings)
            and all(isinstance(e.unalias(), exp.Column) and not e.unalias().is_star
                    for e in parent.expressions))


def bind_column(column, schema, sources):
    if not isinstance(column, exp.Column):
        return None
    name, alias = column.name.casefold(), column.table.casefold()
    bindings = [(table, name) for table, label in sources.items()
                if name in schema.columns[table] and (not alias or alias == label.casefold())]
    return bindings[0] if len(bindings) == 1 else None


def join_graph(parent, schema, sources):
    edges = []
    for join in parent.args["joins"]:
        on = join.args.get("on")
        if join.args.get("side") or join.args.get("kind") not in (None, "INNER") or not isinstance(on, exp.EQ):
            return None
        a = bind_column(on.this, schema, sources)
        b = bind_column(on.expression, schema, sources)
        if not a or not b or (*a, *b) not in schema.edges[a[0]]:
            return None
        edges.append((*a, *b))
    return edges


def paths_from(root, edges):
    paths = {root: []}
    pending = list(edges)
    while pending:
        for index, (a, ak, b, bk) in enumerate(pending):
            if (a in paths) != (b in paths):
                if b in paths:
                    a, ak, b, bk = b, bk, a, ak
                paths[b] = [*paths[a], (a, ak, b, bk)]
                pending.pop(index)
                break
        else:
            return {}
    return paths


def analyze(c, parent):
    roles = intent(c["question"])
    if not roles:
        return [], {"stage": "not_triggered", "reason": "No supported positive event qualification"}
    if not shape(parent, c["columns"]):
        return [], {"stage": "unexpressible", "reason": "Need simple projected columns and an inner event join without grouped, nested or ordered scope"}
    # Bind against the public display slots before asking for schema.
    if [e.alias_or_name.casefold() for e in parent.expressions] != [h.casefold() for h in c["columns"]]:
        return [], {"stage": "binding_failed", "reason": "Draft projection names or order differ from public columns"}
    request = {"bind": "output primary key and event JOIN foreign-key direction",
               "output_terms": sorted(roles[0]), "witness_terms": sorted(roles[1]),
               "projection": [sql_text(e) for e in parent.expressions],
               "joins": [sql_text(join) for join in parent.args["joins"]],
               "predicate_scope": sql_text(parent.args["where"]) if parent.args.get("where") else None}
    if c["schema"] is None:
        return [], {"stage": "binding_failed", "reason": "Declared key path has not been observed", "request": request}
    schema = Schema(c["schema"])
    sources = schema.sources(parent)
    bound = [bind_column(e.unalias(), schema, sources) for e in parent.expressions]
    edges = join_graph(parent, schema, sources)
    if not edges or not all(bound):
        return [], {"stage": "binding_failed", "reason": "Unbound projected column or JOIN is not a declared single-column FK"}
    roots = []
    for root in sources:
        keys = [col for col, attr in schema.columns[root].items() if attr["pk"]]
        if len(keys) != 1 or not terms(root) & roles[0]:
            continue
        key = keys[0]
        equivalent_keys = {(t, k) for a, ak, b, bk in edges
                           for r, rk, t, k in ((a, ak, b, bk), (b, bk, a, ak)) if (r, rk) == (root, key)}
        if all(t == root or (t, col) in equivalent_keys for t, col in bound):
            roots.append((root, key))
    if len(roots) != 1:
        return [], {"stage": "binding_failed", "reason": "Question and projection do not identify one output entity with a single declared key"}
    root, key = roots[0]
    paths = paths_from(root, edges)
    if set(paths) != sources.keys():
        return [], {"stage": "binding_failed", "reason": "JOINs are not a connected acyclic key path"}
    where = parent.args.get("where")
    predicates = [bind_column(col, schema, sources) for col in where.find_all(exp.Column)] if where else []
    if any(p is None for p in predicates):
        return [], {"stage": "binding_failed", "reason": "An event predicate column cannot be qualified uniquely"}
    if where and any(isinstance(e, (exp.Or, exp.Not))
                     or isinstance(e, exp.Func) and not isinstance(e, exp.And)
                     for e in where.walk()):
        return [], {"stage": "unexpressible", "reason": "Disjunction, negation or functional predicate exceeds the positive witness scope"}
    events = [t for t in paths if t != root and
              ((terms(t) - terms(root)) | set().union(*(terms(col) for tbl, col in predicates if tbl == t))) & roles[1]]
    if not events:
        return [], {"stage": "binding_failed", "reason": "No question-named witness role on the draft path"}
    fk_children = {(t, k, f["table"].casefold(), f["to"].casefold())
                   for t in sources for f in schema.tables[t]["foreign_keys"] for k in [f["from"].casefold()]}
    branching = any((b, bk, a, ak) in fk_children
                    and (a, ak, b, bk) not in fk_children
                    and ([col for col, attr in schema.columns[b].items() if attr["pk"]] != [bk])
                    for event in events for a, ak, b, bk in paths[event])
    if not branching:
        return [], {"stage": "unexpressible", "reason": "No declared one-to-many step on a named event path"}
    basis = {"source": "event_grain", "eligible": True, "output_entity": root, "output_key": key,
             "key_paths": paths, "witness_roles": events, "predicate_scope": request["predicate_scope"],
             "predicate_bindings": predicates, "grouping_keys": [], "measure": None,
             "output_columns": bound}
    alias = "s38_output"
    while alias.casefold() in {value.casefold() for value in sources.values()}:
        alias += "_"
    witness = parent.copy().select(exp.Literal.number(1), append=False)
    witness.set("distinct", None)
    witness.where(exp.EQ(this=schema.column(root, key, sources[root]),
                         expression=schema.column(root, key, alias)), copy=False)
    projection = []
    for original, (table, col) in zip(parent.expressions, bound, strict=True):
        field = schema.column(root, col if table == root else key, alias)
        if original.alias or field.name.casefold() != original.alias_or_name.casefold():
            field = exp.alias_(field, original.alias_or_name, quoted=True)
        projection.append(field)
    root_rows = exp.select(*projection).from_(physical_table(schema, root, alias)).where(exp.Exists(this=witness))
    toggled = parent.copy()
    toggle_mode = "one row per joined witness tuple" if parent.args.get("distinct") else "one row per projected value tuple"
    toggled.set("distinct", None if parent.args.get("distinct") else exp.Distinct())
    variants = [(root_rows, "one row per qualifying output key"), (toggled, toggle_mode)]
    if parent.args.get("distinct"):
        variants.reverse()
    return list(compiled(c, parent, (tree_plan(tree, basis | {"row_multiplicity": mode})
                                     for tree, mode in variants))), {"stage": "bound", "reason": None}


