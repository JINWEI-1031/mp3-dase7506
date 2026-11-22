"""Rebuild the scope of an absence condition from relation roles."""
import re

from sqlglot import exp

from repair_rules import sql_text
from step5x_edits import physical_table
from step7_semantics import ANTI, COMPARISON, bindings, join_path, shortest, terms, top_tables

GENERIC = {"any", "have", "used", "use", "named", "located", "name", "record",
           "that", "which", "with", "from", "who", "the", "for", "where"}


def anti_scope(tree, schema, question, arity):
    match = ANTI.search(question)
    if match is None or len(tree.selects) != arity:
        return
    # Include the negated verb for role columns such as advisor or employer.
    tail = question[match.start():]
    words = terms(tail) - GENERIC
    parents = top_tables(tree, schema)
    if not parents or not words or tree.args.get("group"):
        return
    outputs = list(tree.selects)
    columns = [c for p in outputs for c in p.find_all(exp.Column)]
    if not columns:
        return
    options = []
    literals = {p.expression.this: p for p in tree.find_all(*COMPARISON)
                if isinstance(p.this, exp.Column) and isinstance(p.expression, exp.Literal)
                and str(p.expression.this).strip("%").lower() in tail}
    # Quoted values can ground a fresh filter on the related role.
    for value in re.findall(r"""["']([^"']+)["']""", tail):
        literals.setdefault(value, None)
    for entity, _ in bindings(columns[0], parents, schema):
        names = []
        for c in columns:
            names.extend([(c, n) for t, n in bindings(c, parents, schema) if t == entity][:1])
        if len(names) != len(columns):
            continue
        for target in schema.tables:
            if target == entity:
                continue
            for path in shortest(schema, entity, target):
                table_words = terms(target)
                role_words = terms(path[-1][-1])
                overlap = len(words & table_words)
                if not overlap:
                    continue
                score = 5 * overlap + 2 * len(words & role_words) - len(path)
                filters = []
                for value, pred in literals.items():
                    candidates = []
                    for table in (edge[2] for edge in path):
                        for col in schema.columns[table]:
                            direct = pred is not None and pred.this.name.lower() == col
                            shared = terms(col) & (words - table_words)
                            binding_score = 4 * direct + 2 * len(shared)
                            if isinstance(value, str) and not str(value).isdigit():
                                binding_score += 2 * bool(terms(col) & {"name", "detail", "city", "location"})
                            if binding_score:
                                candidates.append((binding_score, table, col))
                    if candidates:
                        binding_score, table, col = max(candidates)
                        filters.append((table, col, value, pred))
                        score += binding_score
                if len(filters) == len(literals):
                    options.append((score, entity, path, names, filters))
    for _, entity, path, names, filters in sorted(options, key=lambda x: -x[0])[:3]:
        outer_alias = "e"
        _, left, first, right = path[0]
        inner_alias = "r0"
        alias = {first: inner_alias}
        body = exp.select(exp.Literal.number(1)).from_(physical_table(schema, first, inner_alias))
        join_path(body, path[1:], schema, alias)
        body.where(exp.EQ(this=schema.column(first, right, inner_alias),
                          expression=schema.column(entity, left, outer_alias)), copy=False)
        for table, name, value, pred in filters:
            value = str(value)
            literal = (exp.Literal.number(value) if pred is not None and not pred.expression.is_string
                       else exp.Literal.string(value.strip("%")))
            comparator = type(pred) if pred is not None and not isinstance(pred, exp.Like) else exp.EQ
            body.where(comparator(this=schema.column(table, name, alias[table]), expression=literal),
                       copy=False)
        output = [p.copy() for p in outputs]
        for src, (_, name) in zip([c for p in output for c in p.find_all(exp.Column)], names, strict=True):
            col = schema.column(entity, name, outer_alias)
            src.set("this", col.this)
            src.set("table", col.args["table"])
        result = exp.select(*output).from_(physical_table(schema, entity, outer_alias))
        result.where(exp.Not(this=exp.Exists(this=body)), copy=False)
        if tree.args.get("distinct"):
            result.distinct(copy=False)
        yield "negation_role_path", sql_text(result), {
            "outer_entity": entity, "foreign_keys": path,
            "filters": [(t, n) for t, n, _, _ in filters]}
