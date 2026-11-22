"""Bind predicates from the submitted query or from prior successful checks."""
from itertools import islice

from sqlglot import exp

from repair_rules import parse, sql_text
from step4_agent import identity
from step4_edits import Schema
from step10_bindings import local_occurrences, tokens
from step19_grounding import observed_schema
from step32_plan import decompose, tree_plan
from step33.relations import plan_options, request_binding
from step34.quantities import checked_source, discover, plans

POOL_LIMIT = 8


def outer_tables(tree):
    sources = [tree.args["from"].this] if tree.args.get("from") else []
    sources.extend(join.this for join in tree.args.get("joins", []))
    if (not sources or any(not isinstance(t, exp.Table) or t.db or t.catalog for t in sources)
            or len({t.name.casefold() for t in sources}) != len(sources)):
        return {}
    return {t.alias_or_name.casefold(): t.name.casefold() for t in sources}


def donors(obs, baseline):
    seen = {identity(baseline)}
    for i, event in enumerate(obs["history"], 1):
        if event["tool"] != "check" or not event["observation"]["ok"]:
            continue
        sql = event["sql"]
        key = identity(sql)
        if key in seen:
            continue
        seen.add(key)
        tree = parse(sql)
        if isinstance(tree, exp.Select) and tree.args.get("where") is not None:
            yield i, sql, tree


def read_binding(column, scopes, schema):
    """Resolve a column from the innermost scope out using visible metadata."""
    if column.is_star or column.db or column.catalog:
        return None
    alias = column.table.casefold()
    for position, scope in enumerate(scopes):
        if alias and alias in scope:
            table = scope[alias]
            return (position, table, column.name) if column.name.casefold() in schema.columns.get(table, {}) else None
        if not alias:
            hits = [table for table in scope.values() if column.name.casefold() in schema.columns.get(table, {})]
            if hits:
                return (position, hits[0], column.name) if len(hits) == 1 else None
    return None


def transplant_where(parent, donor, schema):
    """Keep the carrier query. Move only the donor predicate and bind its free columns."""
    if not isinstance(parent, exp.Select) or not decompose(parent) or not decompose(donor):
        return None
    parent_roles = {table: alias for alias, table in outer_tables(parent).items()}
    donor_roles = outer_tables(donor)
    if not parent_roles or not donor_roles:
        return None
    tree = donor.copy()
    predicate = tree.args.get("where")
    if predicate is None:
        return None
    for column in list(predicate.find_all(exp.Column)):
        local = []
        scope = column.find_ancestor(exp.Select)
        while scope is not None and scope is not tree:
            tables = outer_tables(scope)
            if not tables:
                return None
            local.append(tables)
            scope = scope.find_ancestor(exp.Select)
        if scope is not tree:
            return None
        binding = read_binding(column, [*local, donor_roles], schema)
        if binding is None:
            return None
        level, table, name = binding
        if level == len(local):
            alias = parent_roles.get(table)
            if not alias or any(alias in scope for scope in local):
                return None
            column.replace(exp.column(name, table=alias, quoted=True))
    return parent.copy().where(predicate.this.copy(), append=False)


def topical(where, question):
    values = [v.this for v in where.find_all(exp.Literal)
              if not (v.is_int and v.this == "1" and isinstance(v.parent, exp.Select))]
    q = tokens(question)
    return bool(values) and all(local_occurrences(q, str(value).strip("%")) for value in values)


def from_checked_predicate(obs, sql):
    # Round 34 could also incorporate a predicate from a failed quantified
    # SQL sketch. Enforce a successful anchor in both arms of this contrast.
    return [proof for proof in discover(obs, sql)
            if proof[0] != "population" or not proof[2].needs_schema]


def proposal_inputs(obs, c, parent, public, use_donors):
    sql = c["baseline"]
    base_proofs = from_checked_predicate(obs, sql)
    visible = c["schema"]
    requested = []
    if any(item[-1] for item in base_proofs):
        requested.append("baseline quantified role")
    if checked_source(obs, sql) and request_binding(c, parent, public):
        requested.append("baseline graph binding")
    if not use_donors or not isinstance(parent, exp.Select):
        return base_proofs, [], requested
    bound = []
    for ordinal, donor_sql, donor in donors(obs, sql):
        proofs = from_checked_predicate(obs, donor_sql)
        # A donor is used only to rebuild a predicate. All other submitted
        # slots remain the carrier's, even when the donor displayed other fields.
        predicate = donor.args["where"]
        useful = bool(proofs or topical(predicate, c["question"]) or
                      (not public and predicate.find(exp.Is)))
        if not useful:
            continue
        schema = Schema(visible) if visible is not None else observed_schema(obs, donor)
        if visible is None and proofs and any(item[-1] for item in proofs):
            requested.append("checked donor quantified role")
        direct = transplant_where(parent, donor, schema)
        normalized_parent = transplant_where(parent, parent, schema) if direct is not None else None
        if normalized_parent is not None and identity(sql_text(normalized_parent)) == identity(sql_text(direct)):
            direct = None
        if direct is not None or proofs:
            bound.append((ordinal, donor_sql, direct, proofs, schema))
    return base_proofs, bound, requested


def proposals(obs, c, parent, public, inputs):
    base_proofs, donor_inputs, _ = inputs
    schema = c["schema"]
    yield from plans(obs, base_proofs, schema)
    if schema is not None and checked_source(obs, c["baseline"]):
        for record in islice(plan_options(c, parent, public, schema), 12):
            if record:
                basis = record["basis"]
                yield record | {"basis": basis | {"eligible": bool(basis["advantage"])}}
    for ordinal, donor_sql, direct, proofs, visible in donor_inputs:
        for record in plans(obs, proofs, schema):
            if record is None:
                continue
            candidate = parse_plan(record["plan"])
            transplanted = transplant_where(parent, candidate, visible) if candidate is not None else None
            if transplanted is not None:
                basis = record["basis"] | {"source": "checked_donor", "donor_call": ordinal,
                                           "donor_sql": donor_sql,
                                           "eligible": record["basis"]["eligible"] and not public}
                yield tree_plan(transplanted, basis)
        if direct is not None:
            yield tree_plan(direct, {"kind": "direct_checked_predicate", "eligible": not public,
                                     "source": "checked_donor", "donor_call": ordinal, "donor_sql": donor_sql})
            if schema is not None:
                for record in islice(plan_options(c, direct, public, schema), 8):
                    if record:
                        basis = record["basis"] | {"source": "checked_donor", "donor_call": ordinal,
                                                   "donor_sql": donor_sql,
                                                   "eligible": bool(record["basis"]["advantage"]) and not public}
                        yield record | {"basis": basis}


def parse_plan(plan):
    from step32_plan import compile_plan

    # Generated plans come from a SELECT AST. Use the source output width,
    # which may differ from the carrier width, before taking its WHERE.
    output = parse("SELECT " + plan["output"])
    sql = compile_plan(plan, len(output.expressions)) if isinstance(output, exp.Select) else None
    return parse(sql) if sql is not None else None


