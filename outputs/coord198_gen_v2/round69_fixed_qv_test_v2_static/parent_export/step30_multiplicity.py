"""Change the predicate scope while keeping output-grain evidence separate."""
import re

from sqlglot import exp

from repair_rules import parse
from step4_edits import Schema
from step7_semantics import top_tables
from step10_bindings import existing, field_words
from step19_grounding import observed_schema, package
from step29_scope import scope_signal, separate_witnesses, single_witness, supported_subject


ONE_ROW = re.compile(r"\bone\s+(?:result\s+)?row\s+(?:per|for each)\s+([\w -]+)", re.I)
EACH_ONCE = re.compile(r"\beach\s+([\w -]+?)\s+(?:only\s+)?once\b", re.I)


def one_row_evidence(question):
    """Bind explicit row-count wording to an entity noun, before schema."""
    match = ONE_ROW.search(question) or EACH_ONCE.search(question)
    if not match:
        return ()
    stop = re.split(r"\b(?:who|with|that|which|where|whether|from)\b", match[1], maxsplit=1, flags=re.I)[0]
    return tuple(field_words(stop))


def scope_plan(obs, sql, arm):
    question = obs["task"]["question"]
    proof = one_row_evidence(question) if arm == "gated" else ()
    if proof:
        # Remind the shared recognizer that this use of 'each' is about
        # output rows, not a grouped aggregate.
        normalized = re.sub(r"\bone\s+(?:result\s+)?row\s+(?:per|for each)\s+|\beach\s+",
                            "", question, count=1, flags=re.I)
        observed = {**obs, "task": {**obs["task"], "question": normalized}}
    else:
        observed = obs
    scope = scope_signal(observed, sql)
    if not scope or (scope.kind == "single_witness" and not proof):
        return None
    if scope.kind != "single_witness":
        tree = parse(sql)
        if (not tree.args.get("joins") or tree.args.get("order")
                or not all(isinstance(p.unalias(), exp.Column) for p in tree.selects)):
            return None
        visible = observed_schema(obs, tree)
        sources = top_tables(tree, visible)
        roots = [alias for alias, table in sources.items() if supported_subject(scope.subject, table)]
        if len(roots) != 1 or already_independent(tree, roots[0], scope):
            return None
    return scope, proof


def already_independent(tree, subject_alias, scope):
    """Recognize an existing same-field witness correlated directly to the subject."""
    where = tree.args.get("where")
    if scope.kind != "separate_positive" or not where or not isinstance(where.this, exp.And):
        return False
    for row, child in ((where.this.this, where.this.expression), (where.this.expression, where.this.this)):
        if (not isinstance(row, exp.EQ) or not isinstance(row.this, exp.Column)
                or not isinstance(row.expression, exp.Literal) or not isinstance(child, exp.Exists)):
            continue
        nested = child.this
        if not isinstance(nested, exp.Select) or not nested.args.get("where") or nested.args.get("joins"):
            continue
        source = nested.args.get("from")
        if not source or not isinstance(source.this, exp.Table):
            continue
        local_alias = source.this.alias_or_name.lower()
        outer_table = next((t for t in tree.find_all(exp.Table)
                            if t.find_ancestor(exp.Select) is tree and t.alias_or_name.lower() == row.this.table.lower()), None)
        if not outer_table or outer_table.name.lower() != source.this.name.lower():
            continue
        inner = nested.args["where"].this
        if not isinstance(inner, exp.And) or not all(isinstance(n, exp.EQ) for n in (inner.this, inner.expression)):
            continue
        for eq, value in ((inner.this, inner.expression), (inner.expression, inner.this)):
            if not all(isinstance(c, exp.Column) for c in (eq.this, eq.expression, value.this)):
                continue
            if ({eq.this.table.lower(), eq.expression.table.lower()} == {subject_alias, local_alias}
                    and value.this.table.lower() == local_alias and value.this.name.lower() == row.this.name.lower()
                    and isinstance(value.expression, exp.Literal)
                    and {row.expression.this, value.expression.this} == set(scope.terms)):
                return True
    return False


def row_atoms(node, scope, schema, sources, condition_field):
    """Retain qualifying outer records from the old output relation."""
    if isinstance(node, exp.Paren):
        return row_atoms(node.this, scope, schema, sources, condition_field)
    if isinstance(node, exp.And):
        left = row_atoms(node.this, scope, schema, sources, condition_field)
        right = row_atoms(node.expression, scope, schema, sources, condition_field)
        if left is None or right is None:
            return None
        # Two different equality terms joined by AND on the same field
        # don't establish a usable old output-row selection.
        if left and right:
            return None
        return left or right
    if isinstance(node, exp.Or):
        left = row_atoms(node.this, scope, schema, sources, condition_field)
        right = row_atoms(node.expression, scope, schema, sources, condition_field)
        if not left or not right:
            return None
        return [exp.or_(left[0], right[0])]
    if isinstance(node, exp.Exists):
        return []
    if not isinstance(node, (exp.EQ, exp.NEQ, exp.In)) or not isinstance(node.this, exp.Column):
        return None
    current = existing(node.this, sources, schema)
    if not current or current[1:] != condition_field:
        return None
    if isinstance(node, exp.In):
        if (scope.kind == "separate_positive" and not node.args.get("query")
                and all(isinstance(v, exp.Literal) for v in node.expressions)
                and {v.this for v in node.expressions} == set(scope.terms)):
            return [node.copy()]
        return None
    literal = node.expression
    if not isinstance(literal, exp.Literal) or literal.this not in scope.terms:
        return None
    if isinstance(node, exp.EQ):
        if scope.kind == "separate_positive" or literal.this == scope.terms[0]:
            return [node.copy()]
    elif scope.kind == "positive_negative" and literal.this == scope.terms[1]:
        # Root-level NOT EXISTS excludes this value for every output row.
        # The surviving positive equality also preserves the old NULL filter.
        return []
    return None


def predicate_only(obs, tree, schema, scope):
    sources = top_tables(tree, schema)
    for draft, rank, rule, evidence in separate_witnesses(obs, tree, schema, scope):
        if rule != "independent_entity_witnesses":
            continue
        subject = evidence["subject"]
        roots = [alias for alias, table in sources.items() if table == subject]
        if len(roots) != 1:
            continue
        alias = roots[0]
        predicates = draft.args["where"].this.copy()
        subqueries = list(predicates.find_all(exp.Select))
        if any(alias in top_tables(sub, schema) for sub in subqueries):
            continue
        where = tree.args["where"]
        retained = row_atoms(where.this, scope, schema, sources, evidence["condition_field"])
        if not retained:
            continue
        for column in predicates.find_all(exp.Column):
            if column.table.lower() == "e0":
                column.set("table", exp.to_identifier(alias, quoted=True))
        rewritten = tree.copy()
        rewritten.set("where", None)
        rewritten.where(*retained, predicates, copy=False)
        yield rewritten, rank, "independent_witnesses_keep_output", {
            **evidence, "output_select_joins_and_distinct": "preserved",
            "qualifying_output_rows": [condition.sql(dialect="sqlite") for condition in retained],
        }


def candidate_pool(obs, sql, tables, plan):
    tree = parse(sql)
    schema = Schema(tables)
    scope, proof = plan
    raw = []
    if scope.kind == "single_witness":
        for draft, rank, rule, evidence in single_witness(tree, schema, scope):
            if set(field_words(evidence["subject"])) <= set(proof):
                raw.append((draft, rank, "explicit_one_row_semijoin", {**evidence, "one_row_evidence": proof}))
    else:
        raw.extend(predicate_only(obs, tree, schema, scope))
    return package({"history": []}, sql, raw)
