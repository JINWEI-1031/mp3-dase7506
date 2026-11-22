"""Bind a question-qualified negative witness to the output entity's key."""
import re

from sqlglot import exp

from repair_rules import parse, sql_text
from step4_edits import Schema

NEGATIVE = re.compile(r"\b(?:do(?:es)? not|don['’]t|doesn['’]t|never)\s+(\w+)\s+(.+)", re.I)
ROLE_WORDS = {
    "advise": {"advise", "advisor", "adviser"},
    "authorize": {"authorize", "authorized", "authorizing", "authorization", "authorizer", "authorise", "authorised", "authorising", "authorisation", "authoriser"},
    "pass": {"pass", "passage", "station"},
}
STOP = {"a", "an", "any", "by", "for", "in", "of", "the", "to", "be"}


def words(value):
    return {word.removesuffix("s") for word in re.findall(r"[a-z]+", value.casefold())} - STOP


def source(tree):
    relation = tree.args.get("from")
    return relation.this if relation and isinstance(relation.this, exp.Table) else None


def table(schema, name, alias):
    return exp.Table(this=exp.to_identifier(schema.tables[name]["name"], quoted=True)).as_(alias)


def affinity(column):
    kind = column.get("type", "").upper()
    if "INT" in kind:
        return "integer"
    if any(token in kind for token in ("CHAR", "CLOB", "TEXT")):
        return "text"
    return kind


def bind_path(schema, root, key, target, role):
    paths = []
    for path in schema.path_options({root: "entity"}, target, 2):
        if path and path[0][1] == key:
            labels = words(" ".join(part for edge in path for part in edge[1:]))
            if labels & role:
                paths.append((tuple(path), "declared_foreign_keys"))
    # Some teaching schemas expose a named role reference without declaring an FK.
    for field, column in schema.columns[target].items():
        if not words(field) & role or field == key or column.get("pk"):
            continue
        left, right = affinity(schema.columns[root][key]), affinity(column)
        if left == right and left in ("integer", "text"):
            paths.append((((root, key, target, field),), "named_type_compatible_reference"))
    declared = {item for item in paths if item[1] == "declared_foreign_keys"}
    return declared or set(paths)


def analyze(c):
    q = c["question"]
    match = NEGATIVE.search(q)
    if not match:
        return None, {"stage": "not_triggered"}
    verb, tail = match[1].casefold(), match[2]
    rejected = {"stage": "binding_failed", "role": verb, "witness_phrase": tail}
    if re.search(r"\b(?:and|or|except|every|all)\b", tail, re.I):
        return None, rejected | {"reason": "mixed or ambiguous witness scope"}
    draft = parse(c["draft"])
    root_src = source(draft) if isinstance(draft, exp.Select) else None
    if not root_src or any(draft.args.get(name) for name in ("group", "having", "order", "limit", "with")):
        return None, rejected | {"reason": "plain output entity unavailable in draft"}
    outputs = [p.unalias() for p in draft.expressions]
    if [p.alias_or_name.casefold() for p in draft.expressions] != [h.casefold() for h in c["columns"]] or not all(
        isinstance(p, exp.Column) and not p.is_star and (
            not p.table or p.table.casefold() == root_src.alias_or_name.casefold()) for p in outputs
    ):
        return None, rejected | {"reason": "public projection cannot be bound to the draft root"}
    root = root_src.name.casefold()
    if not words(root) & words(q[:match.start()]):
        return None, rejected | {"reason": "question does not independently name the draft output entity"}
    # Only propose a binding when the draft independently exposes a witness relation.
    other = {t.name.casefold() for t in draft.find_all(exp.Table)} - {root}
    if not other:
        return None, rejected | {"reason": "draft supplies no witness relation"}
    if c["schema"] is None:
        return None, rejected | {"reason": "schema required",
                                 "request": {"output_entity": root, "output_key": "declared primary key",
                                             "witness_role": verb, "witness_phrase": tail,
                                             "needed": "role-specific key path and witness predicate owners"}}
    schema = Schema(c["schema"])
    cols = schema.columns.get(root, {})
    keys = [name for name, column in cols.items() if column.get("pk")]
    if len(keys) != 1 or not all(p.name.casefold() in cols for p in outputs):
        return None, rejected | {"reason": "unique declared output key or field missing"}
    key = keys[0]
    role = ROLE_WORDS.get(verb, {verb})
    witness_words = words(tail)
    direct = {t for t in other if t in schema.tables and words(t) & witness_words}
    # A mention of the entity may be explicit even when the draft followed a different event path.
    named = {t for t in schema.tables if t != root and words(t) <= witness_words and words(t)}
    targets = named or direct
    draft_fields = {c.name.casefold() for c in draft.find_all(exp.Column)}
    bindings = {(path, evidence) for t in targets for path, evidence in bind_path(schema, root, key, t, role)
                if evidence == "declared_foreign_keys" or t in other or
                any(k in draft_fields and col.get("pk") for k, col in schema.columns[t].items())}
    if len(bindings) != 1:
        return None, rejected | {"reason": "witness relation or role-specific key path is not unique",
                                 "targets": sorted(targets), "path_count": len(bindings)}
    (path, evidence), = bindings
    target = path[-1][2]
    anchors = []
    for atom in draft.find_all(exp.EQ):
        if isinstance(atom.this, exp.Column) and isinstance(atom.expression, exp.Literal) and re.search(
            rf"(?<!\w){re.escape(atom.expression.this)}(?!\w)", tail, re.I
        ):
            field = atom.this.name.casefold()
            owners = [t for t in {edge[2] for edge in path} if field in schema.columns[t]]
            if owners != [target]:
                return None, rejected | {"reason": "witness literal owner is ambiguous", "field": field}
            anchors.append((field, atom.expression))
    if re.search(r"\b(?:located|named|called|with|in)\b|\d|['\"]", tail) and not anchors:
        return None, rejected | {"reason": "witness qualifier has no grounded field and literal"}
    query = exp.select(*(schema.column(root, p.name.casefold(), "entity") for p in outputs)).from_(
        table(schema, root, "entity"))
    aliases = {root: "entity"}
    child = exp.select("1")
    for index, (left, a, right, b) in enumerate(path):
        alias = f"w{index}"
        aliases[right] = alias
        condition = exp.EQ(this=schema.column(left, a, aliases[left]), expression=schema.column(right, b, alias))
        if index == 0:
            child = child.from_(table(schema, right, alias)).where(condition)
        else:
            child = child.join(table(schema, right, alias), on=condition)
    predicates = [{"table": target, "role": "excluded witness", "scope": "inner", "column": field,
                   "operator": "eq", "literal": value.this, "is_string": value.is_string}
                  for field, value in anchors]
    for field, value in anchors:
        child = child.where(exp.EQ(this=schema.column(target, field, aliases[target]), expression=value.copy()))
    plan = {"output_entity": root, "output_key": key, "output_fields": [p.name for p in outputs],
            "event_role": verb, "witness": target, "predicate_bindings": predicates,
            "comparison_population": [edge[2] for edge in path], "key_path": path, "key_evidence": evidence,
            "comparison_boundary": "zero qualifying witnesses", "correlation_scope": "inner witness to outer key",
            "grouping": [], "measure": "existence of qualifying witness", "multiplicity": "one per output key"}
    sql = sql_text(query.where(exp.Not(this=exp.Exists(this=child))))
    return {"sql": sql, "basis": {"eligible": True, "source": "negative_witness", "plan": plan}}, {"stage": "bound"}


