"""Bind explicit question constraints to fields visible at submission time."""
import re

from sqlglot import exp

from repair_rules import parse, sql_text
from mp3_sql.sqlite_tool import validate_sql
from step4_agent import identity
from step4_edits import Schema, key_edits
from step10_bindings import (
    ROLE, atoms, candidates, choice, field_words, tokens,
)
from step11_absence import parent_sqls
from step17_conflicts import ALPHA, alias_tables, role_tokens


ID_WORDS = {"id", "code", "key", "number"}
SORT = re.compile(r"\b(?:ordered|sorted|arranged)\s+(?:by|on)\s+([\w ]+)", re.I)
RECENT = re.compile(r"\b(most recent(?:ly)?|latest|earliest)\s+(\w+)", re.I)
TEMPORAL = {"date", "time", "year", "timestamp"}
STOP = {"ascend", "descend", "ascending", "descending", "alphabetical", "order"}


def stem(word):
    for suffix in ("ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[:-len(suffix)]
    return word


def observed_schema(obs, tree):
    """Only columns in executable, unambiguous SQL scopes provide bindings."""
    wanted = set(alias_tables(tree).values())
    fields = {table: {} for table in wanted}
    for event in obs["history"]:
        if event["tool"] != "check" or not event["observation"]["ok"]:
            continue
        query = parse(event["sql"])
        if query is None:
            continue
        for scope in query.find_all(exp.Select):
            sources = alias_tables(scope)
            for column in scope.find_all(exp.Column):
                if column.find_ancestor(exp.Select) is not scope:
                    continue
                table = sources.get(column.table.lower()) if column.table else (
                    next(iter(sources.values())) if len(sources) == 1 else None)
                if table in fields and not column.is_star:
                    fields[table][column.name.lower()] = {"name": column.name, "type": ""}
    return Schema([{"name": table, "columns": list(columns.values()), "foreign_keys": []}
                   for table, columns in sorted(fields.items())])


def order_request(tree, question):
    order = tree.args.get("order")
    if not order or len(order.expressions) != 1 or not isinstance(order.expressions[0].this, exp.Column):
        return None
    column = order.expressions[0].this
    explicit = ALPHA.search(question) or SORT.search(question)
    if explicit:
        phrase = explicit[1]
        fields = field_words(phrase)
        while fields and fields[-1] in STOP:
            fields.pop()
        if fields and (set(fields) != role_tokens(column.name) or (
                ALPHA.search(question) and set(field_words(column.name)) & ID_WORDS)):
            return "sort", fields, phrase
    recent = RECENT.search(question)
    if recent and tree.args.get("limit"):
        word = stem(recent[2].lower())
        if word not in {stem(w) for w in field_words(column.name)}:
            return "time", [word], recent[0]
    return None


def sort_candidate(tree, schema, sources, request, question):
    kind, fields, phrase = request
    old = tree.args["order"].expressions[0].this
    options = []
    for alias, table in sources.items():
        for field, meta in schema.columns[table].items():
            words = set(field_words(field))
            if kind == "time":
                rank = 4 if words & TEMPORAL and set(fields) & {stem(w) for w in words} else 0
            else:
                if ALPHA.search(question) and (words & ID_WORDS or re.match(
                        r"(?:int|real|num|dec|float|double)", meta["type"], re.I)):
                    continue
                rank = 5 if words == set(fields) else 3 if (
                    words and words <= set(fields) and set(fields) <= words | set(field_words(table))) else 0
            if rank and (field != old.name.lower() or old.table and alias != old.table.lower()):
                options.append((rank, alias, table, field))
    options.sort(reverse=True)
    if not options or len(options) > 1 and options[0][0] == options[1][0]:
        return
    _, alias, table, field = options[0]
    query = tree.copy()
    ordering = query.args["order"].expressions[0]
    ordering.set("this", schema.column(table, field, alias))
    if ALPHA.search(question):
        ordering.set("desc", False)
    elif kind == "time":
        ordering.set("desc", not phrase.lower().startswith("earliest"))
    yield query, 110, "ordering", {"phrase": phrase, "old": old.sql(), "field": field}


def predicate_requests(tree, question):
    for index, atom in enumerate(atoms(tree.args.get("where"))):
        literal = atom.expression
        if not isinstance(literal, exp.Literal) or not literal.is_string:
            continue
        value = re.escape(literal.this)
        explicit = re.search(
            rf"\b(?:with|whose|where|having)\s+(?:a |an |the )?([a-z_]+(?: [a-z_]+)?)"
            rf"\s+(?:of |is |equals? )?['\"]{value}['\"]", question, re.I)
        if explicit is None:
            explicit = re.search(
                rf"\b(?:whose|where)\s+(?:the )?([a-z_]+(?: [a-z_]+)?)"
                rf"\s+(?:is|equals?)\s+{value}(?!\w)", question, re.I)
        if explicit:
            role = set(field_words(explicit[1]))
            if role and not role <= set(field_words(atom.this.name)):
                yield index, literal.this, role


def output_requests(tree, headings, question):
    if tree.find(exp.AggFunc, exp.Star) or tree.args.get("group") or len(tree.selects) != len(headings):
        return
    words = set(tokens(question))
    for index, (output, heading) in enumerate(zip(tree.selects, headings)):
        field = output.unalias()
        wanted = set(field_words(heading))
        if (isinstance(field, exp.Column) and wanted and wanted <= words
                and wanted != set(field_words(field.name))):
            yield index, heading, wanted


def role_request(tree, question):
    words = set(tokens(question))
    return bool(words & ROLE and any(isinstance(j.args.get("on"), exp.EQ)
                                     for j in tree.args.get("joins") or []))


def requests(obs, tree):
    if not isinstance(tree, exp.Select) or len(list(tree.find_all(exp.Select))) != 1:
        return {}
    question = obs["task"]["question"]
    return {kind: request for kind, request in {
        "ordering": order_request(tree, question),
        "predicate": list(predicate_requests(tree, question)),
        "projection": list(output_requests(tree, obs["example"]["columns"], question)),
        "relation": role_request(tree, question),
    }.items() if request}


def proposals(obs, tree, schema, evidence, relation_control=False):
    question = obs["task"]["question"]
    words = tokens(question)
    sources = {a: t for a, t in alias_tables(tree).items() if t in schema.tables}
    if "ordering" in evidence:
        yield from sort_candidate(tree, schema, sources, evidence["ordering"], question)
    for index, literal, role in evidence.get("predicate", []):
        atom = atoms(tree.args.get("where"))[index]
        bound = choice(atom, sources, schema, words)
        if bound and role <= set(field_words(bound[2])):
            alias, table, field, _ = bound
            query = tree.copy()
            atoms(query.args.get("where"))[index].this.replace(schema.column(table, field, alias))
            yield query, 95, "predicate", {"literal": literal, "old": atom.this.sql(), "field": field}
    for index, heading, wanted in evidence.get("projection", []):
        matches = [(a, t, c) for a, t in sources.items() for c in schema.columns[t]
                   if set(field_words(c)) == wanted]
        if len(matches) == 1:
            alias, table, field = matches[0]
            query = tree.copy()
            output = query.selects[index]
            output.unalias().replace(schema.column(table, field, alias)) if isinstance(
                output, exp.Alias) else output.replace(schema.column(table, field, alias))
            yield query, 100, "projection", {"heading": heading, "field": field}
    if evidence.get("relation") or relation_control:
        for _, sql, proof in key_edits(tree, schema):
            role = set(words) & ROLE
            old = set(tokens(proof["from"]))
            new = set(tokens(" ".join(proof["foreign_key"][1::2])))
            if relation_control or (new & role) - old:
                yield parse(sql), 90, "relation", proof


def package(obs, fallback, raw):
    seen = {identity(e["sql"]) for e in obs["history"] if "sql" in e} | {identity(fallback)}
    result = []
    for query, rank, rule, proof in sorted(raw, key=lambda x: -x[1]):
        sql = sql_text(query)
        key = identity(sql)
        if key in seen:
            continue
        try:
            validate_sql(sql)
        except ValueError:
            continue
        seen.add(key)
        result.append({"sql": sql, "priority": rank, "rule": rule, "evidence": proof})
    return result


def public_pool(obs, fallback, tree, schema, evidence, relation_control=False):
    return package(obs, fallback, proposals(obs, tree, schema, evidence, relation_control))


def failed_pool(obs, fallback, schema, joint):
    if joint:
        pool, _ = candidates(obs, schema, ("role", "coupled", "projection"), True, (fallback,))
        # Projection-only edits remain in the existing-candidate control.
        pool = [c for c in pool if c["family"] != "projection"]
        parents = [parse(s) for s in dict.fromkeys([fallback, *parent_sqls(obs, fallback)])]
        extras = []
        for tree in parents:
            evidence = requests(obs, tree)
            if not evidence:
                continue
            extras.extend(p for p in proposals(obs, tree, schema, evidence) if p[2] != "projection")
        extra = package(obs, fallback, extras)
        return sorted([*pool, *(c for c in extra if c["sql"] not in {p["sql"] for p in pool})],
                      key=lambda c: -c["priority"])
    return package(obs, fallback, [(tree, 0, "existing_candidate", {})
                                   for sql in parent_sqls(obs, fallback) if (tree := parse(sql)) is not None])

