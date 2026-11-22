"""Sort-key, predicate-role and join repairs justified by observable conflicts."""
import re

from sqlglot import exp

from mp3_sql.sqlite_tool import validate_sql
from repair_rules import parse, sql_text
from step4_agent import identity
from step4_edits import key_edits, simple
from step7_semantics import terms, top_tables
from step10_bindings import existing
from step11_absence import parent_sqls

ALPHA = re.compile(r"\balphabetical(?: order)?\s+(?:of|by)\s+(?:their |the )?([a-z_ ]+)", re.I)
FREQUENCY = re.compile(r"\b(?:most (?:number of )?times|most (?:frequently|often))\b", re.I)
NEGATIVE = re.compile(r"\b(?:without any|(?:do(?:es)? not|don['’]t|doesn['’]t) have any)\b", re.I)


def role_tokens(value):
    return terms(value) - {"id", "code", "number", "key"}


def order_edits(tree, question, schema, parents):
    order = tree.args.get("order")
    if not order or len(order.expressions) != 1 or not isinstance(order.expressions[0].this, exp.Column):
        return
    column = order.expressions[0].this
    if (FREQUENCY.search(question) and tree.args.get("group") and tree.args.get("limit")
            and tree.args.get("joins") and not tree.args.get("having")):
        copy = tree.copy()
        ranked = copy.args["order"].expressions[0]
        ranked.set("this", exp.Count(this=exp.Star()))
        ranked.set("desc", True)
        yield copy, 94, "frequency requested but ordering uses the value of one field"
    match = ALPHA.search(question)
    if not match or len(list(tree.find_all(exp.Select))) != 1:
        return
    requested = role_tokens(match[1])
    if not requested or requested == role_tokens(column.name) and not (
            terms(column.name) & {"id", "code", "number", "key"}):
        return
    base_sources = top_tables(tree, schema) if schema is not None else {}
    options = {}
    if schema is not None:
        for alias, table in base_sources.items():
            for name, metadata in schema.columns[table].items():
                if (not re.match(r"(?:text|(?:n?var)?char|clob)", metadata["type"], re.I)
                        or terms(name) != requested):
                    continue
                options[(alias, name)] = schema.column(table, name, alias)
    else:
        # A same-query identifier is usable without spending a schema call.
        for candidate in tree.find_all(exp.Column):
            if terms(candidate.name) == requested and candidate != column:
                options[(candidate.table.lower(), candidate.name.lower())] = candidate.copy()
        # An alternative draft can provide an exact column on the same table alias.
        qualifier = column.table.lower()
        base_source = alias_tables(tree).get(qualifier)
        if base_source:
            for parent in parents:
                if not isinstance(parent, exp.Select):
                    continue
                sources = alias_tables(parent)
                for c in parent.find_all(exp.Column):
                    if terms(c.name) == requested and sources.get(c.table.lower()) == base_source:
                        options[(qualifier, c.name.lower())] = exp.column(c.name, table=column.table)
    if len(options) == 1:
        copy = tree.copy()
        ranked = copy.args["order"].expressions[0]
        ranked.set("this", next(iter(options.values())))
        ranked.set("desc", False)
        yield copy, 96, "alphabetical sort requested on a different field"


def alias_tables(tree):
    source = tree.args.get("from")
    tables = [source.this] if source is not None else []
    tables.extend(j.this for j in tree.args.get("joins") or [])
    return {t.alias_or_name.lower(): t.name.lower() for t in tables if isinstance(t, exp.Table)}


def name_kind(column):
    compact = column.name.lower().replace("_", "")
    if compact.endswith(("fname", "firstname", "forename", "givenname")):
        return "first"
    if compact.endswith(("lname", "lastname", "surname", "familyname")):
        return "last"
    return None


def parts(node):
    if isinstance(node, exp.And):
        return parts(node.this) + parts(node.expression)
    return [node]


def name_edits(tree, question, where):
    matches = {}
    for predicate in parts(where.this):
        if (isinstance(predicate, exp.EQ) and isinstance(predicate.this, exp.Column)
                and isinstance(predicate.expression, exp.Literal) and predicate.expression.is_string):
            role = name_kind(predicate.this)
            if role:
                matches.setdefault(role, []).append(predicate)
    if any(len(matches.get(role, [])) != 1 for role in ("first", "last")):
        return
    first, last = matches["first"][0], matches["last"][0]
    if first.this.table.lower() != last.this.table.lower():
        return
    a, b = first.expression.this, last.expression.this
    if not (re.fullmatch(r"[A-Za-z][A-Za-z'-]+", a) and re.fullmatch(r"[A-Za-z][A-Za-z'-]+", b)
            and a.casefold() != b.casefold()):
        return
    explicit = re.search(r"\b(?:first|last|given|family) name\b", question, re.I)
    if explicit:
        reverse = all(re.search(
            rf"\b(?:{labels}) name\s+(?:(?:as|of|is)\s+)?[\"']?{re.escape(value)}\b",
            question, re.I) for labels, value in (("first|given", b), ("last|family", a)))
    else:
        reverse = re.search(r"(?<!\w)" + re.escape(b + " " + a) + r"(?!\w)", question, re.I)
    if not reverse:
        return
    result = tree.copy()
    replacement = {"first": b, "last": a}
    for predicate in parts(result.args["where"].this):
        if predicate in (first, last):
            predicate.set("expression", exp.Literal.string(replacement[name_kind(predicate.this)]))
    yield result, 110, "given and family names are reversed relative to the named person"


def predicate_edits(tree, question):
    where = tree.args.get("where")
    if not where:
        return
    yield from name_edits(tree, question, where)
    if re.search(r"\bside\b", question, re.I) and not re.search(r"\b(?:won|win|lost|lose|winner)\b", question, re.I):
        side = terms(question) & {"affirmative", "negative"}
        join_fields = {col.name.lower() for join in tree.args.get("joins") or []
                       for col in join.find_all(exp.Column)}
        leaves = parts(where.this)
        wins = [p for p in leaves if isinstance(p, exp.EQ) and isinstance(p.this, exp.Column)
                and "win" in terms(p.this.name) and isinstance(p.expression, exp.Boolean)]
        if side & join_fields and len(wins) == 1:
            result = tree.copy()
            kept = [p.copy() for p in leaves if p != wins[0]]
            result.set("where", exp.Where(this=exp.and_(*kept)) if kept else None)
            yield result, 105, "affiliation is grounded by the join; winning is an additional filter"
    if not NEGATIVE.search(question):
        return
    for i, predicate in enumerate(tree.walk()):
        if not isinstance(predicate, exp.In) or not isinstance(predicate.parent, exp.Not):
            continue
        sub = predicate.args.get("query")
        if not isinstance(sub, exp.Subquery) or not isinstance(sub.this, exp.Select):
            continue
        inner = sub.this
        inner_where = inner.args.get("where")
        leaf = inner_where.this if inner_where else None
        tables = alias_tables(inner)
        if (not isinstance(leaf, exp.Is) or not isinstance(leaf.expression, exp.Null)
                or not isinstance(leaf.this, exp.Column) or not leaf.this.name.lower().endswith("id")
                or len(tables) != 1 or inner.args.get("joins") or len(inner.selects) != 1
                or inner.selects[0].name.lower() == leaf.this.name.lower()
                or not terms(next(iter(tables.values()))) & terms(question)):
            continue
        result = tree.copy()
        list(result.walk())[i].args["query"].this.set("where", None)
        yield result, 100, "unqualified absence request; subquery only considers rows with a null identifier"


def removable_joins(tree, question):
    joins = tree.args.get("joins") or []
    if not joins or len(list(tree.find_all(exp.Select))) != 1 or tree.find(exp.AggFunc, exp.Star):
        return
    for i, join in enumerate(joins):
        if (not isinstance(join.this, exp.Table) or join.side or join.kind not in {"", "INNER"}
                or not isinstance(join.args.get("on"), exp.EQ)
                or role_tokens(join.this.name) & terms(question)):
            continue
        alias = join.this.alias_or_name.lower()
        # Exclude only columns in this join's equality from the use analysis.
        columns = [col for col in tree.find_all(exp.Column) if col.find_ancestor(exp.Join) is not join]
        if not columns or any(not c.table or c.table.lower() == alias for c in columns):
            continue
        result = tree.copy()
        result.set("joins", [j for k, j in enumerate(result.args["joins"]) if k != i])
        yield result, 76, "unreferenced inner join can drop or duplicate the requested entity"


def join_edits(tree, question, schema):
    yield from removable_joins(tree, question)
    if schema is None:
        return
    variants = list(key_edits(tree, schema))
    if len(variants) == 1:
        _, sql, evidence = variants[0]
        yield parse(sql), 92, {"reason": "join equality conflicts with the only declared key", **evidence}


def conflicts(obs, fallback, families, schema):
    tree = simple(fallback)
    if tree is None and "predicate" not in families:
        return []
    tree = tree if tree is not None else parse(fallback)
    if not isinstance(tree, exp.Select):
        return []
    question = obs["task"]["question"]
    pool = []
    seen = {identity(e["sql"]) for e in obs["history"] if "sql" in e} | {identity(fallback)}
    for family in families:
        if family == "ordering":
            parents = [parse(sql) for sql in parent_sqls(obs, fallback)]
            generated = order_edits(tree, question, schema, parents)
        elif family == "predicate":
            generated = predicate_edits(tree, question)
        else:
            generated = join_edits(tree, question, schema)
        for query, score, reason in generated:
            sql = sql_text(query)
            key = identity(sql)
            if key in seen:
                continue
            try:
                validate_sql(sql)
            except ValueError:
                continue
            seen.add(key)
            pool.append({"sql": sql, "family": family, "priority": score, "conflict": reason})
    return pool
