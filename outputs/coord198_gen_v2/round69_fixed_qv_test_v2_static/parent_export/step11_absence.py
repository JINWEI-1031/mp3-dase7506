"""Build matched NOT EXISTS and EXCEPT candidates from observed relation roles."""
from dataclasses import dataclass
import re

from sqlglot import exp

from mp3_sql.sqlite_tool import validate_sql
from repair_rules import parse, sql_text
from step4_agent import identity
from step5x_edits import physical_table
from step7_semantics import join_path, terms, top_tables
from step8_groups import output_bindings
from step10_bindings import ATOMS, existing, field_words

ABSENCE = re.compile(
    r"\b(?:never|without|where no|(?:do(?:es)?\s+not|don['’]t|doesn['’]t|"
    r"has not|have not)\s+(?:have|advise|authorize|participate|pass|attend|use|visit|belong|play|hold|"
    r"work|take|receive|enroll|include|own|appear|serve|contain)\b|"
    r"(?:has|have)\s+no\b|(?:is|are|were)\s+not\s+(?:used|compatible|produced|involved|enrolled|"
    r"associated|listed|assigned|included|located|in)\b)", re.I)
NOISE = {"any", "all", "one", "without", "never", "not", "have", "has", "had", "are", "were", "use",
         "find", "show", "give", "the", "which", "who", "whose", "that", "name", "named",
         "with", "detail", "record", "locate", "located", "from", "each", "list"}
THRESHOLD = re.compile(r"\b(over|above|more than|greater than|at least|less than|under|at most)\s+(-?\d+(?:\.\d+)?)", re.I)
COMPARE = {"over": exp.GT, "above": exp.GT, "more than": exp.GT, "greater than": exp.GT,
           "at least": exp.GTE, "less than": exp.LT, "under": exp.LT, "at most": exp.LTE}


@dataclass(frozen=True)
class Filter:
    table: str
    field: str
    operator: type
    value: str
    numeric: bool


@dataclass(frozen=True)
class Plan:
    root: str
    path: tuple
    filters: tuple
    label_scope: bool
    priority: int
    parent: str
    inferred_edge: bool
    disjunction: bool = False


def parent_sqls(obs, fallback):
    return list(dict.fromkeys([fallback, obs["issue_sql"], *(
        sql for e in obs["history"] for sql in (
            [e["sql"]] if e["tool"] == "check" else e["observation"] if e["tool"] == "repair_hint" else []))]))


def add_observed_edges(schema, trees):
    """A draft equality can back a same-name key if exactly one end is unique."""
    added = set()
    for tree in trees:
        for select in tree.find_all(exp.Select):
            sources = top_tables(select, schema)
            for join in select.args.get("joins") or []:
                on = join.args.get("on")
                if not isinstance(on, exp.EQ) or not all(
                        isinstance(x, exp.Column) for x in (on.this, on.expression)):
                    continue
                left, right = (existing(c, sources, schema) for c in (on.this, on.expression))
                if not left or not right:
                    continue
                _, a, ac = left
                _, b, bc = right
                if (a == b or ac != bc or ac == "id"
                        or (a, ac, b, bc) in schema.edges[a]):
                    continue
                pk_count = sum([k for k, meta in schema.columns[t].items() if meta["pk"]] == [ac]
                               for t in (a, b))
                if pk_count != 1:
                    continue
                for source, target in ((a, b), (b, a)):
                    edge = (source, ac, target, bc)
                    schema.edges[source].append(edge)
                    added.add(edge)
    return added


def literal_specs(tail, trees):
    """Track values carried by the question and the fields that drafts used."""
    specs = {}
    for value in re.findall(r"""(?<!\w)["']([^"']+)["']""", tail):
        specs[value.casefold()] = (value, False, exp.EQ, set())
    threshold = THRESHOLD.search(tail)
    if threshold:
        operator, value = threshold.groups()
        specs[value] = (value, True, COMPARE[operator.lower()], set())
    for tree in trees:
        for node in tree.find_all(*ATOMS):
            if not isinstance(node.this, exp.Column) or not isinstance(node.expression, exp.Literal):
                continue
            value = node.expression.this.strip("%")
            key = value.casefold()
            if not value or key not in specs and (not node.expression.is_string or
                    not re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", tail, re.I)):
                continue
            specs.setdefault(key, (value, False, exp.EQ, set()))
            specs[key][3].add(node.this.name.lower())
    if not specs:
        category = re.search(r"\b([a-z]+)[ -]type\b|\bwithout (?:a|an) ([a-z]+) (\w+)\b", tail, re.I)
        if category:
            word, value, noun = category.groups()
            # Explicit category modifier plus schema discriminator.
            specs[(word or value).casefold()] = ((word or value).title(), False, exp.EQ,
                                                {"type", f"{noun}_type_code"} if noun else {"type"})
    return list(specs.values())


def bind_filters(schema, tables, specs, tail):
    tail_words = terms(tail)
    output, strength = [], 0
    for value, numeric, comparator, old_fields in specs:
        context_parts = re.split(re.escape(value), tail, maxsplit=1, flags=re.I)
        before = terms(context_parts[0][-45:])
        after = terms(" ".join(context_parts[-1].split()[:2])) if len(context_parts) == 2 else set()
        context = before | after
        options = []
        for t in tables:
            for field, info in schema.columns[t].items():
                words = set(field_words(field))
                textual = re.match(r"(?:text|char|varchar|nvarchar|clob)", info["type"], re.I)
                if (numeric and textual) or (not numeric and not textual
                        and re.match(r"(?:int|real|float|double|decimal|numeric)", info["type"], re.I)):
                    continue
                explicit = words & context
                category = ("type" in words and ("type" in tail_words or field in old_fields))
                score = (6 * len(explicit) + 3 * (field in old_fields) + 4 * category
                         + 2 * len(words & tail_words) + len(terms(t) & context))
                if score and (explicit or field in old_fields or category):
                    options.append((score, t, field))
        options.sort(key=lambda x: (-x[0], x[1:]))
        if not options or len(options) > 1 and options[0][0] == options[1][0]:
            return None, 0
        score, t, field = options[0]
        output.append(Filter(t, field, comparator, value, numeric))
        strength += score
    return tuple(output), strength


def root_query(schema, root, headings, alias="e"):
    query = exp.select().from_(physical_table(schema, root, alias))
    output = output_bindings(query, schema, headings)
    if output:
        return query.select(*output, append=False)
    return None


def plans(obs, schema, fallback, match):
    question = obs["task"]["question"]
    tail = question[match.start():]
    focus = terms(tail) - NOISE
    if not focus:
        return
    headings = obs["example"]["columns"]
    parents = [(sql, parse(sql)) for sql in parent_sqls(obs, fallback)]
    parents = [(sql, tree) for sql, tree in parents if isinstance(tree, exp.Select)]
    trees = [tree for _, tree in parents]
    inferred = add_observed_edges(schema, trees)
    specs = literal_specs(tail, trees)
    prefix = terms(question[:match.start()]) - NOISE
    grouped = match[0].lower() == "where no"
    for root in schema.tables:
        base = root_query(schema, root, headings)
        if base is None:
            continue
        found = [(sql, tree) for sql, tree in parents if root in top_tables(tree, schema).values()]
        root_overlap = len(terms(root) & prefix)
        if not (found or root_overlap):
            continue
        parent = found[0][0] if found else fallback
        root_words = terms(root)
        root_score = 9 * root_overlap + 4 * bool(found) + 8 * (root_words <= prefix)
        if len(root_words) > 1 and root_overlap < len(root_words):
            root_score -= 3
        if grouped and specs and all(isinstance(p, exp.Column) for p in base.selects):
            filters, score = bind_filters(schema, [root], specs, tail)
            if filters:
                yield Plan(root, (), filters, True, root_score + score + 10, parent, False)
        for target in schema.tables:
            if target == root:
                continue
            for path in sorted(set(map(tuple, schema.path_options([root], target, 2)))):
                related = [edge[2] for edge in path]
                table_terms = set().union(*(terms(t) for t in related)) - terms(root)
                relation = focus & (table_terms | set().union(*(terms(edge[1]) | terms(edge[3]) for edge in path)))
                if not relation:
                    continue
                filters, score = bind_filters(schema, related, specs, tail)
                if filters is None or filters and not any(f.table == target for f in filters):
                    continue
                # With no value predicate, a matching bridge itself witnesses an association.
                if not filters and not (focus & (terms(target) - terms(root))):
                    continue
                observed = any(edge in inferred for edge in path)
                priority = root_score + 6 * len(relation) + score - 3 * len(path) - 4 * observed
                disjunction = len(filters) > 1 and bool(re.search(r"\bor\b", tail, re.I))
                yield Plan(root, path, filters, grouped, priority, parent, observed, disjunction)


def qualification(body, plan, schema, aliases):
    predicates = []
    for predicate in plan.filters:
        column = schema.column(predicate.table, predicate.field, aliases[predicate.table])
        value = (exp.Literal.number(predicate.value) if predicate.numeric
                 else exp.Literal.string(predicate.value))
        predicates.append(predicate.operator(this=column, expression=value))
    if predicates:
        expression = exp.or_(*predicates) if plan.disjunction else exp.and_(*predicates)
        body.where(expression, copy=False)


def compile_plan(plan, schema, headings, mode):
    base = root_query(schema, plan.root, headings)
    match = exp.select().from_(physical_table(schema, plan.root, "m"))
    aliases = {plan.root: "m"}
    join_path(match, plan.path, schema, aliases)
    qualification(match, plan, schema, aliases)
    chosen = base.selects
    projection = all(isinstance(p, exp.Column) for p in chosen)
    fields = [p.name.lower() for p in chosen] if plan.label_scope else [
        plan.path[0][1]] if plan.path else []
    if mode == "exists":
        for field in fields:
            left = schema.column(plan.root, field, "m")
            right = schema.column(plan.root, field, "e")
            operator = exp.NullSafeEQ if plan.label_scope else exp.EQ
            match.where(operator(this=left, expression=right), copy=False)
        match.select(exp.Literal.number(1), copy=False)
        base.where(exp.Not(this=exp.Exists(this=match)), copy=False)
        if plan.label_scope:
            base.distinct(copy=False)
        return base
    if projection:
        match.select(*(schema.column(plan.root, p.name.lower(), "m") for p in chosen), copy=False)
        return base.except_(match, distinct=True)
    # Subtract matched relation keys before applying outer aggregates, preserving
    # repeated values on different entities and NULL keys on unrelated entities.
    if len(fields) != 1:
        return None
    key = fields[0]
    outer_key = schema.column(plan.root, key, "e")
    matched_key = schema.column(plan.root, key, "m")
    match.select(matched_key, copy=False)
    match.where(exp.Not(this=exp.Is(this=matched_key.copy(), expression=exp.Null())), copy=False)
    domain = exp.select(schema.column(plan.root, key, "k")).from_(physical_table(schema, plan.root, "k"))
    kept = exp.In(this=outer_key.copy(), query=domain.except_(match, distinct=True).subquery())
    return base.where(exp.Or(this=exp.Is(this=outer_key, expression=exp.Null()), expression=kept))


def pool(obs, schema, fallback, mode):
    match = ABSENCE.search(obs["task"]["question"])
    if not match:
        return []
    seen = {identity(e["sql"]) for e in obs["history"] if "sql" in e} | {identity(fallback)}
    items = []
    for plan in plans(obs, schema, fallback, match):
        query = compile_plan(plan, schema, obs["example"]["columns"], mode)
        if query is None:
            continue
        sql = sql_text(query)
        key = identity(sql)
        if key in seen:
            continue
        try:
            validate_sql(sql)
        except ValueError:
            continue
        seen.add(key)
        items.append({"sql": sql, "mode": mode, "root": plan.root, "path": plan.path,
                      "filters": [[f.table, f.field, f.operator.key, f.value] for f in plan.filters],
                      "label_scope": plan.label_scope, "priority": plan.priority,
                      "parent": plan.parent, "observed_key_edge": plan.inferred_edge,
                      "filter_disjunction": plan.disjunction})
        if len(items) >= 24:
            break
    return items
