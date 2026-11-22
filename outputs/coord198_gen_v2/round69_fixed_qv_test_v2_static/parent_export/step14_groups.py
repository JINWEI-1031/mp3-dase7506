"""Locate a ranked membership group, then exclude its entity keys."""
from dataclasses import dataclass
import re

from sqlglot import exp

from mp3_sql.sqlite_tool import validate_sql
from repair_rules import parse, sql_text
from step4_agent import identity
from step5x_edits import physical_table
from step7_semantics import top_tables
from step10_bindings import existing
from step11_absence import parent_sqls, root_query

NEGATED_MEMBERSHIP = re.compile(
    r"\b(?:not (?:in|a member of|part of|belong(?:ing)? to|assigned to|enrolled in)|"
    r"outside(?: of)?|except (?:those |people |members )?in|excluding (?:those |people |members )?in)\b", re.I)
RANK = re.compile(
    r"\b(?:(highest|largest|greatest|lowest|smallest)\s+(?:number|count)\s+of|"
    r"(most|fewest|least)\b(?!\s+(?:recently|expensive|valuable))|"
    r"(largest|biggest|smallest))\b", re.I)
RELATIVE = re.compile(r"\b(?:which|that|with|having|has|have|containing|whose)\b", re.I)
STOP = {"a", "an", "the", "of", "in", "any", "all", "every", "each", "one", "single",
        "have", "has", "had", "give", "tell", "show", "name", "first", "last", "id",
        "number", "count", "group", "member", "people", "person", "distinct"}
MEMBER_WORDS = {"person", "people", "member", "individual", "entity"}
MEASURES = {"area", "capacity", "age", "length", "height", "width", "size", "weight", "price",
            "salary", "balance", "revenue", "profit", "score", "population", "duration"}


@dataclass(frozen=True)
class Intent:
    prefix: str
    group_text: str
    group_words: frozenset
    population: frozenset
    descending: bool
    all_ties: bool


@dataclass(frozen=True)
class Membership:
    root: str
    member_key: str
    source: str
    source_member_key: str
    group_key: str
    path: tuple
    score: int


def terms(text):
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    words = set()
    for word in re.findall(r"[a-z]+", text.lower()):
        if word.endswith("ies") and len(word) > 4:
            word = word[:-3] + "y"
        elif word.endswith(("sses", "ches", "shes", "xes", "zes")):
            word = word[:-2]
        elif len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
            word = word[:-1]
        words.add(word)
    return words


def group_words(text):
    words = terms(text)
    return (words - STOP) or (words & {"group"})


def intent(question):
    negation = NEGATED_MEMBERSHIP.search(question)
    if negation is None:
        return None
    tail = re.split(r"[?.;]", question[negation.end():], maxsplit=1)[0]
    rank = RANK.search(tail)
    if rank is None:
        return None
    before = tail[:rank.start()]
    after = tail[rank.end():]
    measure = re.match(r"\s+([a-z]+)", after, re.I)
    if rank[3] and measure and terms(measure[1]) & MEASURES:
        return None
    group_text = RELATIVE.split(before, maxsplit=1)[0].strip()
    words = group_words(group_text)
    if not words and rank[3]:
        group_text = RELATIVE.split(after, maxsplit=1)[0].strip()
        words = group_words(group_text)
    if not words or words & MEASURES:
        return None
    population = frozenset(terms(after) - STOP) if rank[1] or rank[2] else frozenset()
    descending = not any(value and value.lower() in {"lowest", "smallest", "fewest", "least"}
                         for value in rank.groups())
    singular = re.search(r"\b(?:single|one)\b", group_text, re.I)
    plural = re.search(r"\b(?:any|all|every|each|tied)\b", tail, re.I) or (
        not singular and any(terms(word) != {word} for word in re.findall(r"[a-z]+", group_text.lower())))
    return Intent(question[:negation.start()], group_text, frozenset(words), population,
                  descending, bool(plural))


def category_score(field, request, foreign_role=()):
    words = terms(field)
    overlap = request.group_words & (words | set(foreign_role))
    if not overlap:
        return 0
    descriptive = {"id", "code", "key", "name", "number"}
    exact = (words - descriptive) == request.group_words
    return 5 * len(overlap) + 5 * exact - len((words - descriptive) - request.group_words)


def observed_groups(trees, schema):
    result = set()
    for tree in trees:
        for select in tree.find_all(exp.Select):
            group, order = select.args.get("group"), select.args.get("order")
            if not group or len(group.expressions) != 1 or not order or not order.find(exp.Count):
                continue
            column = group.expressions[0]
            if isinstance(column, exp.Column):
                binding = existing(column, top_tables(select, schema), schema)
                if binding:
                    result.add(binding[1:])
    return result


def memberships(obs, schema, fallback, request):
    headings = obs["example"]["columns"]
    trees = [tree for sql in parent_sqls(obs, fallback) if (tree := parse(sql)) is not None]
    grouped = observed_groups(trees, schema)
    source_tables = {table for tree in trees for select in tree.find_all(exp.Select)
                     for table in top_tables(select, schema).values()}
    scope_words = terms(request.prefix) - STOP
    roots = []
    for root, fields in schema.columns.items():
        output = root_query(schema, root, headings)
        primary = [key for key, meta in fields.items() if meta["pk"]]
        if output is None or len(primary) != 1 or (root not in source_tables and not terms(root) & scope_words):
            continue
        root_words = terms(root)
        if request.population and not request.population & (root_words | MEMBER_WORDS):
            continue
        score = 12 * len(root_words & scope_words) + 5 * (root in source_tables)
        roots.append((root, primary[0], score))
    for root, key, root_score in roots:
        sources = [(root, key, ())]
        # A direct reverse FK identifies a membership table's entity endpoint.
        for edge in sorted(set(schema.edges[root])):
            _, entity_key, source, member = edge
            if entity_key != key or not any(
                    fk["from"].lower() == member and fk["table"].lower() == root
                    and fk["to"].lower() == key for fk in schema.tables[source]["foreign_keys"]):
                continue
            sources.append((source, member, (edge,)))
        for source, source_key, path in sources:
            for field in schema.columns[source]:
                if field == source_key:
                    continue
                role = set().union(*(terms(edge[2]) for edge in schema.edges[source] if edge[1] == field))
                relevance = category_score(field, request, role)
                if relevance <= 0:
                    continue
                evidence = 4 * ((source, field) in grouped)
                yield Membership(root, key, source, source_key, field, path,
                                 root_score + relevance + evidence - 2 * bool(path))


def membership_query(schema, binding):
    member = schema.column(binding.source, binding.source_member_key, "a")
    group = schema.column(binding.source, binding.group_key, "a")
    query = exp.select(member.as_("member_key"), group.as_("group_key"))
    query.from_(physical_table(schema, binding.source, "a"), copy=False)
    for column in (member, group):
        query.where(exp.Not(this=exp.Is(this=column.copy(), expression=exp.Null())), copy=False)
    if binding.path:
        root_key = schema.column(binding.root, binding.member_key, "r")
        query.join(physical_table(schema, binding.root, "r"),
                   on=exp.EQ(this=root_key, expression=member.copy()), copy=False)
    return query.distinct()


def compile_query(schema, binding, headings, mode, tie_policy, descending):
    names = []
    for suffix in ("membership", "counts", "targets", "excluded"):
        name = "_step14_" + suffix
        while name in schema.tables:
            name += "_"
        names.append(name)
    membership_name, counts_name, targets_name, excluded_name = names
    assignments = membership_query(schema, binding)
    group = exp.column("group_key")
    number = exp.column("n")
    counts = exp.select(group.copy(), exp.Count(this=exp.Star()).as_("n"))
    counts.from_(membership_name, copy=False).group_by(group.copy(), copy=False)
    targets = exp.select(group.copy()).from_(counts_name)
    if tie_policy == "all":
        function = exp.Max if descending else exp.Min
        extreme = exp.select(function(this=number.copy())).from_(counts_name)
        targets.where(exp.EQ(this=number.copy(), expression=extreme.subquery()), copy=False)
    else:
        targets.order_by(exp.Ordered(this=number.copy(), desc=descending, nulls_first=not descending),
                         group.copy(), copy=False).limit(1, copy=False)
    excluded = exp.select(exp.column("member_key", table="a")).from_(membership_name + " AS a")
    excluded.join(targets_name + " AS g", on=exp.EQ(this=exp.column("group_key", table="a"),
                                                   expression=exp.column("group_key", table="g")), copy=False)
    result = root_query(schema, binding.root, headings)
    entity_key = schema.column(binding.root, binding.member_key, "e")
    result.with_(membership_name, assignments, copy=False)
    result.with_(counts_name, counts, copy=False)
    result.with_(targets_name, targets, copy=False)
    result.with_(excluded_name, excluded, copy=False)
    if mode == "exists":
        block = exp.select(exp.Literal.number(1)).from_(excluded_name + " AS x")
        block.where(exp.EQ(this=exp.column("member_key", table="x"), expression=entity_key), copy=False)
        return result.where(exp.Not(this=exp.Exists(this=block)), copy=False)
    root_key = schema.column(binding.root, binding.member_key, "d")
    universe = exp.select(root_key.as_("member_key")).from_(physical_table(schema, binding.root, "d"))
    excluded_keys = exp.select("member_key").from_(excluded_name)
    difference = universe.except_(excluded_keys, distinct=True)
    kept = exp.select(exp.Literal.number(1)).from_(difference.subquery("k"))
    kept.where(exp.NullSafeEQ(this=exp.column("member_key", table="k"), expression=entity_key), copy=False)
    return result.where(exp.Exists(this=kept), copy=False)


def pool(obs, schema, fallback, mode, request):
    seen = {identity(e["sql"]) for e in obs["history"] if "sql" in e} | {identity(fallback)}
    result = []
    for binding in memberships(obs, schema, fallback, request):
        for tie_policy in ("all", "single"):
            query = compile_query(schema, binding, obs["example"]["columns"], mode, tie_policy, request.descending)
            sql = sql_text(query)
            key = identity(sql)
            if key in seen:
                continue
            try:
                validate_sql(sql)
            except ValueError:
                continue
            seen.add(key)
            result.append({"sql": sql, "mode": mode, "root": binding.root, "source": binding.source,
                           "group_key": binding.group_key, "member_key": binding.source_member_key,
                           "path": binding.path, "tie_policy": tie_policy,
                           "priority": binding.score + 3 * ((tie_policy == "all") == request.all_ties)})
            if len(result) >= 16:
                return result
    return result
