"""Compile cardinality and set constraints using the observed schema."""
import re

from sqlglot import exp

from repair_rules import parse, sql_text
from step4_edits import COMPARISONS
from step5x_edits import physical_table

NUM = r"\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten"
NUMBERS = {word: i for i, word in enumerate(NUM.split("|")[1:])}
RANK = re.compile(r"\b(?:most|least) (?:common|popular|frequent)\b|"
                  r"\b(?:most|least) (?:often|frequently)\b|\bfewest\b|"
                  r"\b(?:top|most|least|largest|greatest|highest|lowest) (?:\w+ ){0,2}(?:number|count)\b|"
                  r"\b(?:has|have|with|the) (?:the )?(?:most|fewest)\b")
THRESHOLD = re.compile(rf"\b(at least|at most|more than|less than|greater than|fewer than|"
                       rf"bigger than|over|under) ({NUM})\b")
OPERATORS = {"at least": ">=", "at most": "<=", "more than": ">", "less than": "<",
             "greater than": ">", "fewer than": "<", "bigger than": ">", "over": ">", "under": "<"}
COUNT = re.compile(r"\b(?:number of|count of|how many)\b")
PER = re.compile(r"\b(?:per|each|every)\b")
ANTI = re.compile(r"\b(?:without|never|no|not used|not compatible|(?:do(?:es)? not|don't) \w+)"
                  r"\b")
DUAL = re.compile(r"\bboth\b.*\band\b")
COMPARISON = (exp.EQ, exp.NEQ, exp.Like, *COMPARISONS.values())


def number(text):
    return NUMBERS.get(text, int(text) if text.isdigit() else None)


def terms(text):
    words = re.findall(r"[a-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", text).lower())
    return {w[:-3] + "y" if w.endswith("ies") else w.rstrip("s") if len(w) > 3 else w
            for w in words}


def top_tables(tree, schema):
    entries = [tree.args["from"].this] if tree.args.get("from") else []
    entries.extend(j.this for j in tree.args.get("joins") or [])
    return {t.alias_or_name.lower(): t.name.lower() for t in entries
            if isinstance(t, exp.Table) and t.name.lower() in schema.tables}


def bindings(column, aliases, schema):
    preferred = aliases.get(column.table.lower())
    name = column.name.lower()
    if name in schema.columns.get(preferred, {}):
        return [(preferred, name)]
    local = [(t, name) for t in dict.fromkeys(aliases.values()) if name in schema.columns[t]]
    return local or [(t, name) for t in schema.columns if name in schema.columns[t]]


def shortest(schema, origin, target):
    paths = {tuple(p) for p in schema.path_options([origin], target, 3)}
    length = min(map(len, paths), default=100)
    return sorted(p for p in paths if len(p) == length)


def join_path(query, path, schema, aliases):
    for left, a, right, b in path:
        if right in aliases:
            continue
        aliases[right] = f"r{len(aliases)}"
        on = exp.EQ(this=schema.column(left, a, aliases[left]),
                    expression=schema.column(right, b, aliases[right]))
        query.join(physical_table(schema, right, aliases[right]), on=on, copy=False)


def projections(tree):
    return [p.unalias() for p in tree.selects if isinstance(p.unalias(), exp.Column)]


def aggregate_intent(question, arity):
    rank = RANK.search(question)
    threshold = THRESHOLD.search(question)
    count = COUNT.search(question)
    ordered = re.search(r"\b(?:ascending|descending|most to least|least to most)\b", question)
    top = re.search(rf"\btop ({NUM})\b", question)
    first = re.search(rf"\b({NUM}) \w+ (?:followed|with|having)\b", question)
    if not (rank or top or threshold or (count and (PER.search(question) or arity > 1))):
        return None
    return rank, threshold, count, ordered, top or first


def aggregate_rebuild(tree, schema, question, arity, role_paths=False):
    intent = aggregate_intent(question, arity)
    if not intent or arity not in (1, 2, 3):
        return
    rank, threshold, count, ordered, limit = intent
    parents = top_tables(tree, schema)
    labels = projections(tree)
    if not parents or not labels:
        return
    count_output = bool((arity > 1 and count and not rank and not threshold)
                        or (arity > 1 and re.search(r"\b(?:also|and) (?:tell|show|give|return).{0,14}\bcount\b",
                                                   question)))
    label_count = arity - count_output
    focus = terms(question[:(threshold or rank).start()] if threshold or rank else question)
    qwords = terms(question)
    candidates = []
    for col in labels:
        for table, name in bindings(col, parents, schema):
            score = 3 * len(terms(name) & focus) + len(terms(name) & qwords)
            score += 3 * len(terms(table) & focus) + (2 if parents.get(col.table.lower()) == table else 0)
            candidates.append((score, table, name))
    if role_paths:
        # Supply missing output bindings from schema when the requested field is explicit.
        for table, columns in schema.columns.items():
            for name in columns:
                shared = terms(name) & focus
                entities = terms(table) & focus
                if shared and (entities or table in parents.values()):
                    score = 3 * len(shared) + len(shared) + 3 * len(entities) - 1
                    candidates.append((score, table, name))
    candidates.sort(key=lambda x: -x[0])
    object_text = question[(threshold or rank or count).end():] if threshold or rank or count else question
    object_words = terms(re.split(r"\b(?:ordered|sorted|for each|per|whose)\b", object_text)[0])
    frequency = rank and re.search(r"common|popular|frequent", rank[0])
    literal_predicates = []
    where = tree.args.get("where")
    if where and not where.find(exp.Or):
        literal_predicates = [p for p in where.find_all(*COMPARISON)
                              if isinstance(p.this, exp.Column) and isinstance(p.expression, exp.Literal)
                              and not p.find_ancestor(exp.Subquery)]
    plans = []
    for label_score, entity, first_label in candidates[:8]:
        names = [first_label]
        names.extend(n for _, t, n in candidates if t == entity and n not in names)
        if len(names) < label_count:
            continue
        names = names[:label_count]
        # Restore explicitly named output fields to question order.
        names.sort(key=lambda n: min((question.find(w) for w in terms(n) if w in qwords), default=1000))
        for fact in schema.tables:
            if not role_paths and fact not in parents.values() and fact != entity:
                continue
            paths = shortest(schema, entity, fact)
            for path in paths[:3]:
                relation = len(terms(fact) & object_words)
                score = label_score + 7 * relation + (2 if fact in parents.values() else 0)
                if frequency:
                    score += len(terms(fact) & qwords) * 3
                if fact != entity:
                    score += 2 if path[-1][0] == entity else 0
                if path and role_paths:
                    edge_words = terms(path[-1][-1])
                    if re.search(r"\b(?:start|depart|origin|source)\w*\b", question):
                        score += 9 * bool(edge_words & {"start", "src", "source", "origin", "departure"})
                    if re.search(r"\b(?:arriv|end|destination)\w*\b", question):
                        score += 9 * bool(edge_words & {"end", "dst", "destination", "arrival"})
                score -= len(path) * 2
                plans.append((score, entity, fact, path, names))
    seen = set()
    for _, entity, fact, path, names in sorted(plans, key=lambda p: -p[0]):
        aliases = {entity: "r0"}
        result = exp.select().from_(physical_table(schema, entity, "r0"))
        join_path(result, path, schema, aliases)
        cols = [schema.column(entity, n, "r0") for n in names]
        measure = exp.Count(this=exp.Star())
        measures = [(len(terms(n) & object_words), n) for n in schema.columns[fact]
                    if not n.endswith("id") and terms(n) & object_words]
        if "total number" in question and measures and fact != entity:
            _, name = max(measures)
            measure = exp.Sum(this=schema.column(fact, name, aliases[fact]))
        if "distinct" in object_words:
            keys = sorted((len(terms(n) & object_words), n) for n in schema.columns[fact])
            if keys and keys[-1][0]:
                measure = exp.Count(this=exp.Distinct(expressions=[
                    schema.column(fact, keys[-1][1], aliases[fact])]))
        for pred in literal_predicates:
            binds = [(t, c) for t, c in bindings(pred.this, parents, schema) if t in aliases]
            if len(binds) != 1 or (threshold and str(number(threshold[2])) == pred.expression.this):
                continue
            t, c = binds[0]
            result.where(type(pred)(this=schema.column(t, c, aliases[t]),
                                    expression=pred.expression.copy()), copy=False)
        output = cols
        if count_output:
            output = [measure, *cols] if count and count.start() < 12 else [*cols, measure]
        groups = list(cols)
        if fact != entity:
            keys = [n for n, column in schema.columns[entity].items() if column.get("pk")]
            groups.extend(schema.column(entity, n, "r0") for n in keys if n not in names)
        result.select(*output, append=False, copy=False).group_by(*groups, copy=False)
        if threshold:
            comparator = COMPARISONS[OPERATORS[threshold[1]]]
            result.having(comparator(this=measure.copy(), expression=exp.Literal.number(number(threshold[2]))),
                          copy=False)
        if rank or ordered or limit:
            desc = not re.search(r"\b(?:least|fewest|lowest|ascending)\b", question if not ordered else ordered[0])
            order = cols[0] if "alphabetical" in question and not rank else measure.copy()
            result.order_by(exp.Ordered(this=order, desc=desc, nulls_first=not desc), copy=False)
        if limit or (rank and not ordered):
            result.limit(number(limit[1]) if limit else 1, copy=False)
        sql = sql_text(result)
        if sql in seen:
            continue
        yield ("aggregate_roles" if role_paths else "aggregate_clauses", sql,
               {"entity": entity, "output": names, "fact": fact, "foreign_keys": path,
                "measure": measure.sql(), "threshold": threshold[0] if threshold else None})
        seen.add(sql)
        if len(seen) == 4:
            return


def negative_in(tree, schema):
    for node in tree.find_all(exp.Not):
        clause = node.this
        if not isinstance(clause, exp.In) or not isinstance(clause.this, exp.Column):
            continue
        sub = clause.args.get("query")
        if not sub or not isinstance(sub.this, exp.Select) or len(sub.this.selects) != 1:
            continue
        inner = sub.this
        outer_aliases = top_tables(tree, schema)
        inner_aliases = top_tables(inner, schema)
        a, b = clause.this, inner.selects[0].unalias()
        if len(outer_aliases) != 1 or len(inner_aliases) != 1 or not isinstance(b, exp.Column):
            continue
        a_table = next(iter(outer_aliases.values()))
        b_table = next(iter(inner_aliases.values()))
        if a_table == b_table or inner.args.get("limit") or inner.args.get("group"):
            continue
        edges = [e for e in schema.edges[a_table] if e[2] == b_table]
        edges.sort(key=lambda e: -(e[1] == a.name.lower()) - (e[3] == b.name.lower()))
        for _, left, _, right in edges[:2]:
            result = tree.copy()
            target = next(n for n in result.find_all(exp.Not) if n == node)
            body = target.this.args["query"].this
            outer_alias = next(iter(outer_aliases))
            inner_alias = next(iter(inner_aliases))
            if outer_alias == inner_alias:
                continue
            body.select(exp.Literal.number(1), append=False, copy=False)
            body.where(exp.EQ(this=schema.column(a_table, left, outer_alias),
                              expression=schema.column(b_table, right, inner_alias)), copy=False)
            target.replace(exp.Not(this=exp.Exists(this=body.copy())))
            yield "negation_correlated", sql_text(result), {"foreign_key": [a_table, left, b_table, right]}


def group_conditions(tree, question):
    for node in tree.find_all(exp.In):
        values = node.expressions
        if (isinstance(node.this, exp.Column) and len(values) == 2
                and all(isinstance(v, exp.Literal) for v in values)
                and all(str(v.this).lower() in question for v in values)):
            return node.this.name.lower(), values
    tests = [p for p in tree.find_all(exp.EQ)
             if isinstance(p.this, exp.Column) and isinstance(p.expression, exp.Literal)
             and str(p.expression.this).lower() in question]
    for left in tests:
        for right in tests:
            if left.this.name.lower() == right.this.name.lower() and left.expression != right.expression:
                return left.this.name.lower(), [left.expression, right.expression]
    return None


def both_groups(tree, schema, question, arity):
    condition = group_conditions(tree, question)
    labels = projections(tree)
    if not condition or len(labels) != arity:
        return
    name, values = condition
    parents = top_tables(tree, schema)
    choices = []
    for entity, column in bindings(labels[0], parents, schema):
        names = [column]
        for label in labels[1:]:
            matches = [n for t, n in bindings(label, parents, schema) if t == entity]
            if matches:
                names.append(matches[0])
        if len(names) != arity:
            continue
        for table, cols in schema.columns.items():
            if name not in cols:
                continue
            for path in shortest(schema, entity, table)[:2]:
                choices.append((len(path), entity, table, path, names))
    for _, entity, table, path, names in sorted(choices)[:2]:
        aliases = {entity: "r0"}
        base = exp.select(*(schema.column(entity, n, "r0") for n in names)).from_(
            physical_table(schema, entity, "r0"))
        join_path(base, path, schema, aliases)
        field = schema.column(table, name, aliases[table])
        keys = [n for n, meta in schema.columns[entity].items() if meta.get("pk")]
        evidence = {"group_key": keys if path and keys else names,
                    "output": names, "predicate": name, "foreign_keys": path}
        if not path:
            one, two = [base.where(exp.EQ(this=field.copy(), expression=v.copy())) for v in values]
            yield "both_intersect", sql_text(exp.Intersect(this=one, expression=two, distinct=True)), evidence
        group = keys if path and keys else names
        sql = base.where(exp.In(this=field.copy(), expressions=[v.copy() for v in values]))
        sql.group_by(*(schema.column(entity, n, "r0") for n in group), copy=False)
        sql.having(exp.EQ(this=exp.Count(this=exp.Distinct(expressions=[field])),
                          expression=exp.Literal.number(2)), copy=False)
        yield "both_group_count", sql_text(sql), evidence
