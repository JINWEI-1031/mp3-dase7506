"""Compile bounded edits from the question and observed schema."""
import re
from collections import Counter

from sqlglot import exp

from mp3_sql.sqlite_tool import validate_sql
from repair_rules import parse, sql_text
from step4_agent import identity, passed
from step4_edits import COMPARISONS, threshold_edits
from step5x_edits import physical_table
from step7_semantics import ANTI, join_path, terms, top_tables
from step8_groups import alias_free, output_bindings
from step9_roles import quoted_role_filter

ATOMS = (exp.EQ, exp.NEQ, exp.Like, exp.Between, exp.In, *COMPARISONS.values())
STOP = {"a", "an", "and", "as", "by", "for", "in", "is", "of", "or", "the", "to", "with"}
ROLE = {"source", "origin", "start", "departure", "destination", "end", "arrival",
        "manager", "advisor", "parent", "child", "home", "away"}


def tokens(value):
    value = re.sub(r"([a-z])([A-Z])", r"\1 \2", value)
    result = []
    for word in re.findall(r"[A-Za-z]+|\d+", value):
        word = next(iter(terms(word)), word.lower())
        result.append(word[:-3] if word.endswith("ing") and len(word) > 6 else word)
    return result


def field_words(name):
    return [w for w in tokens(name) if w not in STOP]


def atoms(where):
    if where is None:
        return []
    return [node for node in where.find_all(*ATOMS) if isinstance(node.this, exp.Column)
            and not node.find(exp.Select) and list(node.find_all(exp.Literal))]


def local_occurrences(question, value):
    needle = tokens(value.strip("%"))
    return [i for i in range(len(question)) if needle and question[i:i + len(needle)] == needle]


def column_evidence(question, column, positions):
    words = field_words(column)
    if not words or not positions:
        return 0
    best = 0
    for start, end in positions:
        before = [w for w in question[max(0, start - 7):start] if w not in STOP]
        after = [w for w in question[end:end + len(words) + 1] if w not in STOP]
        for gap in range(3):
            if before[-len(words) - gap:len(before) - gap] == words:
                best = max(best, 12 * len(words) - 4 * gap)
        if after[:len(words)] == words:
            best = max(best, 12 * len(words) + 1)
    return best


def existing(column, sources, schema):
    name = column.name.lower()
    if column.table:
        table = sources.get(column.table.lower())
        return (column.table.lower(), table, name) if name in schema.columns.get(table, {}) else None
    options = [(alias, t, name) for alias, t in sources.items() if name in schema.columns[t]]
    return options[0] if len(options) == 1 else None


def compatible(node, metadata):
    affinity = metadata["type"].lower()
    if isinstance(node, (exp.Between, *COMPARISONS.values())) and all(
            re.fullmatch(r"-?\d+(?:\.\d+)?", v.this) for v in node.find_all(exp.Literal)):
        return not re.match(r"(?:text|varchar|char|clob)", affinity)
    if isinstance(node, (exp.Like, exp.EQ, exp.NEQ, exp.In)):
        values = list(node.find_all(exp.Literal))
        if any(v.is_string and not re.fullmatch(r"-?\d+(?:\.\d+)?", v.this) for v in values):
            return not re.match(r"(?:int|real|float|double|numeric|decimal)", affinity)
    return True


def choice(node, sources, schema, question, remote=False):
    current = existing(node.this, sources, schema)
    values = list(node.find_all(exp.Literal))
    literal = values[0].this
    positions = [(i, i + len(tokens(literal))) for i in local_occurrences(question, literal)]
    if not positions:
        return None
    name = node.this.name.lower()
    options = []
    for alias, table in ([(None, t) for t in schema.tables] if remote else sources.items()):
        for field, meta in schema.columns[table].items():
            if not compatible(node, meta):
                continue
            raw = column_evidence(question, field, positions)
            if field in {"name", "id", "code", "type"} and field != name:
                raw = 0
            score = raw + 3 * (field == name) + 2 * bool(set(field_words(table)) & set(question))
            if raw or remote and field == name and not current:
                options.append((score, alias, table, field))
    options.sort(reverse=True)
    if not options or len(options) > 1 and options[0][0] == options[1][0]:
        return None
    score, alias, table, field = options[0]
    if current:
        old = column_evidence(question, name, positions) + 3
        if (table, field) == current[1:] or score <= old + 2:
            return None
    return alias, table, field, score


def boolean_edits(tree, question):
    where = tree.args.get("where")
    if where is None:
        return
    parts = re.split(r"\bor\b", question, flags=re.I)
    root = where.this
    if len(parts) == 2 and isinstance(root, exp.And):
        left, right = root.this, root.expression
        pair = [(left, right), (right, left)]
        if any(all(isinstance(n, ATOMS) and not isinstance(n, exp.In) and any(local_occurrences(tokens(p), v.this)
                   for v in n.find_all(exp.Literal)) for p, n in zip(parts, nodes)) for nodes in pair):
            yield "predicate_boolean", tree.where(exp.Or(this=left.copy(), expression=right.copy()),
                                                  append=False), 65
        for first, last in pair:
            if (not re.search(r"\beither\b", parts[0], re.I) or not isinstance(first, exp.In)
                    or len(first.expressions) != 2 or not isinstance(last, ATOMS)):
                continue
            a, b = first.expressions
            if (isinstance(a, exp.Literal) and isinstance(b, exp.Literal)
                    and local_occurrences(tokens(parts[0]), a.this)
                    and local_occurrences(tokens(parts[1]), b.this)):
                left = exp.EQ(this=first.this.copy(), expression=a.copy())
                right = exp.EQ(this=first.this.copy(), expression=b.copy())
                yield "predicate_boolean_scope", tree.where(exp.Or(
                    this=left, expression=exp.Paren(this=exp.And(this=right, expression=last.copy()))),
                    append=False), 76
                if simple_layout(tree) and not tree.args.get("order") and not tree.args.get("limit"):
                    first_query = tree.where(left.copy(), append=False)
                    second_query = tree.where(exp.And(this=right.copy(), expression=last.copy()), append=False)
                    yield "predicate_boolean_union", first_query.union(second_query, distinct=True), 77
    if isinstance(root, exp.EQ) and isinstance(root.expression, exp.Literal):
        value = re.escape(root.expression.this)
        if re.search(rf"\b(?:is|are) not (?:\w+ )?['\"]?{value}\b", question, re.I):
            yield "predicate_negation", tree.where(exp.NEQ(this=root.this.copy(),
                                        expression=root.expression.copy()), append=False), 75


def predicate_edits(tree, schema, question, feedback):
    words = tokens(question)
    edited = tree.copy()
    strength = []
    for select in edited.find_all(exp.Select):
        sources = top_tables(select, schema)
        for atom in atoms(select.args.get("where")):
            bound = choice(atom, sources, schema, words)
            if bound:
                alias, table, field, score = bound
                atom.this.replace(schema.column(table, field, alias))
                strength.append(score)
    if strength:
        yield "predicate_column", edited, 80 + min(sum(strength), 60)
        for rule, query, priority in boolean_edits(edited, question):
            yield rule + "_with_binding", query, priority + 80
    for _, sql, _ in threshold_edits(tree, question.lower()):
        yield "predicate_operator", parse(sql), 71
    yield from boolean_edits(tree, question)
    # Explicit value spelling wins. A zero-row comparison can also probe casing.
    explicit = re.findall(r"""(?:"([^"]+)"|'([^']+)')""", question)
    quoted = {a or b for a, b in explicit}
    for idx, atom in enumerate(atoms(tree)):
        value = atom.expression
        if not isinstance(atom, (exp.EQ, exp.NEQ)) or not isinstance(value, exp.Literal):
            continue
        matches = re.findall(rf"(?<!\w){re.escape(value.this)}(?!\w)", question, re.I)
        spellings = set(matches)
        variants = []
        if len(spellings) == 1:
            wanted = matches[0]
            if wanted != value.this and value.is_string:
                variants.append((wanted, 64))
            if feedback.get("row_count") == 0 and value.is_string and value.this.isalpha():
                variants.extend((word, 53) for word in (wanted.title(), wanted.lower()) if word != value.this)
        for word, rank in dict.fromkeys(variants):
            result = tree.copy()
            atoms(result)[idx].set("expression", exp.Literal.string(word))
            yield "predicate_literal", result, rank
        select = atom.find_ancestor(exp.Select)
        if select is None or atom.find_ancestor(exp.Where) is None:
            continue
        binding = existing(atom.this, top_tables(select, schema), schema)
        if not binding:
            continue
        _, table, field = binding
        affinity = schema.columns[table][field]["type"].lower()
        if (not value.is_string and value.this in quoted and re.match(r"(?:char|varchar|text)", affinity)
                or value.is_string and value.this not in quoted and re.match(
                    r"(?:int|real|decimal|numeric)", affinity) and value.this.isdecimal()):
            result = tree.copy()
            new = exp.Literal.number(value.this) if value.is_string else exp.Literal.string(value.this)
            atoms(result)[idx].set("expression", new)
            yield "predicate_literal_type", result, 74
        if (value.is_string and value.this.isdecimal() and re.match(r"(?:char|varchar|text)", affinity)
                and len(field_words(field)) == 1 and re.search(
                    rf"\b{re.escape(field)}\s+(?:is\s+)?(?:not\s+)?{value.this}\b", question, re.I)):
            result = tree.copy()
            atoms(result)[idx].set("expression", exp.Literal.string(f"{field.title()} {value.this}"))
            yield "predicate_category_code", result, 78


def path_rank(path, question):
    roles = set(question) & ROLE
    hit = sum(len(roles & set(field_words(a) + field_words(b))) for _, a, _, b in path)
    return len(path), -hit


def connect(query, schema, aliases, target, question):
    if target in aliases:
        return []
    paths = sorted({tuple(p) for p in schema.path_options(aliases, target, 3)},
                   key=lambda p: (path_rank(p, question), p))
    if not paths or len(paths) > 1 and path_rank(paths[0], question) == path_rank(paths[1], question):
        return None
    join_path(query, paths[0], schema, aliases)
    return paths[0]


def output_tables(headings, schema):
    for table in schema.tables:
        query = exp.select().from_(physical_table(schema, table, "r0"))
        if output_bindings(query, schema, headings):
            yield table


def predicate_plan(tree, sources, schema, question, prune):
    where = tree.args.get("where")
    if not where:
        return [], None
    plan = []
    for atom in atoms(where):
        current = existing(atom.this, sources, schema)
        target = choice(atom, sources, schema, question, remote=True)
        if target is None and current is None:
            return None, None
        plan.append((atom, target or (*current, 0)))
    if not prune:
        return plan, where.this
    # Removing an invented filter is allowed only in a conjunction, and only
    # when its value, role and field lack independent support in the question.
    if where.find(exp.Or):
        return plan, where.this
    def trim(node):
        if isinstance(node, exp.And):
            a, b = trim(node.this), trim(node.expression)
            return exp.And(this=a, expression=b) if a is not None and b is not None else a if a is not None else b
        if node in [a for a, _ in plan]:
            mentions = any(local_occurrences(question, v.this) for v in node.find_all(exp.Literal))
            if not mentions and not (set(field_words(node.this.name)) & set(question)):
                return None
        elif isinstance(node, exp.Is) or (isinstance(node, exp.Not) and isinstance(node.this, exp.Is)):
            column = node.find(exp.Column)
            if column is not None and not set(field_words(column.name)) & set(question):
                return None
        return node.copy()
    clause = trim(where.this)
    return [(atom, b) for atom, b in plan if clause is not None and atom in list(clause.walk())], clause


def simple_layout(tree):
    return (len(list(tree.find_all(exp.Select))) == 1 and not any(
        tree.args.get(k) for k in ("with", "group", "having", "offset", "qualify")))


def bind_clause(clause, plan, sources, query_sources, schema, root):
    if clause is None:
        return None
    clause = clause.copy()
    for atom in atoms(clause):
        binding = next((b for old, b in plan if old == atom), None)
        if binding:
            _, table, field, _ = binding
            atom.this.replace(schema.column(table, field, query_sources[table]))
    aliases = {alias.lower() for alias in query_sources.values()}
    for col in list(clause.find_all(exp.Column)):
        if col.table.lower() in aliases:
            continue
        bound = existing(col, sources, schema)
        field = col.name.lower()
        table = bound[1] if bound and bound[1] in query_sources else root
        if field not in schema.columns[table]:
            return None
        col.replace(schema.column(table, field, query_sources[table]))
    return clause


def role_edits(tree, schema, headings, question, refined):
    words = tokens(question)
    sources = top_tables(tree, schema)
    if not sources or len(sources) != len(set(sources.values())) or not simple_layout(tree):
        return
    if ANTI.search(question.lower()) and not tree.find(exp.NEQ, exp.Not):
        return
    base = alias_free(tree)
    plan, clause = predicate_plan(base, sources, schema, words, False)
    if plan is None:
        return
    missing = {b[1] for _, b in plan} - set(sources.values())
    if missing and not base.find(exp.Or) and not any(j.side for j in base.args.get("joins") or []):
        query = base.copy()
        aliases = {t: a for a, t in sources.items()}
        paths = [connect(query, schema, aliases, t, words) for t in sorted(missing)]
        if all(p is not None for p in paths):
            result = bind_clause(clause, plan, sources, aliases, schema, next(iter(sources.values())))
            if result is not None and output_bindings(query, schema, headings):
                query.where(result, append=False, copy=False)
                yield "role_revision", query, 90
    if not refined:
        return
    plan, clause = predicate_plan(base, sources, schema, words, True)
    if plan is None or clause is not None and clause.find(exp.Or) and missing:
        return
    roots = list(output_tables(headings, schema))
    for root in roots:
        changed_source = root not in sources.values()
        root_support = set(field_words(root)) & set(words)
        if changed_source and not root_support and not all(
                h.lower() in schema.columns[root] for h in headings):
            continue
        query = exp.select().from_(physical_table(schema, root, "r0"))
        aliases = {root: "r0"}
        required = {b[1] for _, b in plan}
        paths = [connect(query, schema, aliases, t, words) for t in sorted(required - {root})]
        if any(p is None for p in paths):
            continue
        where = bind_clause(clause, plan, sources, aliases, schema, root)
        order = bind_clause(base.args.get("order"), [], sources, aliases, schema, root)
        if (clause is not None and where is None or base.args.get("order") is not None and order is None):
            continue
        output = output_bindings(query, schema, headings)
        query.select(*output, append=False, copy=False)
        if where is not None:
            query.where(where, append=False, copy=False)
        for name, expr in (("order", order), ("limit", base.args.get("limit")),
                           ("distinct", base.args.get("distinct"))):
            if expr is not None:
                query.set(name, expr.copy())
        dropped = set(sources.values()) - set(aliases)
        if not (changed_source or dropped or missing):
            continue
        # Inner joins on optional or one-to-many paths can lose or multiply rows.
        # Keep their provenance, and let a public check gate adoption.
        relevance = len(root_support) * 3
        yield "role_skeleton", query, 84 + relevance + 5 * bool(missing) - sum(map(len, paths))


def projection_edit(tree, schema, headings):
    """Control route for a projection mismatch with an otherwise intact query."""
    if not isinstance(tree, exp.Select) or not simple_layout(tree) or not top_tables(tree, schema):
        return
    output = output_bindings(tree, schema, headings)
    if output is None:
        return
    names = [p.alias_or_name.lower() for p in tree.selects]
    if names == [h.strip().lower() for h in headings]:
        return
    yield "projection_contract", alias_free(tree).select(*output, append=False), 68


def candidates(obs, schema, families, refined, exclude=()):
    headings = obs["example"]["columns"]
    question = obs["task"]["question"]
    checks = {e["sql"]: e["observation"] for e in obs["history"] if e["tool"] == "check"}
    hints = [sql for e in obs["history"] if e["tool"] == "repair_hint" for sql in e["observation"]]
    seen = {identity(sql) for sql in (*checks, *exclude)}
    parents = dict.fromkeys([*(s for s, f in checks.items() if f["ok"]), *hints, *checks])
    raw, counts = [], Counter()
    for parent in list(parents)[:10]:
        tree = parse(parent)
        if not isinstance(tree, exp.Select):
            continue
        for family in families:
            if family == "predicate":
                if not checks.get(parent, {}).get("ok") and parent not in hints:
                    continue
                generated = predicate_edits(tree, schema, question, checks.get(parent, {}))
            elif family == "coupled":
                if not checks.get(parent, {}).get("ok") and parent not in hints:
                    continue
                generated = coupled_edits(tree, schema, headings, question, checks.get(parent, {}))
            elif family == "role":
                generated = role_edits(tree, schema, headings, question, refined)
            else:
                if not checks.get(parent, {}).get("ok"):
                    continue
                generated = projection_edit(tree, schema, headings)
            for rule, query, rank in generated:
                counts[family + "_generated"] += 1
                sql = sql_text(query)
                raw.append({"family": family, "rule": rule, "parent": parent, "sql": sql, "priority": rank})
    pool = []
    for item in sorted(raw, key=lambda c: -c["priority"]):
        family, sql = item["family"], item["sql"]
        key = identity(sql)
        if key in seen:
            counts[family + "_deduplicated"] += 1
            continue
        try:
            validate_sql(sql)
        except ValueError:
            counts[family + "_invalid"] += 1
            continue
        seen.add(key)
        pool.append(item)
    return pool, dict(counts)


def coupled_edits(tree, schema, headings, question, feedback):
    for rule, query, rank in predicate_edits(tree, schema, question, feedback):
        if isinstance(query, exp.Select):
            for _, corrected, _ in projection_edit(query, schema, headings):
                yield rule + "_with_projection", corrected, rank + 6


def public_role_conflict(obs, schema, baseline):
    """Only probe a second draft when the same literal has a better supported role."""
    tree = parse(baseline)
    if not isinstance(tree, exp.Select):
        return None
    words = tokens(obs["task"]["question"])
    old_atoms = atoms(tree.args.get("where"))
    for event in obs["history"]:
        if event["tool"] != "check" or not event["observation"]["ok"]:
            continue
        source = parse(event["sql"])
        if not isinstance(source, exp.Select):
            continue
        for new in atoms(source.args.get("where")):
            value = new.expression
            if not isinstance(value, exp.Literal):
                continue
            positions = [(i, i + len(tokens(value.this))) for i in local_occurrences(words, value.this)]
            new_field = new.this.name.lower()
            for old in old_atoms:
                if (not isinstance(old.expression, exp.Literal) or old.expression.this != value.this
                        or new_field == old.this.name.lower()
                        or column_evidence(words, new_field, positions) <= column_evidence(
                            words, old.this.name, positions)):
                    continue
                sources = top_tables(source, schema)
                binding = existing(new.this, sources, schema)
                if not binding or not declared_joins(source, sources, schema):
                    continue
                for rule, candidate, priority in quoted_role_filter(
                        source, schema, obs["example"]["columns"], obs["task"]["question"]):
                    return {"family": "role", "rule": "public_" + rule, "priority": priority,
                            "parent": event["sql"], "sql": sql_text(candidate),
                            "conflict": [old.this.sql(), new.this.sql(), value.this]}
    return None


def declared_joins(query, sources, schema):
    for join in query.args.get("joins") or []:
        on = join.args.get("on")
        if not isinstance(on, exp.EQ) or not all(isinstance(c, exp.Column) for c in (on.this, on.expression)):
            return False
        left, right = [existing(c, sources, schema) for c in (on.this, on.expression)]
        if not left or not right or (left[1], left[2], right[1], right[2]) not in schema.edges[left[1]]:
            return False
    return True
