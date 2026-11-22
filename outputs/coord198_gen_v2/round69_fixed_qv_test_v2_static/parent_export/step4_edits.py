"""Schema-grounded edits shared by step 4 candidate sources."""
from collections import defaultdict, deque
import re

from sqlglot import exp

from repair_rules import error_message, identifier_edits, parse, sql_text

COMPARISONS = {">": exp.GT, ">=": exp.GTE, "<": exp.LT, "<=": exp.LTE}
THRESHOLD = re.compile(
    r"\b(more than|greater than|older than|above|over|at least|less than|below|"
    r"under|at most) (-?\d+(?:\.\d+)?)\b", re.I)
OPERATORS = {"more than": ">", "greater than": ">", "older than": ">", "above": ">",
             "over": ">", "at least": ">=", "less than": "<", "below": "<",
             "under": "<", "at most": "<="}
FREQUENCY = re.compile(
    r"\b((?:most|least) (?:common|frequent)|(?:most|least) frequently|"
    r"(?:most|highest|fewest|least|lowest) number of)\b", re.I)
ABSENCE = re.compile(r"\b(?:do(?:es)? not have|don't have|without|no associated)\b", re.I)


def simple(sql):
    tree = parse(sql)
    if isinstance(tree, exp.Select) and len(list(tree.find_all(exp.Select))) == 1:
        return tree
    return None


def local_columns(tree):
    return [column for column in tree.find_all(exp.Column) if not column.is_star]


class Schema:
    def __init__(self, tables):
        self.tables = {table["name"].lower(): table for table in tables}
        self.columns = {name: {c["name"].lower(): c for c in table["columns"]}
                        for name, table in self.tables.items()}
        self.edges = defaultdict(list)
        for left, table in self.tables.items():
            for key in table["foreign_keys"]:
                right = key["table"].lower()
                a, b = key["from"].lower(), key["to"].lower()
                if left != right and a in self.columns[left] and b in self.columns.get(right, {}):
                    self.edges[left].append((left, a, right, b))
                    self.edges[right].append((right, b, left, a))

    def sources(self, tree):
        sources = {}
        for source in tree.find_all(exp.Table):
            if source.db or source.catalog or source.name.lower() not in self.tables:
                return {}
            name = source.name.lower()
            if name in sources or any(a.lower() == source.alias_or_name.lower()
                                      for a in sources.values()):
                return {}
            sources[name] = source.alias_or_name
        return sources

    def column(self, table, name, alias):
        return exp.column(self.columns[table][name]["name"], table=alias, quoted=True)

    def path_options(self, sources, target, distance):
        queue = deque((table, []) for table in sources)
        while queue:
            vertex, path = queue.popleft()
            if vertex == target:
                yield path
                continue
            if len(path) == distance:
                continue
            visited = {*sources, *(edge[2] for edge in path)}
            for edge in self.edges[vertex]:
                if edge[2] not in visited:
                    queue.append((edge[2], [*path, edge]))

    def related(self, sources, distance):
        related = set(sources)
        frontier = set(sources)
        for _ in range(distance):
            frontier = {edge[2] for name in frontier for edge in self.edges[name]} - related
            related.update(frontier)
        return related


def join_repairs(tree, schema, feedback, reach="bridge"):
    """Resolve a reported missing binding by adding only supported FK paths."""
    message = error_message(feedback)
    if not message.startswith("no such column: "):
        return
    token = message.split(": ", 1)[1].strip().lower()
    qualifier, _, name = token.rpartition(".")
    sources = schema.sources(tree)
    if not sources or tree.args.get("with"):
        return
    missing = [column for column in local_columns(tree)
               if column.name.lower() == name and column.table.lower() == qualifier]
    if not missing or qualifier in {alias.lower() for alias in sources.values()}:
        return
    reach_limit = {"related": 1, "bridge": 2, "full": 3}[reach]
    related = schema.related(sources, reach_limit) if reach != "full" else schema.tables
    options = []
    for target, cols in schema.columns.items():
        if target in sources or target not in related or name not in cols:
            continue
        for path in schema.path_options(sources, target, reach_limit):
            options.append((qualifier != target, len(path), path))
    if not options:
        return
    options.sort()
    best = options[0][:2]
    for _, _, path in list(dict.fromkeys((a, b, tuple(c)) for a, b, c in options
                                         if (a, b) == best))[:2]:
        result = tree.copy()
        aliases = dict(sources)
        for left, a, right, b in path:
            alias = "s4_" + str(len(aliases))
            while alias.lower() in {value.lower() for value in aliases.values()}:
                alias += "_"
            aliases[right] = alias
            result.join(exp.Table(this=exp.to_identifier(schema.tables[right]["name"], quoted=True),
                                  alias=exp.TableAlias(this=exp.to_identifier(alias))),
                        on=exp.EQ(this=schema.column(left, a, aliases[left]),
                                  expression=schema.column(right, b, alias)), copy=False)
        target = path[-1][2]
        for column in local_columns(result):
            if column.name.lower() == name and column.table.lower() == qualifier:
                column.replace(schema.column(target, name, aliases[target]))
        yield ("join/binding", sql_text(result),
               {"error": message, "foreign_keys": path})


def threshold_edits(tree, question):
    requests = defaultdict(set)
    for wording, value in THRESHOLD.findall(question):
        requests[value].add(OPERATORS[wording.lower()])
    for index, node in enumerate(tree.walk()):
        if (type(node) not in COMPARISONS.values() or not isinstance(node.expression, exp.Literal)
                or node.expression.is_string or not isinstance(node.this, (exp.Column, exp.AggFunc))):
            continue
        value = node.expression.this
        operators = requests[value]
        if len(operators) != 1:
            continue
        operator = COMPARISONS[next(iter(operators))]
        if isinstance(node, operator):
            continue
        result = tree.copy()
        position = list(result.walk())[index]
        position.replace(operator(this=position.this.copy(), expression=position.expression.copy()))
        yield ("predicate/comparator", sql_text(result), {"from": node.sql(), "threshold": value})


def frequency_edit(tree, question, arity):
    match = FREQUENCY.search(question)
    order = tree.args.get("order")
    if (match is None or not order or len(tree.selects) != arity or arity != 1
            or not isinstance(tree.selects[0], exp.Column) or tree.args.get("having")
            or tree.args.get("distinct") or tree.args.get("offset")
            or tree.args.get("limit") is None or tree.args["limit"].expression.name != "1"
            or order.find(exp.AggFunc)):
        return
    result = tree.copy()
    if not result.args.get("group"):
        result.set("group", exp.Group(expressions=[result.selects[0].copy()]))
    desc = not match[0].lower().startswith(("least", "fewest", "lowest"))
    result.set("order", exp.Order(expressions=[exp.Ordered(this=exp.Count(this=exp.Star()),
                                                          desc=desc, nulls_first=not desc)]))
    yield ("group/order", sql_text(result), {"phrase": match[0]})


def grouped_sum_edit(tree, question):
    match = re.search(r"\b(?:most|highest|least|lowest)\b", question, re.I)
    order = tree.args.get("order")
    if (not match or not order or len(order.expressions) != 1 or len(tree.selects) != 1
            or not isinstance(tree.selects[0], exp.Column) or tree.args.get("group")
            or tree.args.get("joins") or tree.args.get("distinct") or tree.args.get("offset")
            or not tree.args.get("limit") or tree.args["limit"].expression.name != "1"):
        return
    ordered = order.expressions[0]
    if (not isinstance(ordered.this, exp.Column)
            or ordered.this.name.lower() == tree.selects[0].name.lower()
            or not set(re.findall(r"[a-z]+", ordered.this.name.lower()))
            & set(re.findall(r"[a-z]+", question.lower()))):
        return
    result = tree.copy()
    result.group_by(result.selects[0].copy(), copy=False)
    result.args["order"].expressions[0].set("this", exp.Sum(this=ordered.this.copy()))
    yield ("group/order", sql_text(result), {"phrase": match[0], "measure": ordered.this.sql()})


def absence_edits(tree, schema, question):
    """Turn a missing related-column NULL test into a FK-correlated anti-join."""
    match = ABSENCE.search(question)
    where = tree.args.get("where")
    sources = schema.sources(tree)
    if (match is None or len(sources) != 1 or not where or tree.args.get("group")
            or not isinstance(where.this, exp.Is) or not isinstance(where.this.this, exp.Column)
            or not isinstance(where.this.expression, exp.Null)):
        return
    base, alias = next(iter(sources.items()))
    words = set(re.findall(r"[a-z]+", question[match.end():].lower()))
    words = {word.rstrip("s") for word in words if len(word) > 2}
    options = []
    for _, lhs, target, rhs in schema.edges[base]:
        overlap = words & {word.rstrip("s") for word in re.findall(r"[a-z]+", target)}
        if overlap:
            options.append((lhs, target, rhs, sorted(overlap)))
    for lhs, target, rhs, overlap in options[:2]:
        result = tree.copy()
        inner_alias = "s4_child" if alias.lower() != "s4_child" else "s4_related"
        table = exp.Table(this=exp.to_identifier(schema.tables[target]["name"], quoted=True),
                          alias=exp.TableAlias(this=exp.to_identifier(inner_alias)))
        predicate = exp.EQ(this=schema.column(base, lhs, alias),
                           expression=schema.column(target, rhs, inner_alias))
        inner = exp.select(exp.Literal.number(1)).from_(table).where(predicate)
        result.set("where", exp.Where(this=exp.Not(this=exp.Exists(this=inner))))
        yield ("where/anti_join", sql_text(result), {"phrase": match[0], "terms": overlap,
                                                    "foreign_key": [base, lhs, target, rhs]})


def hint_feedback(sql, feedback, tables, schema, question, arity):
    for kind, candidate in identifier_edits(sql, feedback, tables):
        yield ("identifier/" + kind, candidate, {"error": error_message(feedback)})
    tree = simple(sql)
    if tree is not None:
        yield from join_repairs(tree, schema, feedback)
        if feedback["ok"]:
            yield from threshold_edits(tree, question)
            yield from frequency_edit(tree, question, arity)
            yield from grouped_sum_edit(tree, question)


def hint_delta(draft, hint, arity):
    """Apply a small clause edit while retaining unrelated draft clauses."""
    old, new = simple(draft), simple(hint)
    if old is None or new is None:
        return
    for keys in (("expressions",), ("where",), ("group", "having", "order", "limit"),
                 ("from", "joins", "where")):
        if all(old.args.get(key) == new.args.get(key) for key in keys):
            continue
        if "expressions" in keys and (len(new.selects) != arity or len(old.selects) == arity):
            continue
        if "from" not in keys and any(old.args.get(key) != new.args.get(key)
                                      for key in ("from", "joins")):
            continue
        if "from" in keys and old.args.get("from") == new.args.get("from") and (
                old.args.get("joins") == new.args.get("joins")):
            continue
        result = old.copy()
        for key in keys:
            node = new.args.get(key)
            value = [item.copy() for item in node] if isinstance(node, list) else (
                node.copy() if node is not None else None)
            result.set(key, value)
        if len(result.selects) == arity:
            yield ("/".join(keys), sql_text(result), {"changed_clauses": list(keys),
                                                     "evidence": "hint vs draft AST"})


def key_edits(tree, schema):
    """Suggest declared keys for simple equality joins with unsupported bindings."""
    sources = schema.sources(tree)
    if len(sources) < 2:
        return
    aliases = {alias.lower(): name for name, alias in sources.items()}
    for index, join in enumerate(tree.args.get("joins", [])):
        on = join.args.get("on")
        if not isinstance(on, exp.EQ) or not all(
                isinstance(side, exp.Column) and side.table.lower() in aliases
                for side in (on.this, on.expression)):
            continue
        left, right = (aliases[side.table.lower()] for side in (on.this, on.expression))
        if join.this.alias_or_name.lower() not in {on.this.table.lower(), on.expression.table.lower()}:
            continue
        options = sorted({(a, b) for _, a, target, b in schema.edges[left] if target == right})
        if (on.this.name.lower(), on.expression.name.lower()) in options:
            continue
        for a, b in options[:2]:
            result = tree.copy()
            result.args["joins"][index].set(
                "on", exp.EQ(this=schema.column(left, a, sources[left]),
                             expression=schema.column(right, b, sources[right])))
            yield ("join/on", sql_text(result),
                   {"from": on.sql(), "foreign_key": [left, a, right, b]})
