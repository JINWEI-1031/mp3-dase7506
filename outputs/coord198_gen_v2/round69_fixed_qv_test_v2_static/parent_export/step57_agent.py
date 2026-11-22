"""Move an anchored filter along a unique declared many-to-one path."""
import re

from sqlglot import exp

from repair_rules import error_message, parse, sql_text
from step4_agent import identity, passed
from step4_edits import Schema
from step7_semantics import join_path, shortest
from step10_bindings import local_occurrences, tokens
from step26_relations import edge_is_forward
from step48_agent import create_agent as create_baseline


COURSE = re.compile(r"\benrolled in (?:the )?class\s+([a-z]{2,}-\d+)\s*[?.!]?$", re.I)
WIDTH = re.compile(r"(?:var)?char\s*\(\s*(\d+)\s*\)", re.I)


def parent_and_atom(sql):
    tree = parse(sql)
    if (not isinstance(tree, exp.Select) or any(tree.args.get(k) for k in
            ("joins", "group", "having", "order", "limit", "offset", "with", "distinct"))):
        return None
    frm, where = tree.args.get("from"), tree.args.get("where")
    atom = where.this.unnest() if where else None
    if (not frm or not isinstance(frm.this, exp.Table) or not isinstance(atom, exp.EQ)
            or not isinstance(atom.this, exp.Column) or not isinstance(atom.expression, exp.Literal)
            or atom.this.table.casefold() not in ("", frm.this.alias_or_name.casefold())):
        return None
    if any(not (isinstance(p.unalias(), exp.Column) and not p.unalias().is_star
                or isinstance(p.unalias(), exp.Count) and isinstance(p.unalias().this, exp.Star)) for p in tree.expressions):
        return None
    return tree, frm.this, atom


def intent_for(observation):
    old = parent_and_atom(observation["issue_sql"])
    if old is None:
        return None
    tree, _, atom = old
    if len(tree.expressions) != len(observation["example"]["columns"]):
        return None
    question = observation["task"]["question"]
    course = COURSE.search(question)
    if (course and course[1].casefold() == atom.expression.this.casefold()
            and len(tree.expressions) == 1 and isinstance(tree.expressions[0].unalias(), exp.Count)):
        return "course_code", old
    if not observation["history"]:
        return None
    first = observation["history"][0]
    missing = error_message(first["observation"]).casefold() if first["tool"] == "check" else ""
    if (missing.startswith("no such column: ") and missing.removeprefix("no such column: ").split(".")[-1] == atom.this.name.casefold()
            and local_occurrences(tokens(question), atom.this.name)
            and local_occurrences(tokens(question), atom.expression.this)):
        return "missing_filter_field", old
    return None


def single_pk(schema, table, column):
    return [field for field, meta in schema.columns[table].items() if meta["pk"]] == [column]


def declared_paths(schema, start, field):
    for table, cols in schema.columns.items():
        if table == start or field not in cols:
            continue
        paths = shortest(schema, start, table)
        if len(paths) == 1 and all(edge_is_forward(schema, edge) and single_pk(schema, edge[2], edge[3]) for edge in paths[0]):
            yield paths[0], field


def course_paths(schema, start, old_field, literal):
    match = WIDTH.fullmatch(schema.columns[start].get(old_field, {}).get("type", ""))
    if not match or len(literal) <= int(match[1]):
        return
    for edge in schema.edges[start]:
        _, field, target, _ = edge
        if (field != old_field or target != "class" or not edge_is_forward(schema, edge)
                or not single_pk(schema, target, edge[3])):
            continue
        for fk in schema.tables[target]["foreign_keys"]:
            if fk["table"].casefold() != "course":
                continue
            code = fk["from"].casefold()
            size = WIDTH.fullmatch(schema.columns[target].get(code, {}).get("type", ""))
            if size and len(literal) <= int(size[1]) and code != old_field:
                yield (edge,), code


def compile_candidate(intent, tables):
    mode, (tree, table, atom) = intent
    schema = Schema(tables)
    start, field = table.name.casefold(), atom.this.name.casefold()
    if start not in schema.tables:
        return None
    if any(p.unalias().name.casefold() not in schema.columns[start] for p in tree.expressions
           if isinstance(p.unalias(), exp.Column)):
        return None
    choices = list(course_paths(schema, start, field, atom.expression.this) if mode == "course_code"
                   else declared_paths(schema, start, field))
    if len(choices) != 1:
        return None
    path, destination = choices[0]
    aliases = {start: table.alias_or_name}
    query = tree.copy()
    query.set("where", None)
    for index, p in enumerate(query.expressions):
        if isinstance(p.unalias(), exp.Column):
            column = schema.column(start, p.unalias().name.casefold(), aliases[start])
            p.unalias().replace(column) if isinstance(p, exp.Alias) else query.expressions[index].replace(column)
    join_path(query, path, schema, aliases)
    root = path[-1][2]
    query.where(exp.EQ(this=schema.column(root, destination, aliases[root]), expression=atom.expression.copy()), copy=False)
    return {"sql": sql_text(query), "kind": mode, "path": path, "target_field": destination}


class PathSuffix:
    def __init__(self, fallback):
        self.fallback = fallback
        self.requested = False
        self.candidate = None

    def act(self, observation):
        history, remaining = observation["history"], observation["remaining"]
        if self.candidate:
            return {"tool": "submit", "sql": self.candidate["sql"]} if passed(history[-1]["observation"]) else self.fallback
        intent = intent_for(observation) if remaining else None
        if not intent:
            return self.fallback
        tables = next((e["observation"] for e in history if e["tool"] == "schema"), None)
        if tables is None:
            if remaining >= 2 and not self.requested:
                self.requested = True
                return {"tool": "schema"}
            return self.fallback
        candidate = compile_candidate(intent, tables)
        seen = {identity(e["sql"]) for e in history if e["tool"] == "check"}
        if candidate and identity(candidate["sql"]) not in seen | {identity(self.fallback["sql"])}:
            self.candidate = candidate
            return {"tool": "check", "sql": candidate["sql"]}
        return self.fallback


class Agent:
    def __init__(self, baseline):
        self.baseline = baseline
        self.suffix = None

    def act(self, observation):
        if self.suffix is None:
            action = self.baseline.act(observation)
            if action["tool"] != "submit":
                return action
            self.suffix = PathSuffix(action)
        return self.suffix.act(observation)


def create_agent(checkpoint=None):
    return Agent(create_baseline(checkpoint))
