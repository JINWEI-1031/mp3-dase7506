"""Remove a temporal-category filter incorrectly copied from an explicit relationship role."""
from sqlglot import exp

from repair_rules import parse, sql_text
from step4_agent import identity, passed
from step4_edits import Schema
from step10_bindings import tokens
from step38.event_grain import bind_column, join_graph
from step85_agent import create_agent as create_baseline


TIME_FIELDS = {"cycle", "year", "month", "semester", "term"}


def role_filter(observation, sql):
    tree = parse(sql)
    if (not isinstance(tree, exp.Select)
            or any(tree.args.get(key) for key in ("group", "having", "order", "limit", "offset", "with"))
            or len(tree.args.get("joins", [])) != 1):
        return None
    where = tree.args.get("where")
    atom = where.this if where else None
    if (not isinstance(atom, exp.EQ) or not isinstance(atom.this, exp.Column)
            or not isinstance(atom.expression, exp.Literal) or not atom.expression.is_string):
        return None
    words = tokens(observation["task"]["question"])
    temporal = set(tokens(atom.this.name)) & TIME_FIELDS
    if not temporal or set(words) & TIME_FIELDS:
        return None
    tables = next((e["observation"] for e in observation["history"] if e["tool"] == "schema"), None)
    if tables is None:
        return None
    schema = Schema(tables)
    sources = schema.sources(tree)
    owner = bind_column(atom.this, schema, sources)
    edges = join_graph(tree, schema, sources)
    if owner is None or len(edges or []) != 1:
        return None
    literal = tokens(atom.expression.this)
    if len(literal) != 1:
        return None
    evidence = []
    for a, key, b, reference in edges:
        for table, field, other, linked in ((a, key, b, reference), (b, reference, a, key)):
            role = tokens(field)
            foreign = any(fk["from"].casefold() == field and fk["table"].casefold() == other
                          and fk["to"].casefold() == linked for fk in schema.tables[table]["foreign_keys"])
            if (table == owner[0] and foreign and len(role) >= 3 and role[:1] == literal
                    and any(words[i:i + len(role)] == role for i in range(len(words) - len(role) + 1))):
                evidence.append((table, field))
    if len(evidence) != 1:
        return None
    query = tree.copy()
    query.set("where", None)
    return {"sql": sql_text(query), "kind": "role_literal_in_time_field",
            "role_binding": evidence[0], "removed_field": owner}


class RoleFilterSuffix:
    def __init__(self, fallback):
        self.fallback = fallback
        self.candidate = None
        self.evidence = None

    def act(self, observation):
        if self.candidate:
            return {"tool": "submit", "sql": self.candidate} if passed(
                observation["history"][-1]["observation"]) else self.fallback
        if not observation["remaining"]:
            return self.fallback
        candidate = role_filter(observation, self.fallback["sql"])
        if candidate is None:
            return self.fallback
        self.evidence = candidate
        previous = next((e["observation"] for e in observation["history"] if e["tool"] == "check"
                         and identity(e["sql"]) == identity(candidate["sql"])), None)
        if previous is not None:
            return {"tool": "submit", "sql": candidate["sql"]} if passed(previous) else self.fallback
        self.candidate = candidate["sql"]
        return {"tool": "check", "sql": self.candidate}


class Agent:
    def __init__(self, baseline):
        self.baseline = baseline
        self.suffix = None

    def act(self, observation):
        if self.suffix is None:
            action = self.baseline.act(observation)
            if action["tool"] != "submit":
                return action
            self.suffix = RoleFilterSuffix(action)
        return self.suffix.act(observation)


def create_agent(checkpoint=None):
    return Agent(create_baseline(checkpoint))
