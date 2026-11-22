"""Repair an invalid direct JOIN through a uniquely supported key path."""
import re
from sqlglot import exp

from repair_rules import error_message, parse, sql_text
from step4_agent import identity, passed, usable
from step4_edits import Schema
from step5x_edits import physical_table
from step7_semantics import terms
from step38.event_grain import bind_column
from step40.witness import affinity
from step63_agent import create_agent as create_baseline


def invalid_join(sql, observation):
    tree = parse(sql)
    if (not isinstance(tree, exp.Select)
            or any(value for key, value in tree.args.items()
                   if key not in {"expressions", "from", "joins", "distinct"})
            or len(tree.expressions) != len(observation["example"]["columns"])
            or not all(isinstance(item.unalias(), exp.Column) and not item.unalias().is_star
                       for item in tree.expressions)):
        return None
    joins = tree.args.get("joins", [])
    root = tree.args.get("from")
    if root is None or len(joins) != 1:
        return None
    join = joins[0]
    on = join.args.get("on")
    if (not isinstance(root.this, exp.Table) or not isinstance(join.this, exp.Table)
            or join.side or join.kind not in ("", "INNER") or not isinstance(on, exp.EQ)
            or not all(isinstance(col, exp.Column) and col.table for col in (on.this, on.expression))
            or root.this.db or root.this.catalog or join.this.db or join.this.catalog):
        return None
    roots = (root.this, join.this)
    if roots[0].name.casefold() == roots[1].name.casefold():
        return None
    words = terms(observation["task"]["question"])
    if not all(terms(table.name) & words for table in roots):
        return None
    checked = next((event["observation"] for event in reversed(observation["history"])
                    if event["tool"] == "check" and identity(event["sql"]) == identity(sql)), None)
    message = error_message(checked) if checked is not None else ""
    if not message.startswith("no such column: "):
        return None
    missing = message.removeprefix("no such column: ").casefold()
    if missing not in {f"{col.table}.{col.name}".casefold() for col in (on.this, on.expression)}:
        return None
    return tree, roots, on


def add_named_keys(schema):
    """Supplement an undeclared key only when one table owns that named primary key."""
    owners = {}
    for table, fields in schema.columns.items():
        keys = [name for name, meta in fields.items() if meta["pk"]]
        if len(keys) == 1 and keys[0].endswith("_id"):
            owners.setdefault(keys[0], []).append(table)
    inferred = set()
    for table, fields in schema.columns.items():
        for name, meta in fields.items():
            targets = owners.get(name, [])
            if meta["pk"] or len(targets) != 1 or targets[0] == table:
                continue
            target = targets[0]
            kind = affinity(meta)
            target_meta = schema.columns[target][name]
            decimal_key = (re.fullmatch(r"(?:decimal|numeric)\(\s*\d+\s*,\s*0\s*\)",
                                        meta["type"], re.I)
                           and meta["type"].casefold() == target_meta["type"].casefold())
            if not decimal_key and (kind not in {"integer", "text"} or kind != affinity(target_meta)):
                continue
            if any(edge[1] == name for edge in schema.edges[table]):
                continue
            edge = (table, name, target, name)
            schema.edges[table].append(edge)
            schema.edges[target].append((target, name, table, name))
            inferred.add(frozenset(((table, name), (target, name))))
    return inferred


def compile_bridge(shape, tables):
    tree, roots, on = shape
    schema = Schema(tables)
    sources = schema.sources(tree)
    if len(sources) != 2:
        return None
    outputs = [bind_column(item.unalias(), schema, sources) for item in tree.expressions]
    if not all(outputs) or {table for table, _ in outputs} != sources.keys():
        return None
    aliases = {table.alias_or_name.casefold(): table.name.casefold() for table in roots}
    if {col.table.casefold() for col in (on.this, on.expression)} != aliases.keys():
        return None
    bindings = [(aliases[col.table.casefold()], col.name.casefold()) for col in (on.this, on.expression)]
    if sum(field not in schema.columns[table] for table, field in bindings) != 1:
        return None
    inferred = add_named_keys(schema)
    start, end = (table.name.casefold() for table in roots)
    paths = {tuple(path) for path in schema.path_options([start], end, 3)}
    if len(paths) != 1:
        return None
    (path,) = paths
    if len(path) < 2:
        return None
    observed_table, observed_field = next((table, field) for table, field in bindings
                                         if field in schema.columns[table])
    endpoint = path[0][1] if observed_table == start else path[-1][3]
    if endpoint != observed_field:
        return None
    query = tree.copy()
    query.set("joins", [])
    labels = dict(sources)
    evidence = []
    for left, key, right, reference in path:
        if right not in labels:
            alias = "bridge_" + str(len(labels))
            while alias.casefold() in {value.casefold() for value in labels.values()}:
                alias += "_"
            labels[right] = alias
        condition = exp.EQ(this=schema.column(left, key, labels[left]),
                           expression=schema.column(right, reference, labels[right]))
        query.join(physical_table(schema, right, labels[right]), on=condition, copy=False)
        evidence.append("named_primary_key" if frozenset(((left, key), (right, reference))) in inferred
                        else "declared_foreign_key")
    return {"sql": sql_text(query), "path": path, "edge_evidence": evidence}


class BridgeSuffix:
    def __init__(self, fallback):
        self.fallback = fallback
        self.candidate = None
        self.evidence = None

    def act(self, observation):
        history, remaining = observation["history"], observation["remaining"]
        if self.candidate is not None:
            return {"tool": "submit", "sql": self.candidate} if passed(history[-1]["observation"]) else self.fallback
        if not remaining:
            return self.fallback
        shape = invalid_join(self.fallback["sql"], observation)
        if shape is None:
            return self.fallback
        tables = next((event["observation"] for event in history if event["tool"] == "schema"), None)
        if tables is None:
            return {"tool": "schema"} if remaining >= 2 else self.fallback
        candidate = compile_bridge(shape, tables)
        if candidate is None or not usable(candidate["sql"]):
            return self.fallback
        self.evidence = candidate
        prior = next((event["observation"] for event in history if event["tool"] == "check"
                      and identity(event["sql"]) == identity(candidate["sql"])), None)
        if prior is not None:
            return {"tool": "submit", "sql": candidate["sql"]} if passed(prior) else self.fallback
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
            self.suffix = BridgeSuffix(action)
        return self.suffix.act(observation)


def create_agent(checkpoint=None):
    return Agent(create_baseline(checkpoint))
