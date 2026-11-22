"""Test witness-row output when a prior repair collapsed a plain entity projection."""
from sqlglot import exp

from repair_rules import parse, sql_text
from step4_agent import identity, passed, usable
from step4_edits import Schema
from step21_carry import conjuncts
from step38.event_grain import bind_column, intent, join_graph, terms
from step83_agent import create_agent as create_baseline


def witness_rows(observation, sql):
    roles = intent(observation["task"]["question"])
    tree = parse(sql) if roles else None
    draft = parse(observation["issue_sql"])
    if (not isinstance(tree, exp.Select) or not isinstance(draft, exp.Select)
            or draft.args.get("distinct") or not tree.args.get("distinct")
            or any(tree.args.get(key) for key in ("joins", "group", "having", "order", "limit", "offset", "with"))):
        return None
    outer_from = tree.args.get("from")
    where = tree.args.get("where")
    witness = where.this if where else None
    if (outer_from is None or not isinstance(outer_from.this, exp.Table)
            or not isinstance(witness, exp.Exists) or not isinstance(witness.this, exp.Select)):
        return None
    inner = witness.this
    inner_from = inner.args.get("from")
    if (inner_from is None or not isinstance(inner_from.this, exp.Table)
            or any(inner.args.get(key) for key in ("distinct", "group", "having", "order", "limit", "offset", "with"))
            or len(list(inner.find_all(exp.Select))) != 1 or not inner.args.get("where")):
        return None
    tables = next((event["observation"] for event in observation["history"] if event["tool"] == "schema"), None)
    if tables is None:
        return None
    schema = Schema(tables)
    sources = schema.sources(tree)
    root = outer_from.this.name.casefold()
    child = inner_from.this.name.casefold()
    if (root == child or len(sources) != 2 + len(inner.args.get("joins", []))
            or not roles[0] & terms(root)):
        return None
    if (len(tree.expressions) != len(observation["example"]["columns"])
            or [item.alias_or_name.casefold() for item in tree.expressions]
            != [head.casefold() for head in observation["example"]["columns"]]
            or any((bound := bind_column(item.unalias(), schema, sources)) is None or bound[0] != root
                   for item in tree.expressions)):
        return None
    correlations, filters = [], []
    for atom in conjuncts(inner.args["where"].this):
        if (isinstance(atom, exp.EQ) and isinstance(atom.this, exp.Column)
                and isinstance(atom.expression, exp.Column)):
            a, b = bind_column(atom.this, schema, sources), bind_column(atom.expression, schema, sources)
            if a and b and {a[0], b[0]} == {root, child} and (*a, *b) in schema.edges[a[0]]:
                correlations.append(atom)
                continue
        if any(bind_column(col, schema, sources) is None or
               bind_column(col, schema, sources)[0] == root for col in atom.find_all(exp.Column)):
            return None
        if atom.find(exp.Select):
            return None
        filters.append(atom)
    if len(correlations) != 1:
        return None
    query = tree.copy()
    query.set("distinct", None)
    query.set("where", None)
    query.join(inner_from.this.copy(), on=correlations[0].copy(), copy=False)
    for join in inner.args.get("joins", []):
        query.append("joins", join.copy())
    if filters:
        query.where(*(atom.copy() for atom in filters), copy=False)
    if not join_graph(query, schema, schema.sources(query)):
        return None
    return {"sql": sql_text(query), "kind": "restore_witness_rows",
            "root": root, "witness_tables": sorted(set(sources) - {root})}


class WitnessRowsSuffix:
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
        candidate = witness_rows(observation, self.fallback["sql"])
        if candidate is None or not usable(candidate["sql"]):
            return self.fallback
        self.evidence = candidate
        previous = next((event["observation"] for event in observation["history"] if event["tool"] == "check"
                         and identity(event["sql"]) == identity(candidate["sql"])), None)
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
            self.suffix = WitnessRowsSuffix(action)
        return self.suffix.act(observation)


def create_agent(checkpoint=None):
    return Agent(create_baseline(checkpoint))
