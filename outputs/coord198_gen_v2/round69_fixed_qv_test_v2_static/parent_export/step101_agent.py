"""Preaggregate fact rows before a grouped dimension join."""
from sqlglot import exp

from repair_rules import parse, sql_text
from step4_agent import identity, passed, usable
from step4_edits import Schema
from step38.event_grain import bind_column, join_graph
from step100_agent import create_agent as create_baseline


def preaggregate(observation, sql):
    tree = parse(sql)
    if (not isinstance(tree, exp.Select) or len(tree.args.get("joins", [])) != 1
            or any(tree.args.get(k) for k in ("where", "having", "distinct", "with", "offset"))
            or not tree.args.get("group") or len(list(tree.find_all(exp.Select))) != 1):
        return None
    # A changed aggregation plan can choose a different tied winner.
    # Preserve the existing plan for a single-row ranking.
    if tree.args.get("limit") and tree.args["limit"].expression == exp.Literal.number(1):
        return None
    counts = list(tree.find_all(exp.Count))
    if not counts or any(not isinstance(count.this, exp.Star) for count in counts):
        return None
    if any(not isinstance(node, exp.Count) for node in tree.find_all(exp.AggFunc)):
        return None
    tables = next((event["observation"] for event in observation["history"] if event["tool"] == "schema"), None)
    if tables is None:
        return None
    schema = Schema(tables)
    sources = schema.sources(tree)
    if len(sources) != 2 or not join_graph(tree, schema, sources):
        return None
    root = tree.args["from"].this.name.casefold()
    fact = tree.args["joins"][0].this.name.casefold()
    if root == fact:
        return None
    on = tree.args["joins"][0].args["on"]
    bindings = [bind_column(col, schema, sources) for col in (on.this, on.expression)]
    root_key = next(field for table, field in bindings if table == root)
    fact_key = next(field for table, field in bindings if table == fact)
    if not any(fk["from"].casefold() == fact_key and fk["table"].casefold() == root
               and fk["to"].casefold() == root_key for fk in schema.tables[fact]["foreign_keys"]):
        return None
    grouped = [bind_column(col, schema, sources) for col in tree.args["group"].expressions]
    if not all(pair and pair[0] == root for pair in grouped):
        return None
    outside = tree.copy()
    outside.set("joins", None)
    if any((pair := bind_column(col, schema, sources)) is None or pair[0] != root
           for col in outside.find_all(exp.Column)):
        return None
    labels = [bind_column(item.unalias(), schema, sources) for item in tree.expressions
              if not item.find(exp.Count)]
    primary = {name for name, meta in schema.columns[root].items() if meta["pk"]}
    grouping = {field for _, field in grouped}
    if not (primary and primary <= grouping) and any(pair not in grouped for pair in labels):
        return None
    alias = "fact_counts"
    while alias.casefold() in {name.casefold() for name in sources.values()}:
        alias += "_"
    foreign = schema.column(fact, fact_key, sources[fact])
    summary = exp.select(foreign.copy().as_("link_key"), exp.Count(this=exp.Star()).as_("row_count"))
    summary.from_(tree.args["joins"][0].this.copy(), copy=False).group_by(foreign, copy=False)
    result = tree.copy()
    join = result.args["joins"][0]
    join.set("this", summary.subquery(alias))
    join.set("on", exp.EQ(this=schema.column(root, root_key, sources[root]),
                         expression=exp.column("link_key", table=alias)))
    for count in list(result.find_all(exp.Count)):
        if count.find_ancestor(exp.Subquery) is None:
            count.replace(exp.Sum(this=exp.column("row_count", table=alias)))
    return {"sql": sql_text(result), "kind": "preaggregate_fact_rows", "fact": fact,
            "dimension": root, "foreign_key": fact_key}


class AggregateSuffix:
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
        candidate = preaggregate(observation, self.fallback["sql"])
        if candidate is None or not usable(candidate["sql"]):
            return self.fallback
        self.evidence = candidate
        if any(event["tool"] == "check" and identity(event["sql"]) == identity(candidate["sql"])
               for event in observation["history"]):
            return self.fallback
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
            self.suffix = AggregateSuffix(action)
        return self.suffix.act(observation)


def create_agent(checkpoint=None):
    return Agent(create_baseline(checkpoint))
