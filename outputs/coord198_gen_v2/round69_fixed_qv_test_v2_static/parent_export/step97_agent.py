"""Test root-year scope together with an incorrectly aliased entity label."""
import re

from sqlglot import exp

from repair_rules import parse, sql_text
from step4_agent import identity, passed, usable
from step4_edits import Schema
from step21_carry import conjuncts
from step38.event_grain import bind_column, join_graph
from step82_agent import label_candidate
from step96_agent import create_agent as create_baseline


def label_year_candidate(observation, sql):
    label = label_candidate(observation, sql)
    if label is None:
        return None
    outer = parse(label["sql"])
    where = outer.args.get("where")
    if where is None or not isinstance(where.this, exp.Exists):
        return None
    inner = where.this.this
    if (not isinstance(inner, exp.Select) or len(inner.args.get("joins", [])) != 1
            or any(inner.args.get(key) for key in ("group", "having", "order", "limit", "offset", "distinct", "with"))
            or not inner.args.get("from") or not isinstance(inner.args["from"].this, exp.Table)
            or not inner.args.get("where")):
        return None
    tables = next(event["observation"] for event in observation["history"] if event["tool"] == "schema")
    schema = Schema(tables)
    root = outer.args["from"].this
    inside = inner.args["from"].this
    if root.name.casefold() != inside.name.casefold():
        return None
    sources = schema.sources(inner)
    if len(sources) != 2 or not join_graph(inner, schema, sources):
        return None
    table = root.name.casefold()
    other = next(name for name in sources if name != table)
    filters, year, correlations = [], [], []
    for atom in conjuncts(inner.args["where"].this):
        columns = list(atom.find_all(exp.Column))
        external = [col for col in columns if col.table.casefold() == root.alias_or_name.casefold()]
        if external:
            if (not isinstance(atom, exp.EQ) or len(columns) != 2 or len(external) != 1
                    or columns[0].name.casefold() != columns[1].name.casefold()
                    or not schema.columns[table].get(external[0].name.casefold(), {}).get("pk")):
                return None
            correlations.append(atom)
            continue
        if any(bind_column(col, schema, sources) is None for col in columns):
            return None
        copied = atom.copy()
        if (isinstance(atom, exp.EQ) and isinstance(atom.this, exp.Column)
                and atom.this.name.casefold() == "year"
                and bind_column(atom.this, schema, sources)[0] == other
                and "year" in schema.columns[table] and isinstance(atom.expression, exp.Literal)
                and not atom.expression.is_string and re.search(
                    rf"\b{re.escape(atom.expression.this)}\b", observation["task"]["question"])):
            copied.set("this", schema.column(table, "year", sources[table]))
            year.append(copied)
        filters.append(copied)
    if len(correlations) != 1 or len(year) != 1:
        return None
    ranges = [atom for atom in filters if isinstance(atom, (exp.GTE, exp.LTE))
              and isinstance(atom.this, exp.Column) and atom.this.name.casefold() != "year"
              and bind_column(atom.this, schema, sources)[0] == other]
    if len(ranges) != 2 or ranges[0].this != ranges[1].this:
        return None
    output = [item.copy().transform(lambda node: schema.column(table, node.name.casefold(), sources[table])
                                   if isinstance(node, exp.Column) else node) for item in outer.expressions]
    query = inner.copy().select(*output, append=False)
    query.where(*filters, append=False, copy=False)
    return {"sql": sql_text(query), "kind": "label_and_root_year", "label": label["after"],
            "year_scope": table, "measure_scope": other,
            "heuristic": "test root year when repairing a mislabeled root key"}


class LabelYearSuffix:
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
        candidate = label_year_candidate(observation, self.fallback["sql"])
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
            self.suffix = LabelYearSuffix(action)
        return self.suffix.act(observation)


def create_agent(checkpoint=None):
    return Agent(create_baseline(checkpoint))
