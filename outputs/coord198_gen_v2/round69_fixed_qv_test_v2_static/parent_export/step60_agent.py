"""Compile alternatives with predicates bound to their own branch."""
import re

from sqlglot import exp

from repair_rules import parse, sql_text
from step4_agent import identity, passed
from step36.frequency import terms
from step48_agent import create_agent as create_baseline


NEAREST = re.compile(r"\beither\s+(\d+)\s+([a-z]+)\s+or\s+(\d+)\s+([a-z]+)\s+but\s+(\d+)\s+([a-z]+)\s*[?.!]?$", re.I)
EXTREME = re.compile(r"\b(largest|smallest)\s+value\s+or\s+with value having char\s+(['\"])([a-z0-9])\2\s+in it\s*[?.!]?$", re.I)


def intent_for(question):
    local = NEAREST.search(question)
    if local and local[1] != local[3] and terms(local[2]) == terms(local[4]):
        return "branch_local_constraint", local
    extreme = EXTREME.search(question)
    return ("unfiltered_extremum_or_match", extreme) if extreme else None


def simple_tree(sql, headings):
    tree = parse(sql)
    if (not isinstance(tree, exp.Select) or len(tree.expressions) != 1 or len(headings) != 1
            or any(tree.args.get(k) for k in ("joins", "group", "having", "order", "limit", "offset", "with"))):
        return None
    frm = tree.args.get("from")
    where = tree.args.get("where")
    if not frm or not isinstance(frm.this, exp.Table) or not where:
        return None
    return tree, where.this.unnest()


def branches_local(tree, where, heads, cue):
    selected = tree.expressions[0].unalias()
    if (not isinstance(selected, exp.Column) or selected.name.casefold() != heads[0].casefold()
            or not heads[0].casefold().endswith("name") or not isinstance(where, exp.And)):
        return None
    left, right = where.this.unnest(), where.expression.unnest()
    if isinstance(right, exp.In):
        left, right = right, left
    if (not isinstance(left, exp.In) or left.args.get("query") or len(left.expressions) != 2
            or not isinstance(right, exp.EQ) or not isinstance(left.this, exp.Column) or not isinstance(right.this, exp.Column)
            or not all(isinstance(value, exp.Literal) for value in [*left.expressions, right.expression])):
        return None
    if (terms(left.this.name) != terms(cue[2]) or terms(right.this.name) != terms(cue[6])
            or {value.this for value in left.expressions} != {cue[1], cue[3]} or right.expression.this != cue[5]):
        return None
    # Reuse each typed literal instead of converting it based on its spelling.
    first, second = [next(value for value in left.expressions if value.this == cue[pos]) for pos in (1, 3)]
    a = tree.copy().where(exp.EQ(this=left.this.copy(), expression=first.copy()), append=False, copy=False)
    b = tree.copy().where(exp.EQ(this=left.this.copy(), expression=second.copy()), right.copy(), append=False, copy=False)
    return exp.union(a, b)


def branches_extreme(tree, where, heads, cue):
    reducer = exp.Max if cue[1].casefold() == "largest" else exp.Min
    projected = tree.expressions[0].unalias()
    alias = tree.args["from"].this.alias_or_name.casefold()
    if (not isinstance(projected, reducer) or not isinstance(projected.this, exp.Column)
            or not isinstance(where, exp.Like) or not isinstance(where.this, exp.Column)
            or projected.this.name.casefold() != where.this.name.casefold()
            or any(column.table.casefold() not in ("", alias) for column in (projected.this, where.this))
            or not isinstance(where.expression, exp.Literal) or not where.expression.is_string
            or where.expression.this != f"%{cue[3]}%"):
        return None
    expected = re.fullmatch(r"\s*(max|min)\s*\(\s*([\w]+)\s*\)\s*", heads[0], re.I)
    if not expected or (expected[1].casefold(), expected[2].casefold()) != (reducer.key, projected.this.name.casefold()):
        return None
    extreme = tree.copy()
    extreme.set("where", None)
    matches = tree.copy().select(projected.this.copy(), append=False, copy=False)
    return exp.union(extreme, matches)


class BranchSuffix:
    def __init__(self, fallback):
        self.fallback = fallback
        self.candidate = None

    def act(self, observation):
        history, remaining = observation["history"], observation["remaining"]
        if self.candidate:
            return {"tool": "submit", "sql": self.candidate["sql"]} if passed(history[-1]["observation"]) else self.fallback
        intent = intent_for(observation["task"]["question"]) if remaining else None
        if not intent:
            return self.fallback
        mode, cue = intent
        heads = observation["example"]["columns"]
        parent = simple_tree(self.fallback["sql"], heads)
        if not parent:
            return self.fallback
        compiler = branches_local if mode == "branch_local_constraint" else branches_extreme
        query = compiler(*parent, heads, cue)
        if query is None:
            return self.fallback
        sql = sql_text(query)
        seen = {identity(e["sql"]) for e in history if e["tool"] == "check"}
        if identity(sql) in seen:
            return self.fallback
        self.candidate = {"sql": sql, "kind": mode}
        return {"tool": "check", "sql": sql}


class Agent:
    def __init__(self, baseline):
        self.baseline = baseline
        self.suffix = None

    def act(self, observation):
        if self.suffix is None:
            action = self.baseline.act(observation)
            if action["tool"] != "submit":
                return action
            self.suffix = BranchSuffix(action)
        return self.suffix.act(observation)


def create_agent(checkpoint=None):
    return Agent(create_baseline(checkpoint))
