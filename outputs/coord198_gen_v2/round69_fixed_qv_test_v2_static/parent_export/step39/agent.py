"""One additional eligibility proof for an otherwise idle suffix check."""
from collections import Counter

from sqlglot import exp

from mp3_sql.common import BUDGET
from repair_rules import parse
from step4_agent import identity, passed
from step4_edits import Schema
from step25_agent import create_agent as step25
from step32_plan import context
from step38.agent import Suffix
from step38.event_grain import bind_column

COMPARISONS = (exp.EQ, exp.NEQ, exp.LT, exp.LTE, exp.GT, exp.GTE)
SELECT_KEYS = {"expressions", "from", "joins", "where"}


def conjuncts(tree):
    if isinstance(tree, exp.Paren):
        yield from conjuncts(tree.this)
    elif isinstance(tree, exp.And):
        yield from conjuncts(tree.this)
        yield from conjuncts(tree.expression)
    elif tree is not None:
        yield tree


def plain_select(tree):
    return isinstance(tree, exp.Select) and not any(v for k, v in tree.args.items() if k not in SELECT_KEYS)


def signature(tree, schema, headings):
    """Bind a flat inner join or a single positive correlated witness."""
    if not plain_select(tree):
        return
    sources = schema.sources(tree)
    if not sources or [p.alias_or_name.casefold() for p in tree.expressions] != headings:
        return
    outputs = tuple(bind_column(p.unalias(), schema, sources) for p in tree.expressions)
    if not all(outputs):
        return
    edges, filters = Counter(), Counter()
    scopes = [tree]
    for scope in scopes:
        where = scope.args.get("where")
        joins = scope.args.get("joins", ())
        if any(j.args.get("side") or j.args.get("kind") not in (None, "INNER")
               or not isinstance(j.args.get("on"), exp.EQ) for j in joins):
            return
        atoms = [*(j.args["on"] for j in joins), *conjuncts(where.this if where else None)]
        for atom in atoms:
            if isinstance(atom, exp.Exists) and len(scopes) == 1 and scope is tree and plain_select(atom.this):
                scopes.append(atom.this)
                continue
            if type(atom) not in COMPARISONS:
                return
            a = bind_column(atom.this, schema, sources)
            b = bind_column(atom.expression, schema, sources)
            if a and b and isinstance(atom, exp.EQ) and (*a, *b) in schema.edges[a[0]]:
                edges[tuple(sorted((a, b)))] += 1
            elif a and isinstance(atom.expression, exp.Literal):
                filters[(a[1], atom.key, atom.expression.is_string, atom.expression.this)] += 1
            else:
                return
    if not edges or not filters or len(edges) != len(sources) - 1:
        return
    return outputs, frozenset(sources), edges, filters


def idle_choice(obs, fallback, pool):
    c = context(obs)
    if c["schema"] is None or not obs["remaining"]:
        return None
    checks = {identity(sql): result for sql, result in c["checks"]}
    current = checks.get(identity(fallback["sql"]))
    if not current or not passed(current):
        return None
    schema, headings = Schema(c["schema"]), [h.casefold() for h in c["columns"]]
    original = signature(parse(fallback["sql"]), schema, headings)
    if original is None:
        return None
    for candidate in pool:
        sql = candidate["sql"]
        if candidate["basis"]["eligible"] or identity(sql) in checks:
            continue
        if signature(parse(sql), schema, headings) == original:
            return candidate


class EligibilitySuffix(Suffix):
    def __init__(self, fallback, enabled):
        super().__init__(fallback, use_new=False, allow_schema=False)
        self.enabled = enabled
        self.admitted = None

    def act(self, obs):
        action = super().act(obs)
        if self.enabled and action["tool"] == "submit" and not self.audit["checked"]:
            choice = idle_choice(obs, self.fallback, self.audit["pool"])
            if choice:
                self.admitted = choice
                self.audit.update(checked=[choice["sql"]], selected=choice)
                return {"tool": "check", "sql": choice["sql"]}
        return action


class Agent:
    def __init__(self, baseline):
        self.baseline = baseline
        self.suffix = None

    def act(self, obs):
        if self.suffix is not None:
            return self.suffix.act(obs)
        action = self.baseline.act(obs)
        if action["tool"] != "submit":
            return action
        self.suffix = EligibilitySuffix(action, enabled=True)
        # Use round 38's shared schema request policy, then its old-source pool.
        if len(obs["history"]) <= BUDGET - 2:
            planner = Suffix(action, use_new=True, allow_schema=True)
            schema_action = planner.act(obs)
            if schema_action["tool"] == "schema":
                return schema_action
        return self.suffix.act(obs)


def create_agent(checkpoint):
    return Agent(step25(checkpoint))
