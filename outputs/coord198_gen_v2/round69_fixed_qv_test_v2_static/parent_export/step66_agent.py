"""Request one extra D candidate on independently evidenced public failures."""

import re

from sqlglot import exp

from mp3_sql.sqlite_tool import validate_sql
from repair_rules import parse, sql_text
from step4_agent import identity, passed
from step31_population import source
from step53_agent import already_independent, anchor, comparison_intent
from step60_agent import branches_extreme, branches_local, intent_for, simple_tree
from step63_agent import SYSTEM_PROMPT, active_model, create_agent as create_d, generate_with
from step65_agent import proposal as population_proposal


DIRECTION = re.compile(r"\b(ascending|descending) order\b", re.I)


def conflict_for(observation, fallback):
    question = observation["task"]["question"]
    headings = observation["example"]["columns"]
    population = population_proposal(observation, fallback)
    if population:
        return {"kind": "comparison_population", "expected": population["sql"],
                "hint": f"Apply {population['population_field']} = {population['population_value']!r} "
                        f"inside the {population['reducer'].upper()} comparison subquery."}
    branch = intent_for(question)
    parent = simple_tree(fallback, headings) if branch else None
    if parent:
        kind, cue = branch
        compiled = (branches_local if kind == "branch_local_constraint" else branches_extreme)(*parent, headings, cue)
        if compiled is not None:
            hint = (f"Only the {cue[3]} {cue[4]} branch requires {cue[5]} {cue[6]}."
                    if kind == "branch_local_constraint"
                    else "Compute the extremum without the character filter; apply that filter only to the other branch.")
            return {"kind": kind, "expected": sql_text(compiled), "hint": hint}
    direction = DIRECTION.search(question)
    tree = parse(fallback) if direction and observation["ordered"] else None
    order = tree.args.get("order") if isinstance(tree, exp.Select) else None
    if order and len(order.expressions) == 1:
        descending = direction[1].casefold() == "descending"
        if bool(order.expressions[0].args.get("desc")) != descending:
            return {"kind": "order_direction", "descending": descending,
                    "hint": f"Use the explicitly requested {direction[1].casefold()} ordering."}
    return None


def classify(observation, fallback, primary_reason):
    checks = [e for e in observation["history"] if e["tool"] == "check" and passed(e["observation"])]
    if checks:
        key = identity(fallback)
        if any(identity(e["sql"]) == key for e in checks):
            return conflict_for(observation, fallback)
    elif primary_reason == "budget":
        return {"kind": "no_match_no_D", "hint": ""}
    return None


def comparison_resolved(observation, sql, expected):
    comparator, reducer, role, metric = comparison_intent(observation["task"]["question"])
    width = len(observation["example"]["columns"])
    original, target = anchor(expected, comparator, width)
    bound = anchor(sql, comparator, width)
    if not bound:
        return False
    candidate, atom = bound
    root = source(original).name.casefold()
    if source(candidate).name.casefold() != root or atom.this.name.casefold() != metric:
        return False
    role_field = target.expression.this.args["where"].this.this.name.casefold()
    return already_independent(atom, root, metric, reducer, role.casefold(), role_field)


def branch_signatures(tree):
    if not isinstance(tree, exp.Union) or tree.args.get("distinct") is False:
        return set()
    branches = (tree.this, tree.expression)
    result = set()
    for branch in branches:
        if (not isinstance(branch, exp.Select) or source(branch) is None
                or any(branch.args.get(k) for k in ("joins", "group", "having", "order", "limit", "offset", "with"))):
            return set()
        branch = branch.copy()
        source(branch).set("alias", None)
        for column in branch.find_all(exp.Column):
            column.set("table", None)
        for identifier in branch.find_all(exp.Identifier):
            identifier.set("quoted", False)
        result.add(branch.sql(dialect="sqlite", normalize=True))
    return result


def resolved(observation, sql, evidence):
    kind = evidence["kind"]
    if kind == "no_match_no_D":
        return True
    if kind == "comparison_population":
        return comparison_resolved(observation, sql, evidence["expected"])
    tree = parse(sql)
    if kind == "order_direction":
        order = tree.args.get("order") if isinstance(tree, exp.Select) else None
        return bool(order and len(order.expressions) == 1
                    and bool(order.expressions[0].args.get("desc")) == evidence["descending"])
    expected = branch_signatures(parse(evidence["expected"]))
    return bool(expected and branch_signatures(tree) == expected)


def propose(observation, fallback, schema, evidence):
    draft = {**observation, "issue_sql": fallback}
    hint = evidence["hint"]
    system = SYSTEM_PROMPT + (" Consider this observed discrepancy: " + hint if hint else "")
    return generate_with(draft, schema, active_model(), system_prompt=system, purpose="shadow")


class ShadowSuffix:
    def __init__(self, fallback, evidence, proposer=propose):
        self.fallback = fallback
        self.evidence = evidence
        self.proposer = proposer
        self.raw = None
        self.candidate = None
        self.reason = "no_budget"

    def act(self, observation):
        history, remaining = observation["history"], observation["remaining"]
        if self.candidate is not None:
            if not passed(history[-1]["observation"]):
                self.reason = "public_failed"
            elif not resolved(observation, self.candidate, self.evidence):
                self.reason = "conflict_unresolved"
            else:
                self.reason = "selected"
                return {"tool": "submit", "sql": self.candidate}
            return self.fallback
        if not remaining:
            return self.fallback
        schema = next((e["observation"] for e in history if e["tool"] == "schema"), None)
        if schema is None and remaining >= 2:
            return {"tool": "schema"}
        self.raw = self.proposer(observation, self.fallback["sql"], schema or [], self.evidence)
        try:
            tree = validate_sql(self.raw)
        except ValueError:
            self.reason = "invalid_sql"
            return self.fallback
        seen = {identity(self.fallback["sql"])}
        seen.update(identity(e["sql"]) for e in history if e["tool"] == "check")
        if identity(self.raw) in seen:
            self.reason = "duplicate"
            return self.fallback
        if len(tree.selects) != len(observation["example"]["columns"]):
            self.reason = "projection_width"
            return self.fallback
        self.candidate = self.raw
        return {"tool": "check", "sql": self.candidate}


class Agent:
    def __init__(self, baseline):
        self.baseline = baseline
        self.shadow = None

    def act(self, observation):
        if self.shadow is not None:
            return self.shadow.act(observation)
        action = self.baseline.act(observation)
        if action["tool"] != "submit":
            return action
        evidence = classify(observation, action["sql"], self.baseline.suffix.reason)
        if evidence is None:
            return action
        self.shadow = ShadowSuffix(action, evidence)
        return self.shadow.act(observation)


def create_agent(checkpoint=None):
    return Agent(create_d(checkpoint))
