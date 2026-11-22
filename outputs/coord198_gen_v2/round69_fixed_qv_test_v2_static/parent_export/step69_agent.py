"""Check one deterministic repair for an explicit conflict with the question."""
from repair_rules import parse, sql_text
from step4_agent import identity, passed
from step60_agent import branches_local, intent_for, simple_tree
from step63_agent import create_agent as create_baseline
from step66_agent import conflict_for


class ConflictSuffix:
    def __init__(self, fallback):
        self.fallback = fallback
        self.evidence = None
        self.candidate = None

    def act(self, observation):
        history = observation["history"]
        if self.candidate is not None:
            if passed(history[-1]["observation"]):
                return {"tool": "submit", "sql": self.candidate}
            return self.fallback
        if not observation["remaining"]:
            return self.fallback
        sql = self.fallback["sql"]
        checks = {identity(event["sql"]): event["observation"] for event in history
                  if event["tool"] == "check"}
        result = checks.get(identity(sql))
        if result is not None and passed(result):
            self.evidence = conflict_for(observation, sql)
        elif not any(passed(check) for check in checks.values()):
            self.evidence = unmatched_branch_conflict(observation, sql)
        if self.evidence is None:
            return self.fallback
        if self.evidence["kind"] == "order_direction":
            tree = parse(sql)
            order = tree.args["order"].expressions[0]
            descending = self.evidence["descending"]
            order.set("desc", descending)
            order.set("nulls_first", not descending)
            candidate = sql_text(tree)
        else:
            candidate = self.evidence["expected"]
        old_check = checks.get(identity(candidate))
        if old_check is not None:
            if passed(old_check):
                return {"tool": "submit", "sql": candidate}
            return self.fallback
        self.candidate = candidate
        return {"tool": "check", "sql": candidate}


def unmatched_branch_conflict(observation, sql):
    intent = intent_for(observation["task"]["question"])
    if intent is None or intent[0] != "branch_local_constraint":
        return None
    headings = observation["example"]["columns"]
    parent = simple_tree(sql, headings)
    if parent is None:
        return None
    candidate = branches_local(*parent, headings, intent[1])
    if candidate is not None:
        return {"kind": intent[0], "expected": sql_text(candidate)}
    return None


class Agent:
    def __init__(self, baseline):
        self.baseline = baseline
        self.suffix = None

    def act(self, observation):
        if self.suffix is None:
            action = self.baseline.act(observation)
            if action["tool"] != "submit":
                return action
            self.suffix = ConflictSuffix(action)
        return self.suffix.act(observation)


def create_agent(checkpoint=None):
    return Agent(create_baseline(checkpoint))
