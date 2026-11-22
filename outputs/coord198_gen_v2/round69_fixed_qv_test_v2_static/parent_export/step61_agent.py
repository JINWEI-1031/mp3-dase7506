"""Check one output repair supported by the public columns and original draft."""
from sqlglot import exp

from repair_rules import parse, sql_text
from step4_agent import identity, passed
from step45_agent import DEDUP
from step48_agent import create_agent as create_baseline


def simple_projection(tree):
    return (isinstance(tree, exp.Select)
            and all(isinstance(field.unalias(), exp.Column) for field in tree.expressions)
            and not any(tree.args.get(slot) for slot in ("group", "having", "order", "limit", "offset",
                                                         "qualify", "with")))


def restores_draft_rows(observation, tree):
    question = observation["task"]["question"]
    distinct = tree.args.get("distinct")
    if (not distinct or distinct.args.get("on") or DEDUP.search(question)
            or not observation["example"]["rows"] or tree.args.get("joins")):
        return False
    draft = parse(observation["issue_sql"])
    if not simple_projection(draft) or draft.args.get("distinct") or not tree.args.get("where"):
        return False
    root = tree.args.get("from")
    return (root is not None and isinstance(root.this, exp.Table)
            and all(not join.side and join.kind in ("", "INNER") and join.args.get("on")
                    and isinstance(join.this, exp.Table) for join in draft.args.get("joins", []))
            and all(draft.args.get(slot) == tree.args.get(slot) for slot in ("expressions", "from", "where")))


def proposal(observation, sql):
    tree = parse(sql)
    if observation["ordered"] or not simple_projection(tree):
        return None
    headers = [name.casefold() for name in observation["example"]["columns"]]
    fields = {field.alias_or_name.casefold(): field for field in tree.expressions}
    if len(fields) != len(tree.expressions) or len(set(headers)) != len(headers) or fields.keys() != set(headers):
        return None
    if list(fields) != headers:
        kind = "public_column_order"
        tree = tree.select(*(fields[name].copy() for name in headers), append=False, copy=False)
    elif restores_draft_rows(observation, tree):
        kind = "restore_draft_multiplicity"
        tree.set("distinct", None)
    else:
        return None
    return {"sql": sql_text(tree), "kind": kind}


class OutputSuffix:
    def __init__(self, fallback):
        self.fallback = fallback
        self.candidate = None

    def act(self, observation):
        history = observation["history"]
        if self.candidate:
            feedback = history[-1]["observation"]
            expected = observation["example"]["columns"]
            if passed(feedback) and [c.casefold() for c in feedback["columns"]] == [c.casefold() for c in expected]:
                return {"tool": "submit", "sql": self.candidate["sql"]}
            return self.fallback
        if not observation["remaining"]:
            return self.fallback
        checked = {identity(event["sql"]): event["observation"] for event in history if event["tool"] == "check"}
        before = checked.get(identity(self.fallback["sql"]))
        if before is None or not passed(before):
            return self.fallback
        candidate = proposal(observation, self.fallback["sql"])
        if candidate is None or identity(candidate["sql"]) in checked:
            return self.fallback
        self.candidate = candidate
        return {"tool": "check", "sql": candidate["sql"]}


class Agent:
    def __init__(self, baseline):
        self.baseline = baseline
        self.suffix = None

    def act(self, observation):
        if self.suffix is None:
            action = self.baseline.act(observation)
            if action["tool"] != "submit":
                return action
            self.suffix = OutputSuffix(action)
        return self.suffix.act(observation)


def create_agent(checkpoint=None):
    return Agent(create_baseline(checkpoint))
