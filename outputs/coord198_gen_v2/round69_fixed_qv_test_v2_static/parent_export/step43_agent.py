"""Ask for schema only when an unresolved binding can unlock a check."""
from pathlib import Path

from sqlglot import exp

from repair_rules import parse
from step25_agent import create_agent as create_baseline
from step32_plan import context
from step33.relations import spec
from step36.frequency import COUNT_HEADER, infer as frequency_intent
from step37.membership import infer as membership_intent
from step38.agent import Suffix
from step38.event_grain import terms


def event_can_bind(parent, request):
    joins = parent.args["joins"]
    sources = [parent.args["from"].this, *(join.this for join in joins)] if parent.args.get("from") else []
    if not sources or any(not isinstance(source, exp.Table) for source in sources):
        return False
    if any(join.side or join.kind not in ("", "INNER") or not isinstance(join.args.get("on"), exp.EQ)
           or not all(isinstance(term, exp.Column) for term in (join.args["on"].this, join.args["on"].expression))
           for join in joins):
        return False
    outputs = set(request["output_terms"])
    witnesses = set(request["witness_terms"])
    where = parent.args.get("where")
    if where and any(isinstance(node, (exp.Or, exp.Not, exp.Func)) and not isinstance(node, exp.And)
                     for node in where.walk()):
        return False
    fields = list(where.find_all(exp.Column)) if where else []
    for root in sources:
        if not terms(root.name) & outputs:
            continue
        for source in sources:
            if source is root:
                continue
            named = terms(source.name) - terms(root.name)
            for field in fields:
                if not field.table or field.table.casefold() == source.alias_or_name.casefold():
                    named |= terms(field.name)
            if named & witnesses:
                return True
    return False


def actionable_bindings(observation, sql, requests):
    parent = parse(sql)
    visible = context(observation)
    question, headings = visible["question"], visible["columns"]
    public = any(result["ok"] and result["matches_example"] for _, result in visible["checks"])
    expected = [heading.casefold() for heading in headings]
    actual = [col.alias_or_name.casefold() for col in parent.expressions] if isinstance(parent, exp.Select) else []
    proven = []
    event = requests.get("event_grain")
    if event and event_can_bind(parent, event):
        proven.append(event["bind"])
    for request in requests.get("inherited", []):
        if request == "baseline quantified role":
            proven.append(request)
        elif request == "baseline graph binding":
            shape = spec(parent, question, len(headings))
            if shape and (public or shape[2] or expected != actual):
                proven.append(request)
        elif request == "question event and output roles":
            intent = frequency_intent(question, headings)
            if intent and (not public or intent.mode != "count per label"
                           or actual and actual[0] != expected[0] and COUNT_HEADER.match(expected[0])):
                proven.append(request)
        elif request == "named entity and event relationship":
            intent = membership_intent(question, headings)
            if intent and intent.subject and (intent.event or intent.operator):
                proven.append(request)
    return proven


class SchemaSuffix(Suffix):
    def __init__(self, fallback):
        super().__init__(fallback, use_new=True, allow_schema=True)

    def act(self, observation):
        action = super().act(observation)
        if action["tool"] != "schema":
            return action
        requests = self.audit["binding_request"]
        self.allow_schema = False
        ready = super().act(observation)
        self.allow_schema = True
        has_ready_check = ready["tool"] == "check"
        bindings = [] if has_ready_check else actionable_bindings(observation, self.fallback["sql"], requests)
        self.audit["schema_gate"] = {"bindings": bindings, "ready": has_ready_check}
        if has_ready_check or not bindings:
            return ready
        return action


class Agent:
    def __init__(self, baseline):
        self.baseline = baseline
        self.suffix = None

    def act(self, observation):
        if self.suffix is None:
            action = self.baseline.act(observation)
            if action["tool"] != "submit":
                return action
            self.suffix = SchemaSuffix(action)
        return self.suffix.act(observation)


def create_agent(checkpoint=None):
    config = checkpoint or Path(__file__).with_name("configs") / "step25-exists.json"
    return Agent(create_baseline(config))
