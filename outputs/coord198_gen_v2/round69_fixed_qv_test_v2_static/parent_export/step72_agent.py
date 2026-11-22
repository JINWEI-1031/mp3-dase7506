"""Use the last check for a rule repair when its scheduled SQL has a visible conflict."""

from sqlglot import exp

from mp3_sql.sqlite_tool import validate_sql
from repair_rules import parse
from step4_agent import identity, passed, usable
from step4_edits import Schema
from step26_relations import candidate_pool
from step34.quantities import schedule_pool
from step46_agent import projection_width
from step51_agent import event_side, intent_for as event_intent
from step55_agent import bind_plan, compile_average, intent_for as average_intent
from step56_agent import count_plan, intent_for as role_intent
from step63_agent import create_agent as create_baseline


def final_slot_candidate(observation, scheduled):
    heads = [head.casefold() for head in observation["example"]["columns"]]
    tree = parse(scheduled)
    if tree is None:
        return None
    width = projection_width(scheduled)
    if width is not None and width != len(heads):
        unions = {item["sql"] for item in candidate_pool(observation, observation["issue_sql"])
                  if item["rule"] == "entity_union"}
        if len(unions) == 1:
            return {"kind": "entity_union", "sql": next(iter(unions))}
    if [item.alias_or_name.casefold() for item in tree.selects] != heads:
        projection = schedule_pool(observation)
        if projection:
            return {"kind": "ranked_projection", "sql": projection[0]["sql"]}
    if not isinstance(tree, exp.Select):
        return None

    question = observation["task"]["question"]
    event = event_intent(question)
    role = role_intent(question)
    average = average_intent(question)
    tables = next((item["observation"] for item in observation["history"] if item["tool"] == "schema"), None)
    if tables is None or not (event or role or average):
        return None
    schema = Schema(tables)
    if event and event.direction:
        candidate = event_side(observation, event, schema, parse(observation["issue_sql"]), heads)
        if candidate and candidate["event"] not in {table.name.casefold() for table in tree.find_all(exp.Table)}:
            return candidate
    if role and role[0] == "count" and not tree.args.get("where"):
        return count_plan(observation, schema, *role[1])
    having = tree.args.get("having")
    if average and having and having.find(exp.Count) and not having.find(exp.Avg):
        plan = bind_plan(observation, average, tables)
        if plan:
            return {"kind": "average_instead_of_count", "sql": compile_average(plan)}
    return None


class Agent:
    def __init__(self, baseline):
        self.baseline = baseline
        self.replacement = None

    def act(self, observation):
        history = observation["history"]
        if self.replacement is not None:
            if passed(history[-1]["observation"]):
                return {"tool": "submit", "sql": self.replacement["sql"]}
            action = self.baseline.act(observation)
            return action if action["tool"] == "submit" else {"tool": "submit", "sql": observation["issue_sql"]}

        action = self.baseline.act(observation)
        if observation["remaining"] != 1 or action["tool"] != "check":
            return action
        checks = [event for event in history if event["tool"] == "check"]
        if any(passed(event["observation"]) for event in checks):
            return action
        candidate = final_slot_candidate(observation, action["sql"])
        if candidate is None or not usable(candidate["sql"]):
            return action
        sql = candidate["sql"]
        if identity(sql) in {identity(action["sql"]), *(identity(item["sql"]) for item in checks)}:
            return action
        try:
            tree = validate_sql(sql)
        except ValueError:
            return action
        if len(tree.selects) != len(observation["example"]["columns"]):
            return action
        self.replacement = {**candidate, "displaced": action["sql"]}
        return {"tool": "check", "sql": sql}


def create_agent(checkpoint=None):
    return Agent(create_baseline(checkpoint))
