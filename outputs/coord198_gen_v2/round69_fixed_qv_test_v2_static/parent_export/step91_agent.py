"""Use a final check to disambiguate same-named text columns on a declared join."""
import re

from sqlglot import exp

from repair_rules import parse, sql_text
from step4_agent import identity, passed, usable
from step4_edits import Schema
from step38.event_grain import bind_column, join_graph
from step90_agent import create_agent as create_baseline


def alternative_projection(observation):
    tree = parse(observation["issue_sql"])
    heads = observation["example"]["columns"]
    if (not isinstance(tree, exp.Select) or len(heads) != 1 or len(tree.expressions) != 1
            or any(tree.args.get(key) for key in ("distinct", "group", "having", "order", "limit", "offset", "with"))
            or len(tree.args.get("joins", [])) != 1 or not isinstance(tree.expressions[0], exp.Column)
            or tree.expressions[0].name.casefold() != heads[0].casefold()):
        return None
    executed = next((event["observation"] for event in observation["history"] if event["tool"] == "check"
                     and event["sql"] == observation["issue_sql"]), None)
    if executed is None or not executed["ok"]:
        return None
    tables = next((event["observation"] for event in observation["history"] if event["tool"] == "schema"), None)
    if tables is None:
        return None
    schema = Schema(tables)
    sources = schema.sources(tree)
    selected = bind_column(tree.expressions[0], schema, sources)
    if selected is None or len(sources) != 2 or not join_graph(tree, schema, sources):
        return None
    table, field = selected
    other = next(name for name in sources if name != table)
    if field not in schema.columns[other] or any(
            not re.search(r"text|char|clob", schema.columns[name][field]["type"], re.I)
            for name in (table, other)):
        return None
    query = tree.copy().select(schema.column(other, field, sources[other]), append=False)
    return {"sql": sql_text(query), "kind": "same_name_projection_source",
            "field": field, "from_table": table, "to_table": other}


class Agent:
    def __init__(self, baseline):
        self.baseline = baseline
        self.replacement = None

    def act(self, observation):
        history = observation["history"]
        if self.replacement is not None:
            if passed(history[-1]["observation"]):
                return {"tool": "submit", "sql": self.replacement["sql"]}
            return self.baseline.act(observation)
        action = self.baseline.act(observation)
        if (observation["remaining"] != 1 or action["tool"] != "check"
                or any(event["tool"] == "check" and passed(event["observation"]) for event in history)):
            return action
        candidate = alternative_projection(observation)
        if candidate is None or not usable(candidate["sql"]):
            return action
        if identity(candidate["sql"]) in {identity(action["sql"]), *(
                identity(event["sql"]) for event in history if event["tool"] == "check")}:
            return action
        self.replacement = {**candidate, "displaced": action["sql"]}
        return {"tool": "check", "sql": candidate["sql"]}


def create_agent(checkpoint=None):
    return Agent(create_baseline(checkpoint))
