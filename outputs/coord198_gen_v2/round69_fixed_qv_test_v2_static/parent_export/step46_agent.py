"""Spend the last check on a grounded union when the scheduled width is wrong."""
from pathlib import Path

from sqlglot import exp

from repair_rules import parse
from step4_agent import identity, passed
from step25_agent import create_agent as create_baseline
from step26_relations import candidate_pool


def projection_width(sql):
    tree = parse(sql)
    while isinstance(tree, exp.SetOperation):
        tree = tree.this.unnest()
    if not isinstance(tree, exp.Select) or any(column.unalias().is_star for column in tree.expressions):
        return None
    return len(tree.expressions)


class LastCheckAgent:
    def __init__(self, baseline):
        self.baseline = baseline
        self.replacement = None

    def act(self, observation):
        history = observation["history"]
        if self.replacement is not None:
            if passed(history[-1]["observation"]):
                return {"tool": "submit", "sql": self.replacement["candidate"]["sql"]}
            action = self.baseline.act(observation)
            return action if action["tool"] == "submit" else {"tool": "submit", "sql": observation["issue_sql"]}

        action = self.baseline.act(observation)
        if observation["remaining"] != 1 or action["tool"] != "check":
            return action
        if not any(event["tool"] == "schema" for event in history):
            return action
        if any(event["tool"] == "check" and passed(event["observation"]) for event in history):
            return action
        expected = len(observation["example"]["columns"])
        width = projection_width(action["sql"])
        if width is None or width == expected:
            return action
        previous = {identity(event["sql"]) for event in history if event["tool"] == "check"}
        choices = [candidate for candidate in candidate_pool(observation, observation["issue_sql"])
                   if candidate["rule"] == "entity_union" and projection_width(candidate["sql"]) == expected
                   and identity(candidate["sql"]) not in previous]
        if len(choices) != 1:
            return action
        candidate = choices[0]
        self.replacement = {"displaced": action["sql"], "displaced_width": width,
                            "expected_width": expected, "candidate": candidate}
        return {"tool": "check", "sql": candidate["sql"]}


def create_agent(checkpoint=None):
    config = checkpoint or Path(__file__).with_name("configs") / "step25-exists.json"
    return LastCheckAgent(create_baseline(config))
