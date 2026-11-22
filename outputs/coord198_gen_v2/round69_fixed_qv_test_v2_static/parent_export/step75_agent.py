"""Check a course-code filter on its declared relation after an early public match."""
from sqlglot import exp

from step4_agent import identity, passed, usable
from step57_agent import compile_candidate, intent_for
from step63_agent import create_agent as create_baseline


class CourseSuffix:
    def __init__(self, fallback):
        self.fallback = fallback
        self.candidate = None
        self.evidence = None

    def act(self, observation):
        history, remaining = observation["history"], observation["remaining"]
        if self.candidate is not None:
            return {"tool": "submit", "sql": self.candidate} if passed(
                history[-1]["observation"]) else self.fallback
        if not remaining or identity(self.fallback["sql"]) != identity(observation["issue_sql"]):
            return self.fallback
        intent = intent_for(observation)
        if intent is None or intent[0] != "course_code":
            return self.fallback
        tree, relation, atom = intent[1]
        count = tree.expressions[0].unalias()
        if (not isinstance(count.this, exp.Star) or not atom.expression.is_string
                or relation.db or relation.catalog):
            return self.fallback
        tables = next((event["observation"] for event in history if event["tool"] == "schema"), None)
        if tables is None:
            return {"tool": "schema"} if remaining >= 2 else self.fallback
        candidate = compile_candidate(intent, tables)
        if candidate is None or not usable(candidate["sql"]):
            return self.fallback
        self.evidence = candidate
        sql = candidate["sql"]
        checked = next((event["observation"] for event in history if event["tool"] == "check"
                        and identity(event["sql"]) == identity(sql)), None)
        if checked is not None:
            return {"tool": "submit", "sql": sql} if passed(checked) else self.fallback
        self.candidate = sql
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
            self.suffix = CourseSuffix(action)
        return self.suffix.act(observation)


def create_agent(checkpoint=None):
    return Agent(create_baseline(checkpoint))
