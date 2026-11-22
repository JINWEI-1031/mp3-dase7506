"""Change either check order or final selection in the shared round-38 pool."""
import json
from pathlib import Path

from step4_agent import identity, passed
from step4_edits import Schema
from step25_agent import create_agent as create_baseline
from step32_plan import context
from step38.agent import Suffix
from step40.evidence import event_signature, evidence, priority


POLICIES = ("control", "ranking", "selection")


class PairedSuffix(Suffix):
    def __init__(self, fallback, policy):
        super().__init__(fallback, use_new=True, allow_schema=True)
        self.policy = policy

    def act(self, observation):
        action = super().act(observation)
        tool = action["tool"]
        if (self.policy, tool) not in (("ranking", "check"), ("selection", "submit")):
            return action

        visible = context(observation)
        if visible["schema"] is None:
            return action
        chosen = action["sql"]
        if tool == "check":
            previous = {identity(sql) for sql, _ in visible["checks"]}
            pool = [item["sql"] for item in self.audit["pool"]
                    if item["basis"]["eligible"] and identity(item["sql"]) not in previous]
        else:
            pool = list(dict.fromkeys(sql for sql, result in visible["checks"] if passed(result)))
            if identity(chosen) not in {identity(sql) for sql in pool}:
                return action
        if len(pool) < 2:
            return action

        schema = Schema(visible["schema"])
        question = visible["question"]
        headings = [heading.casefold() for heading in visible["columns"]]
        original = self.fallback["sql"]
        reference = event_signature(original, schema, headings, evidence(original, schema, question))

        def score(sql):
            return priority(sql, reference, headings, schema, question, evidence(sql, schema, question))

        ordered = sorted(dict.fromkeys([chosen, *pool]), key=score, reverse=True)
        self.audit["decision_order"] = ordered
        if tool == "check":
            self.audit["checked"] = [ordered[0]]
            self.audit["selected"] = next(p for p in self.audit["pool"] if p["sql"] == ordered[0])
        return {"tool": tool, "sql": ordered[0]}


class Agent:
    def __init__(self, baseline, policy):
        self.baseline = baseline
        self.policy = policy
        self.suffix = None

    def act(self, observation):
        if self.suffix is None:
            action = self.baseline.act(observation)
            if action["tool"] != "submit":
                return action
            self.suffix = PairedSuffix(action, self.policy)
        return self.suffix.act(observation)


def create_agent(checkpoint):
    path = Path(checkpoint).resolve()
    policy = json.loads(path.read_text(encoding="utf-8"))["policy"]
    if policy not in POLICIES:
        raise ValueError(f"Unknown round-42 policy: {policy}")
    return Agent(create_baseline(path.with_name("step25-exists.json")), policy)
